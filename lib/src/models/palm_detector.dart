import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show setEquals;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:meta/meta.dart';
import 'package:flutter_litert/native.dart';
import '../shared/hand_geometry.dart';
import '../util/image_utils.dart';

// The [PalmDetection] result type and the pure-Dart box decoding /
// post-processing live in `shared/hand_geometry.dart` so the web implementation
// can reuse them. Re-exported here so existing `palm_detector.dart` importers
// keep seeing [PalmDetection].
export '../shared/hand_geometry.dart' show PalmDetection;

/// SSD-based palm detector for Stage 1 of the hand detection pipeline.
///
/// Detects palm locations in images using a Single Shot Detector (SSD) architecture
/// with anchor-based decoding. Returns rotation rectangles suitable for cropping
/// hand regions for landmark extraction.
///
/// This is a direct port of the Python PalmDetection class.
class PalmDetector {
  final InterpreterPool _pool = InterpreterPool(poolSize: 1);
  bool _isInitialized = false;

  /// Input dimensions (192x192 for palm detection model).
  late int _inH;
  late int _inW;

  /// Pre-generated SSD anchors.
  late List<List<double>> _anchors;

  /// Preprocessing state - matches Python's calculation.
  /// These use original image dimensions like Python does.
  int _imageHeight = 0;
  int _imageWidth = 0;

  /// square_standard_size = max(image_height, image_width)
  int _squareStandardSize = 0;

  /// square_padding_half_size = abs(image_height - image_width) // 2
  int _squarePaddingHalfSize = 0;

  /// Score threshold for detection filtering.
  final double scoreThreshold;

  /// IoU threshold for the weighted non-maximum suppression that fuses
  /// overlapping palm boxes. Higher keeps more nearby detections.
  final double nmsIouThreshold;

  /// Expansion factor applied to each palm box when building its rotated
  /// square ROI for the landmark model (MediaPipe's detection-to-ROI scale).
  final double roiScale;

  /// Pre-allocated input buffer.
  Float32List? _inputBuffer;

  /// Pre-allocated box regressor output, flat Float32 view of [1, 2016, 18].
  /// Backed by a ByteBuffer that's passed directly to TFLite to avoid the
  /// boxed-double round-trip in flutter_litert's Tensor.copyTo for List dst.
  Float32List? _boxesData;

  /// Pre-allocated classification score output, flat Float32 view of [1, 2016, 1].
  Float32List? _scoresData;

  /// Cached `Float32List` views of the input/output tensor native memory,
  /// captured once after `allocateTensors` and reused every inference on
  /// the direct-invoke path.
  TensorFloat32Views? _views;

  /// Number of values per anchor in the box regressor output (18: cx, cy, w, h
  /// followed by 7 keypoint x/y pairs). Cached after init.
  int _boxStride = 18;

  /// LiteRT Next [CompiledModel] engine (GPU with CPU fallback), used instead
  /// of the [InterpreterPool] when initialized via
  /// [initializeCompiledFromBuffer]. Non-null selects the compiled path.
  CompiledModel? _compiled;

  /// CompiledModel output indices, resolved from output byte sizes at init
  /// (the boxes output is `stride` floats per anchor; scores is one per anchor).
  int _cmBoxesIdx = 0;
  int _cmScoresIdx = 1;

  /// Creates a palm detector with the specified score threshold, NMS IoU
  /// threshold, and ROI expansion scale.
  PalmDetector({
    this.scoreThreshold = 0.45,
    this.nmsIouThreshold = 0.45,
    this.roiScale = 2.6,
  });

  /// Initializes the palm detector by loading the TFLite model.
  Future<void> initialize({PerformanceConfig? performanceConfig}) async {
    const String assetPath =
        'packages/hand_detection/assets/models/hand_detection.tflite';
    await _initWith(
      (options) async => Interpreter.fromAsset(assetPath, options: options),
      performanceConfig,
    );
  }

  /// Initializes the palm detector from pre-loaded model bytes.
  ///
  /// Used by [HandDetectorIsolate] to initialize within a background isolate
  /// where Flutter asset loading is not available. Passes
  /// `useIsolateInterpreter: false` to skip the nested IsolateInterpreter
  /// that would otherwise add a per-inference message hop while already
  /// running inside a worker isolate.
  Future<void> initializeFromBuffer(
    Uint8List modelBytes, {
    PerformanceConfig? performanceConfig,
  }) async {
    await _initWith(
      (options) async => Interpreter.fromBuffer(modelBytes, options: options),
      performanceConfig,
      useIsolateInterpreter: false,
    );
  }

  /// Initializes the palm detector from pre-loaded bytes using the LiteRT Next
  /// [CompiledModel] engine (GPU with CPU fallback) instead of the classic
  /// Interpreter pool.
  ///
  /// Used inside the detection isolate when CompiledModel is requested. Input
  /// size and box stride are derived from the compiled model's tensor byte
  /// sizes. Throws [UnsupportedError] if the model's I/O shapes are not the
  /// expected square-input / anchor-aligned-output palm layout (the caller
  /// falls back to the Interpreter engine).
  Future<void> initializeCompiledFromBuffer(
    Uint8List modelBytes, {
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
    if (_isInitialized) await dispose();
    final compiled =
        setEquals(accelerators, const {Accelerator.gpu, Accelerator.cpu})
            ? CompiledModel.fromBufferWithGpuFallback(modelBytes,
                precision: precision)
            : CompiledModel.fromBuffer(modelBytes,
                accelerators: accelerators, precision: precision);
    try {
      _setupCompiled(compiled);
    } catch (_) {
      compiled.close();
      rethrow;
    }
    _compiled = compiled;
    _isInitialized = true;
  }

  Future<void> _initWith(
    Future<Interpreter> Function(InterpreterOptions) loader,
    PerformanceConfig? performanceConfig, {
    bool useIsolateInterpreter = true,
  }) async {
    if (_isInitialized) await dispose();
    await _pool.initialize(
      (options, _) async {
        final interpreter = await loader(options);
        interpreter.allocateTensors();
        _setupAnchorsAndBuffers(interpreter);
        return interpreter;
      },
      performanceConfig: performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );
    _isInitialized = true;
  }

  void _setupAnchorsAndBuffers(Interpreter interpreter) {
    final inTensor = interpreter.getInputTensor(0);
    final inShape = inTensor.shape;
    _inH = inShape[1];
    _inW = inShape[2];

    _anchors = buildPalmAnchors(_inH, _inW);

    final numAnchors = _anchors.length;

    // Read the box stride from the actual output tensor shape so we don't
    // hardcode 18 (some model variants use a different layout).
    final boxesShape = interpreter.getOutputTensor(0).shape;
    _boxStride = boxesShape.last;

    // Allocate flat Float32 outputs and pass their .buffer (ByteBuffer)
    // to TFLite. This avoids the slow Tensor.copyTo path that otherwise
    // boxes ~38k Doubles per inference.
    _boxesData = Float32List(numAnchors * _boxStride);
    _scoresData = Float32List(numAnchors);

    // Pre-allocate the input buffer eagerly so the first inference doesn't
    // pay an alloc.
    _inputBuffer = Float32List(_inH * _inW * 3);

    // Cache Float32List views into the interpreter's tensor native memory
    // for the direct-invoke path. Tensors are stable after allocateTensors()
    // so the views remain valid for the lifetime of this detector.
    _views = TensorFloat32Views.capture(interpreter);
  }

  /// Derives input size, anchors, and box stride from a [CompiledModel]'s
  /// tensor byte sizes (there is no Interpreter to query shapes from).
  void _setupCompiled(CompiledModel compiled) {
    final int side = compiledSquareInputSide(compiled, label: 'palm detection');
    _inH = side;
    _inW = side;
    _anchors = buildPalmAnchors(_inH, _inW);
    final int numAnchors = _anchors.length;

    if (compiled.outputCount < 2) {
      throw UnsupportedError(
        'Compiled palm detection expects at least two outputs; got '
        '${compiled.outputCount}.',
      );
    }
    final int out0 = compiled.outputByteSizes[0] ~/ 4;
    final int out1 = compiled.outputByteSizes[1] ~/ 4;
    // Scores carry one value per anchor; boxes carry `stride` values per anchor.
    // Resolve the index order from the byte sizes so we do not assume it.
    if (out1 == numAnchors && out0 % numAnchors == 0) {
      _cmBoxesIdx = 0;
      _cmScoresIdx = 1;
      _boxStride = out0 ~/ numAnchors;
    } else if (out0 == numAnchors && out1 % numAnchors == 0) {
      _cmBoxesIdx = 1;
      _cmScoresIdx = 0;
      _boxStride = out1 ~/ numAnchors;
    } else {
      throw UnsupportedError(
        'Compiled palm detection outputs ($out0, $out1 floats) do not align '
        'with $numAnchors anchors.',
      );
    }

    _boxesData = Float32List(numAnchors * _boxStride);
    _scoresData = Float32List(numAnchors);
    _inputBuffer = Float32List(_inH * _inW * 3);
  }

  /// Returns true if the detector has been initialized.
  bool get isInitialized => _isInitialized;

  /// Disposes the detector and releases resources.
  Future<void> dispose() async {
    await _pool.dispose();
    _compiled?.close();
    _compiled = null;
    _inputBuffer = null;
    _boxesData = null;
    _scoresData = null;
    _views = null;
    _isInitialized = false;
  }

  /// Detects palms in the given image.
  ///
  /// Returns a list of [PalmDetection] objects containing rotation rectangle
  /// parameters for each detected palm.
  Future<List<PalmDetection>> detectOnMat(cv.Mat image) async {
    if (!_isInitialized) {
      throw StateError('PalmDetector not initialized.');
    }

    _imageHeight = image.rows;
    _imageWidth = image.cols;

    _squareStandardSize = math.max(_imageHeight, _imageWidth);
    _squarePaddingHalfSize = (_imageHeight - _imageWidth).abs() ~/ 2;

    final (paddedImage, resizedImage) = ImageUtils.keepAspectResizeAndPad(
      image,
      _inW,
      _inH,
    );

    late Float32List boxesView;
    late Float32List scoresView;

    final compiled = _compiled;
    if (compiled != null) {
      // CompiledModel (LiteRT Next) path: preprocess into the scratch input
      // buffer, then runAsync returns fresh Float32List outputs we decode
      // directly. Same [0,1] RGB preprocessing as the Interpreter path.
      final input = _inputBuffer!;
      ImageUtils.matToFloat32Tensor(paddedImage, buffer: input);
      final outputs = await compiled.runAsync([input]);
      boxesView = outputs[_cmBoxesIdx];
      scoresView = outputs[_cmScoresIdx];
    } else {
      await _pool.withInterpreter((interp, iso) async {
        if (iso != null) {
          // IsolateInterpreter path: must go through runForMultipleInputs.
          // Convert into our scratch _inputBuffer first, then ship its
          // ByteBuffer to the iso. Outputs land in _boxesData / _scoresData
          // via the ByteBuffer fast branch of Tensor.copyTo.
          ImageUtils.matToFloat32Tensor(paddedImage, buffer: _inputBuffer);
          await iso.runForMultipleInputs(
            [_inputBuffer!.buffer],
            <int, Object>{
              0: _boxesData!.buffer,
              1: _scoresData!.buffer,
            },
          );
          boxesView = _boxesData!;
          scoresView = _scoresData!;
        } else {
          // Direct path (no nested isolate): write the BGR→RGB normalized
          // tensor straight into the input tensor's native memory, then
          // invoke() and read output tensors as Float32List views, no
          // runForMultipleInputs, no Tensor.copyTo, no marshalling. The
          // tensor views are cached at init so there is no per-inference
          // wrapper allocation here.
          final views = _views!;
          ImageUtils.matToFloat32Tensor(paddedImage, buffer: views.inputs[0]);

          interp.invoke();

          boxesView = views.outputs[0];
          scoresView = views.outputs[1];
        }
      });
    }

    resizedImage.dispose();
    paddedImage.dispose();

    final decodedBoxes = decodePalmBoxes(
      boxesView,
      scoresView,
      _anchors,
      _boxStride,
      scoreThreshold,
    );
    return postprocessPalms(
      decodedBoxes,
      imageWidth: _imageWidth,
      imageHeight: _imageHeight,
      squareStandardSize: _squareStandardSize,
      squarePaddingHalfSize: _squarePaddingHalfSize,
      roiScale: roiScale,
      iouThreshold: nmsIouThreshold,
    );
  }

  /// Exposes anchor generation for testing.
  @visibleForTesting
  List<List<double>> get anchorsForTest => _anchors;

  /// Exposes input width for testing.
  @visibleForTesting
  int get inputWidth => _inW;

  /// Exposes input height for testing.
  @visibleForTesting
  int get inputHeight => _inH;
}
