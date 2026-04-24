import 'dart:async';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../util/image_utils.dart';
import '../types.dart';

/// Pre-allocated inference buffers for one pool slot.
///
/// Avoids GC pressure by reusing the same buffers across invocations.
/// Each [InterpreterPool] slot has its own [_HandBuffers] instance.
///
/// Outputs are stored as flat [Float32List]s and passed to TFLite as their
/// underlying [ByteBuffer], which avoids the boxed-double allocation that
/// `Tensor.copyTo` performs when handed a nested `List<List<double>>` dst.
///
/// [views] caches [Float32List] views directly into the interpreter's
/// tensor native memory, used by the direct-invoke path (i.e. when running
/// inside an outer isolate and no nested IsolateInterpreter is active).
class _HandBuffers {
  final Float32List inputBuffer;
  final Float32List outputLandmarks;
  final Float32List outputScore;
  final Float32List outputHandedness;
  final Float32List outputWorldLandmarks;

  final TensorFloat32Views views;

  _HandBuffers({
    required this.inputBuffer,
    required this.outputLandmarks,
    required this.outputScore,
    required this.outputHandedness,
    required this.outputWorldLandmarks,
    required this.views,
  });
}

/// Hand landmark extraction model runner for Stage 2 of the hand detection pipeline.
///
/// Extracts 21 landmarks from hand crops using the MediaPipe hand landmark model.
/// Supports three model variants (lite, full, heavy) with different accuracy/performance trade-offs.
///
/// This is a port of the Python HandLandmark class that uses 224x224 input and
/// outputs 21 3D landmarks plus handedness detection.
///
/// **Interpreter Pool Architecture:**
/// To enable parallel processing of multiple hands, this runner maintains a pool of
/// TensorFlow Lite interpreter instances using a **round-robin selection pattern**.
class HandLandmarkModelRunner {
  final InterpreterPool _pool;

  /// Per-slot pre-allocated buffers, keyed by interpreter identity.
  final Map<Interpreter, _HandBuffers> _buffers = {};

  bool _isInitialized = false;

  /// Input dimensions (224x224 for MediaPipe hand landmark model).
  static const int inputSize = 224;

  /// Creates a landmark model runner with the specified pool size.
  HandLandmarkModelRunner({int poolSize = 1})
      : _pool = InterpreterPool(poolSize: poolSize);

  /// Initializes the hand landmark model.
  ///
  /// Creates a pool of interpreter instances based on the configured [poolSize].
  /// Each interpreter is loaded independently, allowing for parallel inference execution.
  ///
  /// Parameters:
  /// - [performanceConfig]: Optional performance configuration for TFLite delegates.
  Future<void> initialize({
    PerformanceConfig? performanceConfig,
  }) async {
    if (_isInitialized) await dispose();

    final String path =
        'packages/hand_detection/assets/models/hand_landmark_full.tflite';

    await _initializePool(
      performanceConfig: performanceConfig,
      loader: (options) async {
        final interpreter = await Interpreter.fromAsset(path, options: options);
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
    );
  }

  /// Initializes the hand landmark model from pre-loaded model bytes.
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
    if (_isInitialized) await dispose();

    await _initializePool(
      performanceConfig: performanceConfig,
      useIsolateInterpreter: false,
      loader: (options) async {
        final interpreter =
            Interpreter.fromBuffer(modelBytes, options: options);
        interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
        interpreter.allocateTensors();
        return interpreter;
      },
    );
  }

  Future<void> _initializePool({
    required PerformanceConfig? performanceConfig,
    required Future<Interpreter> Function(InterpreterOptions) loader,
    bool useIsolateInterpreter = true,
  }) async {
    await _pool.initialize(
      (options, _) => loader(options),
      performanceConfig: performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );
    _allocateBuffers();
    _isInitialized = true;
  }

  void _allocateBuffers() {
    _buffers.clear();
    for (final interp in _pool.interpreters) {
      _buffers[interp] = _HandBuffers(
        inputBuffer: Float32List(inputSize * inputSize * 3),
        outputLandmarks: Float32List(63),
        outputScore: Float32List(1),
        outputHandedness: Float32List(1),
        outputWorldLandmarks: Float32List(63),
        views: TensorFloat32Views.capture(interp),
      );
    }
  }

  /// Returns true if the model runner has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Returns the configured pool size.
  int get poolSize => _pool.poolSize;

  /// Disposes the model runner and releases all resources.
  Future<void> dispose() async {
    await _pool.dispose();
    _buffers.clear();
    _isInitialized = false;
  }

  /// Runs landmark extraction on a hand crop image.
  ///
  /// Extracts 21 landmarks from the input hand crop using the MediaPipe hand landmark model.
  /// The input image should be a cropped and rotated hand region from the palm detector.
  ///
  /// Parameters:
  /// - [roiImage]: Cropped hand image (will be resized to 224x224 internally)
  ///
  /// Returns [HandLandmarks] containing 21 landmarks with coordinates in the
  /// original crop image pixel space (matching Python's postprocessing),
  /// a confidence score, and handedness (left/right).
  Future<HandLandmarks> run(cv.Mat roiImage) async {
    if (!_isInitialized) {
      throw StateError(
          'HandLandmarkModelRunner not initialized. Call initialize() first.');
    }

    return _pool.withInterpreter((interp, iso) async {
      final buf = _buffers[interp]!;

      final (paddedImage, resizedImage) = ImageUtils.keepAspectResizeAndPad(
        roiImage,
        inputSize,
        inputSize,
      );

      final resizeScaleH = resizedImage.rows / roiImage.rows;
      final resizeScaleW = resizedImage.cols / roiImage.cols;
      final padH = paddedImage.rows - resizedImage.rows;
      final padW = paddedImage.cols - resizedImage.cols;
      final halfPadH = math.max(0, padH ~/ 2).toDouble();
      final halfPadW = math.max(0, padW ~/ 2).toDouble();

      late Float32List landmarksView;
      late Float32List worldLandmarksView;
      late Float32List scoreView;
      late Float32List handednessView;

      if (iso != null) {
        // IsolateInterpreter path: use scratch buffers and runForMultipleInputs.
        ImageUtils.matToFloat32Tensor(paddedImage, buffer: buf.inputBuffer);
        await iso.runForMultipleInputs(
          [buf.inputBuffer.buffer],
          <int, Object>{
            0: buf.outputLandmarks.buffer,
            1: buf.outputScore.buffer,
            2: buf.outputHandedness.buffer,
            3: buf.outputWorldLandmarks.buffer,
          },
        );
        landmarksView = buf.outputLandmarks;
        worldLandmarksView = buf.outputWorldLandmarks;
        scoreView = buf.outputScore;
        handednessView = buf.outputHandedness;
      } else {
        // Direct path: write straight into the input tensor's native memory,
        // invoke(), then read outputs as Float32List views, no marshalling.
        // Tensor views were cached at init so nothing is allocated here.
        ImageUtils.matToFloat32Tensor(paddedImage, buffer: buf.views.inputs[0]);

        interp.invoke();

        landmarksView = buf.views.outputs[0];
        scoreView = buf.views.outputs[1];
        handednessView = buf.views.outputs[2];
        worldLandmarksView = buf.views.outputs[3];
      }

      resizedImage.dispose();
      paddedImage.dispose();

      return _parseLandmarks(
        landmarksView,
        worldLandmarksView,
        scoreView,
        handednessView,
        halfPadW: halfPadW,
        halfPadH: halfPadH,
        resizeScaleW: resizeScaleW,
        resizeScaleH: resizeScaleH,
        cropWidth: roiImage.cols,
        cropHeight: roiImage.rows,
      );
    });
  }

  /// Parses model outputs into HandLandmarks.
  ///
  /// The model outputs (read from flat Float32List buffers):
  /// - landmarks: [1, 63] - 21 points × 3 (x, y, z) in 224x224 space
  /// - worldLandmarks: [1, 63] - 21 world-space points × 3 (x, y, z)
  /// - score: [1, 1] - hand confidence (0-1 after sigmoid)
  /// - handedness: [1, 1] - 0=left, 1=right
  ///
  /// Transforms landmarks from 224x224 padded space to original crop pixel space
  /// using the exact formula from Python hand_landmark.py:
  /// rrn_lms = rrn_lms / input_h
  /// rescaled_xy[:, 0] = (rescaled_xy[:, 0] * input_w - half_pad_size[0]) / resize_scale[0]
  /// rescaled_xy[:, 1] = (rescaled_xy[:, 1] * input_h - half_pad_size[1]) / resize_scale[1]
  HandLandmarks _parseLandmarks(
    Float32List landmarksData,
    Float32List worldLandmarksData,
    Float32List scoreData,
    Float32List handednessData, {
    required double halfPadW,
    required double halfPadH,
    required double resizeScaleW,
    required double resizeScaleH,
    required int cropWidth,
    required int cropHeight,
  }) {
    final score = sigmoidClipped(scoreData[0]);
    final handedness =
        handednessData[0] > 0.5 ? Handedness.right : Handedness.left;

    final landmarks = <HandLandmark>[];
    for (int i = 0; i < numHandLandmarks; i++) {
      final base = i * 3;
      landmarks.add(HandLandmark(
        type: HandLandmarkType.values[i],
        x: ((landmarksData[base] - halfPadW) / resizeScaleW)
            .clamp(0.0, cropWidth.toDouble()),
        y: ((landmarksData[base + 1] - halfPadH) / resizeScaleH)
            .clamp(0.0, cropHeight.toDouble()),
        z: landmarksData[base + 2],
        visibility: score,
      ));
    }

    final worldLandmarks = <HandLandmark>[];
    for (int i = 0; i < numHandLandmarks; i++) {
      final base = i * 3;
      worldLandmarks.add(HandLandmark(
        type: HandLandmarkType.values[i],
        x: worldLandmarksData[base],
        y: worldLandmarksData[base + 1],
        z: worldLandmarksData[base + 2],
        visibility: score,
      ));
    }

    return HandLandmarks(
      landmarks: landmarks,
      worldLandmarks: worldLandmarks,
      score: score,
      handedness: handedness,
    );
  }
}
