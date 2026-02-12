import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import 'image_utils.dart';
import 'types.dart';

/// A single interpreter instance with its associated resources.
///
/// Encapsulates a TensorFlow Lite interpreter and its isolate wrapper,
/// allowing for clean resource management in the interpreter pool.
/// Also holds pre-allocated input/output buffers to avoid GC pressure.
class _InterpreterInstance {
  final Interpreter interpreter;
  final IsolateInterpreter? isolateInterpreter;

  // Pre-allocated input buffer as flat Float32List [1 * 224 * 224 * 3]
  final Float32List inputBuffer;

  // Pre-allocated output buffers - matches Python hand landmark model outputs
  final List<List<double>>
      outputLandmarks; // [1, 63] - 21 landmarks × 3 (x, y, z)
  final List<List<double>> outputScore; // [1, 1] - hand confidence score
  final List<List<double>> outputHandedness; // [1, 1] - 0=left, 1=right
  final List<List<double>>
      outputWorldLandmarks; // [1, 63] - 21 world landmarks × 3

  _InterpreterInstance({
    required this.interpreter,
    required this.isolateInterpreter,
    required this.inputBuffer,
    required this.outputLandmarks,
    required this.outputScore,
    required this.outputHandedness,
    required this.outputWorldLandmarks,
  });

  /// Disposes interpreter and isolate wrapper.
  Future<void> dispose() async {
    isolateInterpreter?.close();
    interpreter.close();
  }
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
  /// Pool of interpreter instances for parallel processing.
  final List<_InterpreterInstance> _interpreterPool = [];

  /// Maximum number of concurrent inferences.
  final int _poolSize;

  /// Delegate instances - one per interpreter (XNNPACK is NOT thread-safe for sharing).
  final List<Delegate> _delegates = [];

  /// Serialization locks to prevent concurrent inference on the same interpreter.
  final List<Future<void>> _interpreterLocks = [];

  /// Round-robin counter for interpreter selection.
  int _poolCounter = 0;

  bool _isInitialized = false;

  /// Input dimensions (224x224 for MediaPipe hand landmark model).
  static const int inputSize = 224;

  /// Creates a landmark model runner with the specified pool size.
  HandLandmarkModelRunner({int poolSize = 1})
      : _poolSize = poolSize.clamp(1, 10);

  /// Initializes the hand landmark model with the specified variant.
  ///
  /// Creates a pool of interpreter instances based on the configured [_poolSize].
  /// Each interpreter is loaded independently, allowing for parallel inference execution.
  ///
  /// Parameters:
  /// - [model]: Which hand landmark variant to use (lite, full, or heavy)
  /// - [performanceConfig]: Optional performance configuration for TFLite delegates.
  Future<void> initialize(
    HandLandmarkModel model, {
    PerformanceConfig? performanceConfig,
  }) async {
    if (_isInitialized) await dispose();

    final String path = _getModelPath(model);

    // Create pool of interpreter instances
    for (int i = 0; i < _poolSize; i++) {
      final (options, delegate) = _createInterpreterOptions(performanceConfig);
      if (delegate != null) {
        _delegates.add(delegate);
      }

      final interpreter = await Interpreter.fromAsset(path, options: options);
      interpreter.resizeInputTensor(0, [1, inputSize, inputSize, 3]);
      interpreter.allocateTensors();

      final isolateInterpreter = delegate == null
          ? await IsolateInterpreter.create(address: interpreter.address)
          : null;

      // Pre-allocate input buffer as flat Float32List [1 * 224 * 224 * 3]
      final inputBuffer = Float32List(inputSize * inputSize * 3);

      // Pre-allocate output buffers matching Python model outputs
      // Output 0: [1, 63] - 21 landmarks × 3 (x, y, z)
      // Output 1: [1, 1] - hand confidence score
      // Output 2: [1, 1] - handedness (0=left, 1=right)
      // Output 3: [1, 63] - 21 world landmarks × 3 (x, y, z)
      final outputLandmarks = [List<double>.filled(63, 0.0, growable: false)];
      final outputScore = [List<double>.filled(1, 0.0, growable: false)];
      final outputHandedness = [List<double>.filled(1, 0.0, growable: false)];
      final outputWorldLandmarks = [
        List<double>.filled(63, 0.0, growable: false)
      ];

      _interpreterPool.add(_InterpreterInstance(
        interpreter: interpreter,
        isolateInterpreter: isolateInterpreter,
        inputBuffer: inputBuffer,
        outputLandmarks: outputLandmarks,
        outputScore: outputScore,
        outputHandedness: outputHandedness,
        outputWorldLandmarks: outputWorldLandmarks,
      ));

      // Initialize serialization lock for this interpreter
      _interpreterLocks.add(Future.value());
    }

    _isInitialized = true;
  }

  /// Creates interpreter options with delegates based on performance configuration.
  (InterpreterOptions, Delegate?) _createInterpreterOptions(
      PerformanceConfig? config) {
    final options = InterpreterOptions();

    if (config == null || config.mode == PerformanceMode.disabled) {
      return (options, null);
    }

    final threadCount = config.numThreads?.clamp(0, 8) ??
        math.min(4, Platform.numberOfProcessors);

    options.threads = threadCount;

    if (config.mode == PerformanceMode.xnnpack ||
        config.mode == PerformanceMode.auto) {
      try {
        final xnnpackDelegate = XNNPackDelegate(
          options: XNNPackDelegateOptions(numThreads: threadCount),
        );
        options.addDelegate(xnnpackDelegate);
        return (options, xnnpackDelegate);
      } catch (e) {
        // Graceful fallback
      }
    }

    return (options, null);
  }

  String _getModelPath(HandLandmarkModel model) {
    // Only full model is available to match Python implementation
    return 'packages/hand_detection_tflite/assets/models/hand_landmark_full.tflite';
  }

  /// Returns true if the model runner has been initialized and is ready to use.
  bool get isInitialized => _isInitialized;

  /// Returns the configured pool size.
  int get poolSize => _poolSize;

  /// Disposes the model runner and releases all resources.
  Future<void> dispose() async {
    for (final instance in _interpreterPool) {
      await instance.dispose();
    }
    _interpreterPool.clear();

    for (final delegate in _delegates) {
      delegate.delete();
    }
    _delegates.clear();

    _interpreterLocks.clear();
    _isInitialized = false;
  }

  /// Serializes inference calls on a specific interpreter to prevent race conditions.
  Future<T> _withInterpreterLock<T>(
      Future<T> Function(_InterpreterInstance) fn) async {
    if (_interpreterPool.isEmpty) {
      throw StateError('Interpreter pool is empty. Call initialize() first.');
    }

    // Round-robin selection
    final int poolIndex = _poolCounter % _interpreterPool.length;
    _poolCounter = (_poolCounter + 1) % _interpreterPool.length;

    final previous = _interpreterLocks[poolIndex];
    final completer = Completer<void>();
    _interpreterLocks[poolIndex] = completer.future;

    try {
      await previous;
      return await fn(_interpreterPool[poolIndex]);
    } finally {
      completer.complete();
    }
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

    return await _withInterpreterLock((instance) async {
      // Use keep_aspect_resize_and_pad to match Python implementation
      final (paddedImage, resizedImage) = ImageUtils.keepAspectResizeAndPad(
        roiImage,
        inputSize,
        inputSize,
      );

      // Calculate padding info for coordinate transformation
      // resize_scale = resized_dim / original_dim (how much we scaled down)
      final resizeScaleH = resizedImage.rows / roiImage.rows;
      final resizeScaleW = resizedImage.cols / roiImage.cols;
      final padH = paddedImage.rows - resizedImage.rows;
      final padW = paddedImage.cols - resizedImage.cols;
      final halfPadH = math.max(0, padH ~/ 2).toDouble();
      final halfPadW = math.max(0, padW ~/ 2).toDouble();

      // Convert to flat Float32List tensor with normalization (BGR -> RGB)
      ImageUtils.matToFloat32Tensor(paddedImage, buffer: instance.inputBuffer);

      // Clean up OpenCV Mats
      resizedImage.dispose();
      paddedImage.dispose();

      // Run inference using IsolateInterpreter for thread safety,
      // or direct invocation when delegates are active
      final outputs = {
        0: instance.outputLandmarks,
        1: instance.outputScore,
        2: instance.outputHandedness,
        3: instance.outputWorldLandmarks,
      };
      if (instance.isolateInterpreter != null) {
        await instance.isolateInterpreter!.runForMultipleInputs(
          [instance.inputBuffer.buffer],
          outputs,
        );
      } else {
        instance.interpreter.runForMultipleInputs(
          [instance.inputBuffer.buffer],
          outputs,
        );
      }

      // Parse landmarks and transform to original crop pixel space
      // This matches Python's postprocessing: (raw - half_pad) / resize_scale
      return _parseLandmarks(
        instance.outputLandmarks,
        instance.outputWorldLandmarks,
        instance.outputScore,
        instance.outputHandedness,
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
  /// The model outputs:
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
    List<List<double>> landmarksData,
    List<List<double>> worldLandmarksData,
    List<List<double>> scoreData,
    List<List<double>> handednessData, {
    required double halfPadW,
    required double halfPadH,
    required double resizeScaleW,
    required double resizeScaleH,
    required int cropWidth,
    required int cropHeight,
  }) {
    // Apply sigmoid to score
    final rawScore = scoreData[0][0];
    final score = 1.0 / (1.0 + math.exp(-rawScore));

    // Determine handedness (>0.5 = right hand)
    final rawHandedness = handednessData[0][0];
    final handedness = rawHandedness > 0.5 ? Handedness.right : Handedness.left;

    // Parse 21 landmarks and transform to original crop pixel space
    // Match Python exactly: first normalize by input_h, then multiply by input_w/input_h
    final raw = landmarksData[0];
    final landmarks = <HandLandmark>[];

    for (int i = 0; i < numHandLandmarks; i++) {
      final base = i * 3;

      // Transform from 224x224 padded space to original crop pixel space
      // Python: rrn_lms = rrn_lms / input_h (normalize all coords by 224)
      // Python: rescaled_xy[:, 0] = (rescaled_xy[:, 0] * input_w - half_pad[0]) / resize_scale[0]
      // Python: rescaled_xy[:, 1] = (rescaled_xy[:, 1] * input_h - half_pad[1]) / resize_scale[1]
      final normalizedX = raw[base] / inputSize;
      final normalizedY = raw[base + 1] / inputSize;
      final x = (normalizedX * inputSize - halfPadW) / resizeScaleW;
      final y = (normalizedY * inputSize - halfPadH) / resizeScaleH;
      final z = raw[base + 2]; // Z is relative depth, keep as-is

      // Clamp to crop bounds
      landmarks.add(HandLandmark(
        type: HandLandmarkType.values[i],
        x: x.clamp(0.0, cropWidth.toDouble()),
        y: y.clamp(0.0, cropHeight.toDouble()),
        z: z,
        visibility: score, // Use overall score as visibility
      ));
    }

    // Parse 21 world landmarks (used for gesture recognition)
    // World landmarks are in a normalized 3D coordinate system relative to the hand
    // They don't need transformation - use raw values directly
    final rawWorld = worldLandmarksData[0];
    final worldLandmarks = <HandLandmark>[];

    for (int i = 0; i < numHandLandmarks; i++) {
      final base = i * 3;
      worldLandmarks.add(HandLandmark(
        type: HandLandmarkType.values[i],
        x: rawWorld[base],
        y: rawWorld[base + 1],
        z: rawWorld[base + 2],
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
