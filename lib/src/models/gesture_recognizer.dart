import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_litert/flutter_litert.dart';
import '../types.dart';

/// Gesture recognition using MediaPipe's gesture embedder and classifier models.
///
/// Recognizes 7 hand gestures from hand landmarks:
/// - Closed fist, Open palm, Pointing up, Thumbs down, Thumbs up, Victory, I Love You
///
/// This is a two-stage pipeline:
/// 1. Gesture Embedder: Converts hand landmarks to a 128-dim embedding
/// 2. Gesture Classifier: Classifies the embedding into 8 gesture categories
///
/// Usage:
/// ```dart
/// final recognizer = GestureRecognizer();
/// await recognizer.initialize();
///
/// final gesture = await recognizer.recognize(
///   landmarks: hand.landmarks,
///   worldLandmarks: worldLandmarks,
///   handedness: hand.handedness,
///   imageWidth: image.width,
///   imageHeight: image.height,
/// );
///
/// print('Gesture: ${gesture.type}, confidence: ${gesture.confidence}');
/// await recognizer.dispose();
/// ```
class GestureRecognizer {
  final InterpreterPool _embedderPool = InterpreterPool(poolSize: 1);
  final InterpreterPool _classifierPool = InterpreterPool(poolSize: 1);

  bool _isInitialized = false;

  /// Minimum confidence threshold for returning a gesture.
  /// Gestures below this threshold will return [GestureType.unknown].
  final double minConfidence;

  /// Hand landmarks input tensor [1, 21, 3], stored as flat Float32List.
  /// Passed to TFLite as `.buffer` (a `ByteBuffer`) to skip the boxed-double
  /// allocation that nested-list inputs incur in `Tensor.copyTo`.
  late Float32List _handInput;

  /// Handedness input tensor [1, 1].
  late Float32List _handednessInput;

  /// World-space hand landmarks input tensor [1, 21, 3].
  late Float32List _worldHandInput;

  /// Embedder output tensor [1, 128].
  late Float32List _embeddingOutput;

  /// Classifier output tensor [1, 8] (gesture probabilities).
  late Float32List _gestureOutput;

  /// Creates a gesture recognizer with the specified minimum confidence threshold.
  ///
  /// Parameters:
  /// - [minConfidence]: Minimum confidence (0.0-1.0) for a gesture to be recognized.
  ///   Defaults to 0.5. Set to 0.0 to always return the most likely gesture.
  GestureRecognizer({this.minConfidence = 0.5});

  /// Initializes the gesture recognizer by loading TFLite models.
  ///
  /// Must be called before [recognize].
  Future<void> initialize({PerformanceConfig? performanceConfig}) async {
    if (_isInitialized) await dispose();

    const embedderPath =
        'packages/hand_detection/assets/models/gesture_embedder.tflite';
    const classifierPath =
        'packages/hand_detection/assets/models/canned_gesture_classifier.tflite';

    await _initializeWith(
      performanceConfig: performanceConfig,
      embedderLoader: (options) =>
          Interpreter.fromAsset(embedderPath, options: options),
      classifierLoader: (options) =>
          Interpreter.fromAsset(classifierPath, options: options),
    );
  }

  /// Initializes the gesture recognizer from pre-loaded model bytes.
  ///
  /// Used by [HandDetectorIsolate] to initialize within a background isolate
  /// where Flutter asset loading is not available. Passes
  /// `useIsolateInterpreter: false` to skip the nested IsolateInterpreters
  /// that would otherwise add a per-inference message hop on each pool while
  /// already running inside a worker isolate.
  Future<void> initializeFromBuffers({
    required Uint8List embedderBytes,
    required Uint8List classifierBytes,
    PerformanceConfig? performanceConfig,
  }) async {
    if (_isInitialized) await dispose();

    await _initializeWith(
      performanceConfig: performanceConfig,
      useIsolateInterpreter: false,
      embedderLoader: (options) async =>
          Interpreter.fromBuffer(embedderBytes, options: options),
      classifierLoader: (options) async =>
          Interpreter.fromBuffer(classifierBytes, options: options),
    );
  }

  Future<void> _initializeWith({
    required PerformanceConfig? performanceConfig,
    required Future<Interpreter> Function(InterpreterOptions) embedderLoader,
    required Future<Interpreter> Function(InterpreterOptions) classifierLoader,
    bool useIsolateInterpreter = true,
  }) async {
    await _embedderPool.initialize(
      (options, _) async {
        final interp = await embedderLoader(options);
        interp.allocateTensors();
        return interp;
      },
      performanceConfig: performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );

    await _classifierPool.initialize(
      (options, _) async {
        final interp = await classifierLoader(options);
        interp.allocateTensors();
        return interp;
      },
      performanceConfig: performanceConfig,
      useIsolateInterpreter: useIsolateInterpreter,
    );

    _allocateBuffers();
    _isInitialized = true;
  }

  void _allocateBuffers() {
    _handInput = Float32List(21 * 3);
    _handednessInput = Float32List(1);
    _worldHandInput = Float32List(21 * 3);
    _embeddingOutput = Float32List(128);
    _gestureOutput = Float32List(8);
  }

  /// Returns true if the recognizer has been initialized.
  bool get isInitialized => _isInitialized;

  /// Releases all resources used by the recognizer.
  Future<void> dispose() async {
    await _embedderPool.dispose();
    await _classifierPool.dispose();
    _isInitialized = false;
  }

  /// Recognizes a gesture from hand landmarks.
  ///
  /// Parameters:
  /// - [landmarks]: List of 21 hand landmarks from the landmark model
  /// - [worldLandmarks]: List of 21 world-space landmarks from the landmark model
  /// - [handedness]: Whether this is a left or right hand
  /// - [imageWidth]: Width of the original image (for normalization)
  /// - [imageHeight]: Height of the original image (for normalization)
  ///
  /// Returns a [GestureResult] containing the recognized gesture type and confidence.
  /// Returns [GestureType.unknown] if confidence is below [minConfidence].
  /// The classifier outputs softmax probabilities directly, so no additional softmax is applied.
  /// Model output order matches [GestureType] enum: Unknown, Closed_Fist, Open_Palm, Pointing_Up, Thumb_Down, Thumb_Up, Victory, ILoveYou.
  Future<GestureResult> recognize({
    required List<HandLandmark> landmarks,
    required List<HandLandmark> worldLandmarks,
    required Handedness? handedness,
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (!_isInitialized) {
      throw StateError(
          'GestureRecognizer not initialized. Call initialize() first.');
    }

    if (landmarks.length != 21 || worldLandmarks.length != 21) {
      return const GestureResult(type: GestureType.unknown, confidence: 0.0);
    }

    for (int i = 0; i < 21; i++) {
      final base = i * 3;
      _handInput[base] = landmarks[i].x / imageWidth;
      _handInput[base + 1] = landmarks[i].y / imageHeight;
      _handInput[base + 2] = landmarks[i].z / imageWidth;
    }

    _handednessInput[0] = (handedness == Handedness.right) ? 1.0 : 0.0;

    for (int i = 0; i < 21; i++) {
      final base = i * 3;
      _worldHandInput[base] = worldLandmarks[i].x;
      _worldHandInput[base + 1] = worldLandmarks[i].y;
      _worldHandInput[base + 2] = worldLandmarks[i].z;
    }

    final embedderInputs = <Object>[
      _handInput.buffer,
      _handednessInput.buffer,
      _worldHandInput.buffer,
    ];
    final embedderOutputs = <int, Object>{0: _embeddingOutput.buffer};
    await _embedderPool.withInterpreter((interp, iso) async {
      if (iso != null) {
        await iso.runForMultipleInputs(embedderInputs, embedderOutputs);
      } else {
        interp.runForMultipleInputs(embedderInputs, embedderOutputs);
      }
    });

    final classifierInputs = <Object>[_embeddingOutput.buffer];
    final classifierOutputs = <int, Object>{0: _gestureOutput.buffer};
    await _classifierPool.withInterpreter((interp, iso) async {
      if (iso != null) {
        await iso.runForMultipleInputs(classifierInputs, classifierOutputs);
      } else {
        interp.runForMultipleInputs(classifierInputs, classifierOutputs);
      }
    });

    final probs = _gestureOutput;
    var maxIdx = 0;
    for (int i = 1; i < 8; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }
    final confidence = probs[maxIdx];

    if (confidence < minConfidence) {
      return GestureResult(type: GestureType.unknown, confidence: confidence);
    }

    final gestureType = GestureType.values[maxIdx];

    return GestureResult(type: gestureType, confidence: confidence);
  }
}
