import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
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
  Interpreter? _embedderInterpreter;
  Interpreter? _classifierInterpreter;
  IsolateInterpreter? _embedderIso;
  IsolateInterpreter? _classifierIso;
  Delegate? _embedderDelegate;
  Delegate? _classifierDelegate;

  bool _isInitialized = false;

  /// Minimum confidence threshold for returning a gesture.
  /// Gestures below this threshold will return [GestureType.unknown].
  final double minConfidence;

  // Pre-allocated input buffers for embedder
  // Input 0: hand [1, 21, 3] - normalized landmarks
  late List<List<List<double>>> _handInput;
  // Input 1: handedness [1, 1]
  late List<List<double>> _handednessInput;
  // Input 2: world_hand [1, 21, 3] - world coordinates
  late List<List<List<double>>> _worldHandInput;

  // Pre-allocated output buffer for embedder
  // Output: [1, 128] embedding
  late List<List<double>> _embeddingOutput;

  // Pre-allocated output buffer for classifier
  // Output: [1, 8] gesture probabilities
  late List<List<double>> _gestureOutput;

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

    // Load embedder model
    const embedderPath =
        'packages/hand_detection/assets/models/gesture_embedder.tflite';
    final embedderOptions = _createInterpreterOptions(performanceConfig, true);
    _embedderInterpreter =
        await Interpreter.fromAsset(embedderPath, options: embedderOptions);
    _embedderInterpreter!.allocateTensors();
    if (_embedderDelegate == null) {
      _embedderIso = await IsolateInterpreter.create(
          address: _embedderInterpreter!.address);
    }

    // Load classifier model
    const classifierPath =
        'packages/hand_detection/assets/models/canned_gesture_classifier.tflite';
    final classifierOptions =
        _createInterpreterOptions(performanceConfig, false);
    _classifierInterpreter =
        await Interpreter.fromAsset(classifierPath, options: classifierOptions);
    _classifierInterpreter!.allocateTensors();
    if (_classifierDelegate == null) {
      _classifierIso = await IsolateInterpreter.create(
          address: _classifierInterpreter!.address);
    }

    // Pre-allocate input buffers for embedder
    // hand: [1, 21, 3]
    _handInput = List.generate(
      1,
      (_) => List.generate(
        21,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    // handedness: [1, 1]
    _handednessInput = List.generate(
      1,
      (_) => List<double>.filled(1, 0.0, growable: false),
      growable: false,
    );

    // world_hand: [1, 21, 3]
    _worldHandInput = List.generate(
      1,
      (_) => List.generate(
        21,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    // Pre-allocate output buffers
    // Embedding: [1, 128]
    _embeddingOutput = List.generate(
      1,
      (_) => List<double>.filled(128, 0.0, growable: false),
      growable: false,
    );

    // Gesture probabilities: [1, 8]
    _gestureOutput = List.generate(
      1,
      (_) => List<double>.filled(8, 0.0, growable: false),
      growable: false,
    );

    _isInitialized = true;
  }

  /// Initializes the gesture recognizer from pre-loaded model bytes.
  ///
  /// Used by [HandDetectorIsolate] to initialize within a background isolate
  /// where Flutter asset loading is not available.
  Future<void> initializeFromBuffers({
    required Uint8List embedderBytes,
    required Uint8List classifierBytes,
    PerformanceConfig? performanceConfig,
  }) async {
    if (_isInitialized) await dispose();

    // Load embedder model
    final embedderOptions = _createInterpreterOptions(performanceConfig, true);
    _embedderInterpreter =
        Interpreter.fromBuffer(embedderBytes, options: embedderOptions);
    _embedderInterpreter!.allocateTensors();
    if (_embedderDelegate == null) {
      _embedderIso = await IsolateInterpreter.create(
          address: _embedderInterpreter!.address);
    }

    // Load classifier model
    final classifierOptions =
        _createInterpreterOptions(performanceConfig, false);
    _classifierInterpreter =
        Interpreter.fromBuffer(classifierBytes, options: classifierOptions);
    _classifierInterpreter!.allocateTensors();
    if (_classifierDelegate == null) {
      _classifierIso = await IsolateInterpreter.create(
          address: _classifierInterpreter!.address);
    }

    // Pre-allocate input buffers for embedder
    _handInput = List.generate(
      1,
      (_) => List.generate(
        21,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    _handednessInput = List.generate(
      1,
      (_) => List<double>.filled(1, 0.0, growable: false),
      growable: false,
    );

    _worldHandInput = List.generate(
      1,
      (_) => List.generate(
        21,
        (_) => List<double>.filled(3, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    _embeddingOutput = List.generate(
      1,
      (_) => List<double>.filled(128, 0.0, growable: false),
      growable: false,
    );

    _gestureOutput = List.generate(
      1,
      (_) => List<double>.filled(8, 0.0, growable: false),
      growable: false,
    );

    _isInitialized = true;
  }

  InterpreterOptions _createInterpreterOptions(
      PerformanceConfig? config, bool isEmbedder) {
    final options = InterpreterOptions();

    if (config == null || config.mode == PerformanceMode.disabled) {
      return options;
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
        if (isEmbedder) {
          _embedderDelegate = xnnpackDelegate;
        } else {
          _classifierDelegate = xnnpackDelegate;
        }
      } catch (_) {
        // Fallback to CPU
      }
    }

    return options;
  }

  /// Returns true if the recognizer has been initialized.
  bool get isInitialized => _isInitialized;

  /// Releases all resources used by the recognizer.
  Future<void> dispose() async {
    _embedderIso?.close();
    _embedderIso = null;
    _classifierIso?.close();
    _classifierIso = null;

    _embedderInterpreter?.close();
    _embedderInterpreter = null;
    _classifierInterpreter?.close();
    _classifierInterpreter = null;

    _embedderDelegate?.delete();
    _embedderDelegate = null;
    _classifierDelegate?.delete();
    _classifierDelegate = null;

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

    // Prepare normalized hand landmarks [1, 21, 3]
    // Normalize x, y, z to 0-1 range
    // z represents depth relative to wrist, scaled roughly like x (by image width)
    for (int i = 0; i < 21; i++) {
      _handInput[0][i][0] = landmarks[i].x / imageWidth;
      _handInput[0][i][1] = landmarks[i].y / imageHeight;
      _handInput[0][i][2] = landmarks[i].z / imageWidth;
    }

    // Prepare handedness [1, 1]
    // MediaPipe convention: 0 = left, 1 = right
    _handednessInput[0][0] = (handedness == Handedness.right) ? 1.0 : 0.0;

    // Prepare world landmarks [1, 21, 3]
    // World landmarks are already in world coordinates, use as-is
    for (int i = 0; i < 21; i++) {
      _worldHandInput[0][i][0] = worldLandmarks[i].x;
      _worldHandInput[0][i][1] = worldLandmarks[i].y;
      _worldHandInput[0][i][2] = worldLandmarks[i].z;
    }

    // Stage 1: Run embedder
    // Inputs: hand (index 0), handedness (index 1), world_hand (index 2)
    if (_embedderIso != null) {
      await _embedderIso!.runForMultipleInputs(
        [_handInput, _handednessInput, _worldHandInput],
        {0: _embeddingOutput},
      );
    } else {
      _embedderInterpreter!.runForMultipleInputs(
        [_handInput, _handednessInput, _worldHandInput],
        {0: _embeddingOutput},
      );
    }

    // Stage 2: Run classifier
    if (_classifierIso != null) {
      await _classifierIso!.run(_embeddingOutput, _gestureOutput);
    } else {
      _classifierInterpreter!.run(_embeddingOutput, _gestureOutput);
    }

    // Find the gesture with highest probability
    // Note: The classifier already outputs softmax probabilities (sum to 1.0),
    // so we use the raw output directly without applying softmax again.
    final probs = _gestureOutput[0];
    int maxIdx = 0;
    double maxProb = probs[0];
    for (int i = 1; i < 8; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i];
        maxIdx = i;
      }
    }

    final confidence = maxProb;

    // Return unknown if confidence is below threshold
    if (confidence < minConfidence) {
      return GestureResult(type: GestureType.unknown, confidence: confidence);
    }

    // Map index to gesture type
    // Model output order: Unknown, Closed_Fist, Open_Palm, Pointing_Up,
    //                     Thumb_Down, Thumb_Up, Victory, ILoveYou
    final gestureType = GestureType.values[maxIdx];

    return GestureResult(type: gestureType, confidence: confidence);
  }
}
