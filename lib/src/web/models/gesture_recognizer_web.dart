// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/src/web/litertjs_interpreter.dart'
    show LiteRtInterpreter;

import '../../shared/hand_types.dart';

/// Web gesture recognizer. Mirrors native [GestureRecognizer]: a two-stage
/// embedder + classifier over hand-landmark vectors, here backed by LiteRT.js.
/// No canvas is involved; inputs are landmark vectors fed directly to the model.
class GestureRecognizerWeb {
  LiteRtInterpreter? _embedder;
  LiteRtInterpreter? _classifier;
  String? _activeAccelerator;

  /// Minimum confidence threshold for returning a gesture.
  final double minConfidence;

  late Float32List _handInput;
  late Float32List _handednessInput;
  late Float32List _worldHandInput;
  Float32List? _embeddingOut;
  Float32List? _gestureOut;
  bool _initialized = false;

  GestureRecognizerWeb({this.minConfidence = 0.5});

  bool get isInitialized => _initialized;
  String? get activeAccelerator =>
      _embedder != null ? _activeAccelerator : null;

  Future<void> initialize({String liteRtAccelerator = 'auto'}) async {
    if (_initialized) await dispose();

    const embedderPath =
        'packages/hand_detection/assets/models/gesture_embedder.tflite';
    const classifierPath =
        'packages/hand_detection/assets/models/canned_gesture_classifier.tflite';
    final embBytes = (await rootBundle.load(embedderPath)).buffer.asUint8List();
    final clsBytes =
        (await rootBundle.load(classifierPath)).buffer.asUint8List();

    final String resolved =
        liteRtAccelerator == 'auto' ? 'webgpu' : liteRtAccelerator;
    _embedder =
        await LiteRtInterpreter.fromBytes(embBytes, accelerator: resolved);
    _classifier =
        await LiteRtInterpreter.fromBytes(clsBytes, accelerator: resolved);
    _activeAccelerator = resolved;

    _handInput = Float32List(21 * 3);
    _handednessInput = Float32List(1);
    _worldHandInput = Float32List(21 * 3);
    _embeddingOut = Float32List(128);
    _gestureOut = Float32List(8);

    _initialized = true;
  }

  Future<void> dispose() async {
    _embedder?.close();
    _embedder = null;
    _classifier?.close();
    _classifier = null;
    _activeAccelerator = null;
    _embeddingOut = null;
    _gestureOut = null;
    _initialized = false;
  }

  Future<GestureResult> recognize({
    required List<HandLandmark> landmarks,
    required List<HandLandmark> worldLandmarks,
    required Handedness? handedness,
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (!_initialized) {
      throw StateError('GestureRecognizerWeb not initialized.');
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

    await _embedder!.runForMultipleInputs(
      <Object>[_handInput, _handednessInput, _worldHandInput],
      <int, Object>{0: _embeddingOut!},
    );
    await _classifier!.runForMultipleInputs(
      <Object>[_embeddingOut!],
      <int, Object>{0: _gestureOut!},
    );

    final probs = _gestureOut!;
    var maxIdx = 0;
    for (int i = 1; i < 8; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }
    final confidence = probs[maxIdx];
    if (confidence < minConfidence) {
      return GestureResult(type: GestureType.unknown, confidence: confidence);
    }
    return GestureResult(
        type: GestureType.values[maxIdx], confidence: confidence);
  }
}
