import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/src/models/gesture_recognizer.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:hand_detection/src/types.dart';

/// Creates 21 dummy landmarks at origin with zero visibility.
List<HandLandmark> _createDummyLandmarks() {
  return List.generate(
    21,
    (i) => HandLandmark(
      type: HandLandmarkType.values[i],
      x: 0.0,
      y: 0.0,
      z: 0.0,
      visibility: 0.5,
    ),
  );
}

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('GestureRecognizer constructor', () {
    test('default minConfidence is 0.5', () {
      final recognizer = GestureRecognizer();
      expect(recognizer.minConfidence, 0.5);
    });

    test('custom minConfidence', () {
      final recognizer = GestureRecognizer(minConfidence: 0.8);
      expect(recognizer.minConfidence, 0.8);
    });

    test('isInitialized is false before initialization', () {
      final recognizer = GestureRecognizer();
      expect(recognizer.isInitialized, false);
    });
  });

  group('GestureRecognizer lifecycle', () {
    test('throws StateError when recognize called before init', () {
      final recognizer = GestureRecognizer();

      expect(
        () => recognizer.recognize(
          landmarks: _createDummyLandmarks(),
          worldLandmarks: _createDummyLandmarks(),
          handedness: Handedness.right,
          imageWidth: 640,
          imageHeight: 480,
        ),
        throwsA(isA<StateError>()),
      );
    });

    test('dispose before init does not throw', () async {
      final recognizer = GestureRecognizer();
      await recognizer.dispose();
      expect(recognizer.isInitialized, false);
    });

    test('initializeFromBuffers sets isInitialized', () async {
      final embedderBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/gesture_embedder.tflite')
            .readAsBytesSync(),
      );
      final classifierBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/canned_gesture_classifier.tflite')
            .readAsBytesSync(),
      );

      final recognizer = GestureRecognizer();
      await recognizer.initializeFromBuffers(
        embedderBytes: embedderBytes,
        classifierBytes: classifierBytes,
      );

      expect(recognizer.isInitialized, true);
      await recognizer.dispose();
      expect(recognizer.isInitialized, false);
    });

    test('re-initialization disposes previous state', () async {
      final embedderBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/gesture_embedder.tflite')
            .readAsBytesSync(),
      );
      final classifierBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/canned_gesture_classifier.tflite')
            .readAsBytesSync(),
      );

      final recognizer = GestureRecognizer();
      await recognizer.initializeFromBuffers(
        embedderBytes: embedderBytes,
        classifierBytes: classifierBytes,
      );
      expect(recognizer.isInitialized, true);

      await recognizer.initializeFromBuffers(
        embedderBytes: embedderBytes,
        classifierBytes: classifierBytes,
      );
      expect(recognizer.isInitialized, true);

      await recognizer.dispose();
    });

    test('multiple disposes do not throw', () async {
      final embedderBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/gesture_embedder.tflite')
            .readAsBytesSync(),
      );
      final classifierBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/canned_gesture_classifier.tflite')
            .readAsBytesSync(),
      );

      final recognizer = GestureRecognizer();
      await recognizer.initializeFromBuffers(
        embedderBytes: embedderBytes,
        classifierBytes: classifierBytes,
      );
      await recognizer.dispose();
      await recognizer.dispose();
      expect(recognizer.isInitialized, false);
    });
  });

  group('GestureRecognizer inference', () {
    late GestureRecognizer recognizer;

    setUpAll(() async {
      final embedderBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/gesture_embedder.tflite')
            .readAsBytesSync(),
      );
      final classifierBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/canned_gesture_classifier.tflite')
            .readAsBytesSync(),
      );

      recognizer = GestureRecognizer(minConfidence: 0.3);
      await recognizer.initializeFromBuffers(
        embedderBytes: embedderBytes,
        classifierBytes: classifierBytes,
      );
    });

    tearDownAll(() async {
      await recognizer.dispose();
    });

    test('returns GestureResult for valid landmarks', () async {
      final result = await recognizer.recognize(
        landmarks: _createDummyLandmarks(),
        worldLandmarks: _createDummyLandmarks(),
        handedness: Handedness.right,
        imageWidth: 640,
        imageHeight: 480,
      );

      expect(result, isNotNull);
      expect(result.confidence, greaterThanOrEqualTo(0.0));
      expect(result.confidence, lessThanOrEqualTo(1.0));
      expect(GestureType.values.contains(result.type), true);
    });

    test('returns unknown for wrong landmark count', () async {
      // Only 10 landmarks instead of 21
      final shortLandmarks = _createDummyLandmarks().sublist(0, 10);

      final result = await recognizer.recognize(
        landmarks: shortLandmarks,
        worldLandmarks: _createDummyLandmarks(),
        handedness: Handedness.right,
        imageWidth: 640,
        imageHeight: 480,
      );

      expect(result.type, GestureType.unknown);
      expect(result.confidence, 0.0);
    });

    test('returns unknown for wrong world landmark count', () async {
      final result = await recognizer.recognize(
        landmarks: _createDummyLandmarks(),
        worldLandmarks: _createDummyLandmarks().sublist(0, 5),
        handedness: Handedness.right,
        imageWidth: 640,
        imageHeight: 480,
      );

      expect(result.type, GestureType.unknown);
      expect(result.confidence, 0.0);
    });

    test('works with null handedness', () async {
      final result = await recognizer.recognize(
        landmarks: _createDummyLandmarks(),
        worldLandmarks: _createDummyLandmarks(),
        handedness: null,
        imageWidth: 640,
        imageHeight: 480,
      );

      // Should not crash and should return a valid result
      expect(result, isNotNull);
      expect(GestureType.values.contains(result.type), true);
    });

    test('works with left hand', () async {
      final result = await recognizer.recognize(
        landmarks: _createDummyLandmarks(),
        worldLandmarks: _createDummyLandmarks(),
        handedness: Handedness.left,
        imageWidth: 640,
        imageHeight: 480,
      );

      expect(result, isNotNull);
    });

    test('sequential calls produce results', () async {
      for (int i = 0; i < 3; i++) {
        final result = await recognizer.recognize(
          landmarks: _createDummyLandmarks(),
          worldLandmarks: _createDummyLandmarks(),
          handedness: Handedness.right,
          imageWidth: 640,
          imageHeight: 480,
        );

        expect(result, isNotNull);
      }
    });
  });

  group('GestureRecognizer confidence thresholding', () {
    test('high threshold returns unknown for low confidence gestures',
        () async {
      final embedderBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/gesture_embedder.tflite')
            .readAsBytesSync(),
      );
      final classifierBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/canned_gesture_classifier.tflite')
            .readAsBytesSync(),
      );

      final strictRecognizer = GestureRecognizer(minConfidence: 0.99);
      await strictRecognizer.initializeFromBuffers(
        embedderBytes: embedderBytes,
        classifierBytes: classifierBytes,
      );

      final result = await strictRecognizer.recognize(
        landmarks: _createDummyLandmarks(),
        worldLandmarks: _createDummyLandmarks(),
        handedness: Handedness.right,
        imageWidth: 640,
        imageHeight: 480,
      );

      // With dummy data and 0.99 threshold, should likely be unknown
      // (confidence below threshold returns unknown)
      if (result.confidence < 0.99) {
        expect(result.type, GestureType.unknown);
      }

      await strictRecognizer.dispose();
    });

    test('zero threshold always returns a gesture type', () async {
      final embedderBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/gesture_embedder.tflite')
            .readAsBytesSync(),
      );
      final classifierBytes = Uint8List.fromList(
        File('${Directory.current.path}/assets/models/canned_gesture_classifier.tflite')
            .readAsBytesSync(),
      );

      final lenientRecognizer = GestureRecognizer(minConfidence: 0.0);
      await lenientRecognizer.initializeFromBuffers(
        embedderBytes: embedderBytes,
        classifierBytes: classifierBytes,
      );

      final result = await lenientRecognizer.recognize(
        landmarks: _createDummyLandmarks(),
        worldLandmarks: _createDummyLandmarks(),
        handedness: Handedness.right,
        imageWidth: 640,
        imageHeight: 480,
      );

      // With minConfidence=0.0, any confidence passes threshold
      // so we should get a non-unknown gesture (or at least a defined type)
      expect(result, isNotNull);
      expect(result.confidence, greaterThanOrEqualTo(0.0));

      await lenientRecognizer.dispose();
    });
  });
}
