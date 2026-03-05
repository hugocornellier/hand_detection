import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection/src/models/hand_landmark_model.dart';
import 'package:hand_detection/hand_detection.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('HandLandmarkModelRunner constructor', () {
    test('default pool size is 1', () {
      final runner = HandLandmarkModelRunner();
      expect(runner.poolSize, 1);
    });

    test('custom pool size', () {
      final runner = HandLandmarkModelRunner(poolSize: 4);
      expect(runner.poolSize, 4);
    });

    test('pool size is clamped to minimum 1', () {
      final runner = HandLandmarkModelRunner(poolSize: 0);
      expect(runner.poolSize, 1);

      final runner2 = HandLandmarkModelRunner(poolSize: -5);
      expect(runner2.poolSize, 1);
    });

    test('pool size is clamped to maximum 10', () {
      final runner = HandLandmarkModelRunner(poolSize: 20);
      expect(runner.poolSize, 10);
    });

    test('isInitialized is false before initialization', () {
      final runner = HandLandmarkModelRunner();
      expect(runner.isInitialized, false);
    });

    test('inputSize constant is 224', () {
      expect(HandLandmarkModelRunner.inputSize, 224);
    });
  });

  group('HandLandmarkModelRunner lifecycle', () {
    test('throws StateError when run called before init', () {
      final runner = HandLandmarkModelRunner();
      final mat = cv.Mat.zeros(224, 224, cv.MatType.CV_8UC3);

      try {
        expect(
          () => runner.run(mat),
          throwsA(isA<StateError>()),
        );
      } finally {
        mat.dispose();
      }
    });

    test('dispose before init does not throw', () async {
      final runner = HandLandmarkModelRunner();
      await runner.dispose();
      expect(runner.isInitialized, false);
    });

    test('initializeFromBuffer sets isInitialized to true', () async {
      final modelPath =
          '${Directory.current.path}/assets/models/hand_landmark_full.tflite';
      final modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());

      final runner = HandLandmarkModelRunner();
      await runner.initializeFromBuffer(modelBytes);

      expect(runner.isInitialized, true);
      await runner.dispose();
      expect(runner.isInitialized, false);
    });

    test('re-initialization disposes previous state', () async {
      final modelPath =
          '${Directory.current.path}/assets/models/hand_landmark_full.tflite';
      final modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());

      final runner = HandLandmarkModelRunner();
      await runner.initializeFromBuffer(modelBytes);
      expect(runner.isInitialized, true);

      // Re-initialize
      await runner.initializeFromBuffer(modelBytes);
      expect(runner.isInitialized, true);

      await runner.dispose();
    });

    test('multiple disposes do not throw', () async {
      final modelPath =
          '${Directory.current.path}/assets/models/hand_landmark_full.tflite';
      final modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());

      final runner = HandLandmarkModelRunner();
      await runner.initializeFromBuffer(modelBytes);
      await runner.dispose();
      await runner.dispose();
      expect(runner.isInitialized, false);
    });
  });

  group('HandLandmarkModelRunner pool sizes', () {
    late Uint8List modelBytes;

    setUpAll(() {
      final modelPath =
          '${Directory.current.path}/assets/models/hand_landmark_full.tflite';
      modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());
    });

    test('pool size 1 initializes correctly', () async {
      final runner = HandLandmarkModelRunner(poolSize: 1);
      await runner.initializeFromBuffer(modelBytes);

      expect(runner.isInitialized, true);
      expect(runner.poolSize, 1);

      await runner.dispose();
    });

    test('pool size 2 initializes correctly', () async {
      final runner = HandLandmarkModelRunner(poolSize: 2);
      await runner.initializeFromBuffer(modelBytes);

      expect(runner.isInitialized, true);
      expect(runner.poolSize, 2);

      await runner.dispose();
    });

    test('pool size 4 initializes correctly', () async {
      final runner = HandLandmarkModelRunner(poolSize: 4);
      await runner.initializeFromBuffer(modelBytes);

      expect(runner.isInitialized, true);
      expect(runner.poolSize, 4);

      await runner.dispose();
    });
  });

  group('HandLandmarkModelRunner inference', () {
    late HandLandmarkModelRunner runner;
    late Uint8List modelBytes;

    setUpAll(() async {
      final modelPath =
          '${Directory.current.path}/assets/models/hand_landmark_full.tflite';
      modelBytes = Uint8List.fromList(File(modelPath).readAsBytesSync());

      runner = HandLandmarkModelRunner(poolSize: 2);
      await runner.initializeFromBuffer(modelBytes);
    });

    tearDownAll(() async {
      await runner.dispose();
    });

    test('run returns 21 landmarks', () async {
      final mat = cv.Mat.zeros(224, 224, cv.MatType.CV_8UC3);

      try {
        final result = await runner.run(mat);
        expect(result.landmarks.length, 21);
        expect(result.worldLandmarks.length, 21);
      } finally {
        mat.dispose();
      }
    });

    test('run returns valid score between 0 and 1', () async {
      final mat = cv.Mat.zeros(224, 224, cv.MatType.CV_8UC3);

      try {
        final result = await runner.run(mat);
        expect(result.score, greaterThanOrEqualTo(0.0));
        expect(result.score, lessThanOrEqualTo(1.0));
      } finally {
        mat.dispose();
      }
    });

    test('run returns handedness', () async {
      final mat = cv.Mat.zeros(224, 224, cv.MatType.CV_8UC3);

      try {
        final result = await runner.run(mat);
        expect(result.handedness, isNotNull);
        expect(
          result.handedness,
          anyOf(Handedness.left, Handedness.right),
        );
      } finally {
        mat.dispose();
      }
    });

    test('landmarks have correct types in order', () async {
      final mat = cv.Mat.zeros(224, 224, cv.MatType.CV_8UC3);

      try {
        final result = await runner.run(mat);

        for (int i = 0; i < 21; i++) {
          expect(result.landmarks[i].type, HandLandmarkType.values[i]);
          expect(result.worldLandmarks[i].type, HandLandmarkType.values[i]);
        }
      } finally {
        mat.dispose();
      }
    });

    test('handles non-square input image', () async {
      // Non-square images should be handled by keepAspectResizeAndPad
      final mat = cv.Mat.zeros(100, 200, cv.MatType.CV_8UC3);

      try {
        final result = await runner.run(mat);
        expect(result.landmarks.length, 21);
      } finally {
        mat.dispose();
      }
    });

    test('sequential calls produce results (round-robin pool)', () async {
      final mat1 = cv.Mat.zeros(224, 224, cv.MatType.CV_8UC3);
      final mat2 = cv.Mat.zeros(224, 224, cv.MatType.CV_8UC3);

      try {
        // With poolSize=2, these should use different interpreters
        final result1 = await runner.run(mat1);
        final result2 = await runner.run(mat2);

        expect(result1.landmarks.length, 21);
        expect(result2.landmarks.length, 21);
      } finally {
        mat1.dispose();
        mat2.dispose();
      }
    });

    test('landmarks are clamped to crop bounds', () async {
      final mat = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);

      try {
        final result = await runner.run(mat);

        for (final lm in result.landmarks) {
          expect(lm.x, greaterThanOrEqualTo(0.0));
          expect(lm.x, lessThanOrEqualTo(100.0));
          expect(lm.y, greaterThanOrEqualTo(0.0));
          expect(lm.y, lessThanOrEqualTo(100.0));
        }
      } finally {
        mat.dispose();
      }
    });

    test('concurrent runs succeed without error', () async {
      final mats = List.generate(
        4,
        (_) => cv.Mat.zeros(224, 224, cv.MatType.CV_8UC3),
      );

      try {
        // Run concurrently via Future.wait
        final futures = mats.map((m) => runner.run(m)).toList();
        final results = await Future.wait(futures);

        for (final result in results) {
          expect(result.landmarks.length, 21);
        }
      } finally {
        for (final m in mats) {
          m.dispose();
        }
      }
    });
  });
}
