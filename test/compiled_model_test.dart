import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection/src/models/palm_detector.dart';
import 'package:hand_detection/src/isolate/hand_detector_core.dart';
import 'package:hand_detection/src/types.dart';

/// Validates the LiteRT Next [CompiledModel] engine added in the hand_detection
/// dual-engine work. These tests run the compiled path end to end and compare
/// it against the classic Interpreter path.
///
/// The compiled runtime is not guaranteed to be present in every host test
/// environment (it needs the LiteRT native library / a usable accelerator). The
/// tests probe for it once and self-skip when it is unavailable, so the suite
/// stays green on hosts without it while still validating on CI/devices that
/// have it.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  final String root = Directory.current.path;
  final String palmModelPath = '$root/assets/models/hand_detection.tflite';
  final String landmarkModelPath =
      '$root/assets/models/hand_landmark_full.tflite';
  final String embedderModelPath =
      '$root/assets/models/gesture_embedder.tflite';
  final String classifierModelPath =
      '$root/assets/models/canned_gesture_classifier.tflite';
  final String imagePath = '$root/example/assets/samples/2-hands.png';

  Uint8List loadBytes(String p) =>
      Uint8List.fromList(File(p).readAsBytesSync());

  bool? cmAvail;
  bool compiledAvailable() {
    if (cmAvail != null) return cmAvail!;
    try {
      final m =
          CompiledModel.fromBufferWithGpuFallback(loadBytes(palmModelPath));
      m.close();
      cmAvail = true;
    } catch (_) {
      cmAvail = false;
    }
    return cmAvail!;
  }

  group('PalmDetector CompiledModel engine', () {
    test('compiled palm detection produces valid detections', () async {
      if (!compiledAvailable()) {
        markTestSkipped('LiteRT CompiledModel runtime unavailable on host');
        return;
      }
      final mat = cv.imdecode(loadBytes(imagePath), cv.IMREAD_COLOR);

      final itp = PalmDetector();
      await itp.initializeFromBuffer(loadBytes(palmModelPath));
      final itpPalms = await itp.detectOnMat(mat);
      await itp.dispose();

      final cm = PalmDetector();
      await cm.initializeCompiledFromBuffer(loadBytes(palmModelPath));
      final cmPalms = await cm.detectOnMat(mat);
      await cm.dispose();

      mat.dispose();

      // The 2-hands fixture must yield palms on both engines, and the compiled
      // count should track the interpreter count closely (small differences are
      // allowed for GPU floating-point variance near NMS thresholds).
      expect(itpPalms, isNotEmpty);
      expect(cmPalms, isNotEmpty);
      expect((cmPalms.length - itpPalms.length).abs(), lessThanOrEqualTo(1));
      for (final p in cmPalms) {
        expect(p.score, inInclusiveRange(0.0, 1.0));
        expect(p.sqnRrSize, greaterThan(0.0));
      }
    });
  });

  group('HandDetectorCore CompiledModel engine', () {
    test('compiled full pipeline detects hands with 21 landmarks', () async {
      if (!compiledAvailable()) {
        markTestSkipped('LiteRT CompiledModel runtime unavailable on host');
        return;
      }
      final mat = cv.imdecode(loadBytes(imagePath), cv.IMREAD_COLOR);

      final core = HandDetectorCore();
      await core.initializeFromBuffers(
        palmDetectionBytes: loadBytes(palmModelPath),
        handLandmarkBytes: loadBytes(landmarkModelPath),
        mode: HandMode.boxesAndLandmarks,
        maxDetections: 10,
        minLandmarkScore: 0.5,
        detectorConf: 0.45,
        interpreterPoolSize: 1,
        performanceConfig: const PerformanceConfig(),
        enableGestures: false,
        gestureMinConfidence: 0.5,
        useCompiledModel: true,
      );
      final hands = await core.detectDirect(mat);
      await core.dispose();
      mat.dispose();

      expect(hands, isNotEmpty);
      for (final h in hands) {
        expect(h.landmarks.length, 21);
        expect(h.score, inInclusiveRange(0.0, 1.0));
      }
    });

    test('compiled pipeline runs gesture recognition (embedder + classifier)',
        () async {
      if (!compiledAvailable()) {
        markTestSkipped('LiteRT CompiledModel runtime unavailable on host');
        return;
      }
      final mat = cv.imdecode(loadBytes(imagePath), cv.IMREAD_COLOR);

      final core = HandDetectorCore();
      await core.initializeFromBuffers(
        palmDetectionBytes: loadBytes(palmModelPath),
        handLandmarkBytes: loadBytes(landmarkModelPath),
        gestureEmbedderBytes: loadBytes(embedderModelPath),
        gestureClassifierBytes: loadBytes(classifierModelPath),
        mode: HandMode.boxesAndLandmarks,
        maxDetections: 10,
        minLandmarkScore: 0.5,
        detectorConf: 0.45,
        interpreterPoolSize: 1,
        performanceConfig: const PerformanceConfig(),
        enableGestures: true,
        gestureMinConfidence: 0.0,
        useCompiledModel: true,
      );
      final hands = await core.detectDirect(mat);
      await core.dispose();
      mat.dispose();

      // With gestures enabled and a 0.0 threshold, every detected hand should
      // carry a gesture result produced by the compiled embedder + classifier.
      expect(hands, isNotEmpty);
      for (final h in hands) {
        expect(h.gesture, isNotNull);
        expect(h.gesture!.confidence, inInclusiveRange(0.0, 1.0));
      }
    });
  });
}
