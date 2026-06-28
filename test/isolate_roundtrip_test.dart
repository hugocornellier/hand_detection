import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/src/native/hand_native_lib.dart';

/// End-to-end test of the background-isolate RPC path that now runs on the
/// shared `serveIsolateRpc` server: spawn the isolate, run a real `detect`
/// through the handler, and tear down via `disposeGracefully` (the dispose ack).
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final String root = Directory.current.path;
  Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());

  test('HandDetector full isolate round-trip detects hands and disposes',
      () async {
    final detector = HandDetector();
    await detector.initializeFromBuffers(
      palmDetectionBytes: load('$root/assets/models/hand_detection.tflite'),
      handLandmarkBytes: load('$root/assets/models/hand_landmark_full.tflite'),
      mode: HandMode.boxesAndLandmarks,
      maxDetections: 4,
      detectorConf: 0.5,
      performanceConfig: const PerformanceConfig(),
    );
    expect(detector.isReady, isTrue);

    // Drives serveIsolateRpc's 'detect' handler over the real isolate.
    final hands = await detector.detect(
      load('$root/example/assets/samples/2-hands.png'),
    );
    expect(hands, isNotEmpty);
    expect(hands.first.landmarks, isNotEmpty);

    // Drives disposeGracefully -> serveIsolateRpc dispose ack.
    await detector.dispose();
    expect(detector.isReady, isFalse);
  });
}
