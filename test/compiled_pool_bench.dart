import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection/src/isolate/hand_detector_core.dart';
import 'package:hand_detection/src/types.dart';

/// Manual benchmark (not a CI test — run explicitly:
///   POOL=1 flutter test test/compiled_pool_bench.dart
/// Times the CompiledModel 2-hand detect path. POOL env var sets the landmark
/// CompiledModel pool size.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final String root = Directory.current.path;
  Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());
  final int pool = int.tryParse(Platform.environment['POOL'] ?? '1') ?? 1;

  test('CM 2-hand detect latency (pool=$pool)', () async {
    final palm = load('$root/assets/models/hand_detection.tflite');
    final lm = load('$root/assets/models/hand_landmark_full.tflite');
    cv.Mat img;
    try {
      img = cv.imdecode(
          load('$root/example/assets/samples/2-hands.png'), cv.IMREAD_COLOR);
    } catch (_) {
      markTestSkipped('sample image missing');
      return;
    }

    final core = HandDetectorCore();
    try {
      await core.initializeFromBuffers(
        palmDetectionBytes: palm,
        handLandmarkBytes: lm,
        mode: HandMode.boxesAndLandmarks,
        maxDetections: 4,
        minLandmarkScore: 0.5,
        detectorConf: 0.5,
        interpreterPoolSize: pool,
        performanceConfig: const PerformanceConfig(),
        enableGestures: false,
        gestureMinConfidence: 0.5,
        useCompiledModel: true,
      );
    } catch (e) {
      markTestSkipped('CompiledModel unavailable: $e');
      img.dispose();
      return;
    }

    int hands = 0;
    for (int i = 0; i < 8; i++) {
      hands = (await core.detectDirect(img)).length;
    }

    const int n = 60;
    final List<int> us = [];
    for (int i = 0; i < n; i++) {
      final sw = Stopwatch()..start();
      await core.detectDirect(img);
      sw.stop();
      us.add(sw.elapsedMicroseconds);
    }
    us.sort();
    double ms(int u) => u / 1000.0;
    final mean = us.reduce((a, b) => a + b) / n;
    // ignore: avoid_print
    print('[pool-bench] pool=$pool hands=$hands n=$n '
        'mean=${ms(mean.round()).toStringAsFixed(2)}ms '
        'median=${ms(us[n ~/ 2]).toStringAsFixed(2)}ms '
        'p10=${ms(us[n ~/ 10]).toStringAsFixed(2)}ms '
        'p90=${ms(us[(n * 9) ~/ 10]).toStringAsFixed(2)}ms');

    await core.dispose();
    img.dispose();
  }, timeout: const Timeout(Duration(minutes: 4)));
}
