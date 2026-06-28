import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/src/native/hand_native_lib.dart';

/// Manual golden harness (untracked; persists across `git checkout`). Runs the
/// full e2e detect on the deterministic Interpreter path and dumps every
/// detection (toMap) as JSON to $GOLDEN_OUT, for before/after result-parity.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  test('golden parity dump', () async {
    final root = Directory.current.path;
    Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());
    final detector = HandDetector();
    await detector.initializeFromBuffers(
      palmDetectionBytes: load('$root/assets/models/hand_detection.tflite'),
      handLandmarkBytes: load('$root/assets/models/hand_landmark_full.tflite'),
      maxDetections: 4,
      detectorConf: 0.5,
      useCompiledModel: false,
    );
    final results =
        await detector.detect(load('$root/example/assets/samples/2-hands.png'));
    final out = Platform.environment['GOLDEN_OUT'] ?? '/tmp/golden_hand.json';
    File(out).writeAsStringSync(
      jsonEncode(results.map((h) => h.toMap()).toList()),
    );
    stderr.writeln('[golden] hand: ${results.length} detections -> $out');
    await detector.dispose();
  });
}
