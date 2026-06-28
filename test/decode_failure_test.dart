import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/src/native/hand_native_lib.dart';

/// Verifies the documented `FormatException` on undecodable bytes (#2 fix —
/// previously this surfaced an opaque `StateError: Infinity or NaN toInt`).
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final String root = Directory.current.path;
  Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());

  test('detect() throws FormatException on undecodable bytes', () async {
    final detector = HandDetector();
    await detector.initializeFromBuffers(
      palmDetectionBytes: load('$root/assets/models/hand_detection.tflite'),
      handLandmarkBytes: load('$root/assets/models/hand_landmark_full.tflite'),
      useCompiledModel: false,
    );
    await expectLater(
      detector.detect(Uint8List.fromList([1, 2, 3, 4, 5, 6, 7, 8])),
      throwsA(isA<FormatException>()),
    );
    await detector.dispose();
  });
}
