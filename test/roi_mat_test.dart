import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection/src/native/hand_native_lib.dart';

/// #3 regression: detectFromMat on a non-continuous (ROI) Mat must produce the
/// same detections as on the equivalent continuous image. Previously the ROI's
/// `.data` was read with scrambled stride -> wrong detections.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  final String root = Directory.current.path;
  Uint8List load(String p) => Uint8List.fromList(File(p).readAsBytesSync());

  test('detectFromMat: non-continuous ROI matches the continuous image',
      () async {
    final detector = HandDetector();
    await detector.initializeFromBuffers(
      palmDetectionBytes: load('$root/assets/models/hand_detection.tflite'),
      handLandmarkBytes: load('$root/assets/models/hand_landmark_full.tflite'),
      useCompiledModel: false,
    );
    final img = cv.imdecode(
      load('$root/example/assets/samples/2-hands.png'),
      cv.IMREAD_COLOR,
    );
    // Embed into a wider canvas so an ROI of the original region is
    // non-continuous (its row stride exceeds its width).
    final canvas = cv.copyMakeBorder(
      img,
      10,
      10,
      20,
      30,
      cv.BORDER_CONSTANT,
      value: cv.Scalar.black,
    );
    final roi = canvas.region(cv.Rect(20, 10, img.cols, img.rows));
    expect(roi.isContinuous, isFalse, reason: 'ROI must be non-continuous');

    final fromImg = await detector.detectFromMat(img);
    final fromRoi = await detector.detectFromMat(roi);
    expect(
      jsonEncode(fromRoi.map((h) => h.toMap()).toList()),
      jsonEncode(fromImg.map((h) => h.toMap()).toList()),
      reason: 'ROI detections must match the continuous image (clone guard)',
    );

    img.dispose();
    canvas.dispose();
    roi.dispose();
    await detector.dispose();
  });
}
