import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:hand_detection/src/util/image_utils.dart';

/// Validates the shared camera-frame reconstruction (change B): the same
/// cameraFrameToBgrMat path used by the face and pose detectors.
void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('cameraFrameToBgrMat decodes a BGRA frame to BGR (alpha dropped)', () {
    const int w = 64, h = 48;
    final bytes = Uint8List(w * h * 4);
    for (int i = 0; i < w * h; i++) {
      bytes[i * 4] = 10; // B
      bytes[i * 4 + 1] = 20; // G
      bytes[i * 4 + 2] = 30; // R
      bytes[i * 4 + 3] = 255; // A
    }
    final frame = CameraFrame(
      bytes: bytes,
      width: w,
      height: h,
      strideCols: w,
      conversion: CameraFrameConversion.bgra2bgr,
      rotation: null,
    );
    final mat = ImageUtils.cameraFrameToBgrMat(frame);
    try {
      expect(mat.cols, w);
      expect(mat.rows, h);
      expect(mat.channels, 3);
      final px = mat.atPixel(0, 0); // [B, G, R]
      expect(px[0], 10);
      expect(px[1], 20);
      expect(px[2], 30);
    } finally {
      mat.dispose();
    }
  });

  test('cameraFrameToBgrMat applies maxDim downscale', () {
    const int w = 640, h = 480;
    final bytes = Uint8List(w * h * 4)..fillRange(0, w * h * 4, 128);
    final frame = CameraFrame(
      bytes: bytes,
      width: w,
      height: h,
      strideCols: w,
      conversion: CameraFrameConversion.bgra2bgr,
      rotation: null,
    );
    final mat = ImageUtils.cameraFrameToBgrMat(frame, maxDim: 320);
    try {
      expect(mat.cols <= 320 && mat.rows <= 320, isTrue);
      expect(mat.cols, 320); // longest side capped to maxDim
    } finally {
      mat.dispose();
    }
  });
}
