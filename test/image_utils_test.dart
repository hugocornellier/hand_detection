import 'dart:math' as math;
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:hand_detection/src/util/image_utils.dart';
import 'package:hand_detection/src/models/palm_detector.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('keepAspectResizeAndPad', () {
    test('portrait image padded to square', () {
      final source = cv.Mat.zeros(200, 100, cv.MatType.CV_8UC3);
      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(source, 192, 192);

      try {
        expect(padded.cols, 192);
        expect(padded.rows, 192);
        // Width-constrained: 100:200 ratio, target 192 -> 96x192
        expect(resized.cols, 96);
        expect(resized.rows, 192);
      } finally {
        source.dispose();
        padded.dispose();
        resized.dispose();
      }
    });

    test('landscape image padded to square', () {
      final source = cv.Mat.zeros(100, 200, cv.MatType.CV_8UC3);
      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(source, 192, 192);

      try {
        expect(padded.cols, 192);
        expect(padded.rows, 192);
        // Height-constrained: 200:100 ratio, target 192 -> 192x96
        expect(resized.cols, 192);
        expect(resized.rows, 96);
      } finally {
        source.dispose();
        padded.dispose();
        resized.dispose();
      }
    });

    test('square image requires no padding', () {
      final source = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);
      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(source, 192, 192);

      try {
        expect(padded.cols, 192);
        expect(padded.rows, 192);
        expect(resized.cols, 192);
        expect(resized.rows, 192);
      } finally {
        source.dispose();
        padded.dispose();
        resized.dispose();
      }
    });

    test('non-square target dimensions', () {
      final source = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);
      final (padded, resized) =
          ImageUtils.keepAspectResizeAndPad(source, 224, 192);

      try {
        expect(padded.cols, 224);
        expect(padded.rows, 192);
      } finally {
        source.dispose();
        padded.dispose();
        resized.dispose();
      }
    });
  });

  group('rotateAndCropRectangle', () {
    test('returns cropped image for valid palm detection', () {
      final image = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC3);
      final palm = PalmDetection(
        sqnRrSize: 0.3,
        rotation: 0.0,
        sqnRrCenterX: 0.5,
        sqnRrCenterY: 0.5,
        score: 0.9,
      );

      final result = ImageUtils.rotateAndCropRectangle(image, palm);

      try {
        expect(result, isNotNull);
        final expectedSize = (0.3 * math.max(640, 480)).round();
        expect(result!.cols, expectedSize);
        expect(result.rows, expectedSize);
      } finally {
        result?.dispose();
        image.dispose();
      }
    });

    test('returns null for zero-size crop', () {
      final image = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC3);
      final palm = PalmDetection(
        sqnRrSize: 0.0,
        rotation: 0.0,
        sqnRrCenterX: 0.5,
        sqnRrCenterY: 0.5,
        score: 0.9,
      );

      final result = ImageUtils.rotateAndCropRectangle(image, palm);
      expect(result, isNull);
      image.dispose();
    });

    test('handles rotation', () {
      final image = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC3);
      final palm = PalmDetection(
        sqnRrSize: 0.3,
        rotation: math.pi / 4, // 45 degrees
        sqnRrCenterX: 0.5,
        sqnRrCenterY: 0.5,
        score: 0.9,
      );

      final result = ImageUtils.rotateAndCropRectangle(image, palm);

      try {
        expect(result, isNotNull);
        // Output should still be square
        expect(result!.cols, result.rows);
      } finally {
        result?.dispose();
        image.dispose();
      }
    });

    test('handles crop near image edge', () {
      final image = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC3);
      final palm = PalmDetection(
        sqnRrSize: 0.3,
        rotation: 0.0,
        sqnRrCenterX: 0.05, // Near left edge
        sqnRrCenterY: 0.05, // Near top edge
        score: 0.9,
      );

      // Should not crash - border mode handles out-of-bounds
      final result = ImageUtils.rotateAndCropRectangle(image, palm);

      try {
        expect(result, isNotNull);
      } finally {
        result?.dispose();
        image.dispose();
      }
    });

    test('handles crop near bottom-right edge', () {
      final image = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC3);
      final palm = PalmDetection(
        sqnRrSize: 0.3,
        rotation: 0.0,
        sqnRrCenterX: 0.95,
        sqnRrCenterY: 0.95,
        score: 0.9,
      );

      final result = ImageUtils.rotateAndCropRectangle(image, palm);

      try {
        expect(result, isNotNull);
      } finally {
        result?.dispose();
        image.dispose();
      }
    });

    test('handles large rotation angle', () {
      final image = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC3);
      final palm = PalmDetection(
        sqnRrSize: 0.2,
        rotation: math.pi, // 180 degrees
        sqnRrCenterX: 0.5,
        sqnRrCenterY: 0.5,
        score: 0.9,
      );

      final result = ImageUtils.rotateAndCropRectangle(image, palm);

      try {
        expect(result, isNotNull);
      } finally {
        result?.dispose();
        image.dispose();
      }
    });

    test('works with padding=false', () {
      final image = cv.Mat.zeros(480, 640, cv.MatType.CV_8UC3);
      final palm = PalmDetection(
        sqnRrSize: 0.3,
        rotation: 0.0,
        sqnRrCenterX: 0.5,
        sqnRrCenterY: 0.5,
        score: 0.9,
      );

      final result =
          ImageUtils.rotateAndCropRectangle(image, palm, padding: false);

      try {
        expect(result, isNotNull);
      } finally {
        result?.dispose();
        image.dispose();
      }
    });
  });

  group('palmToRect', () {
    test('converts center coordinates to pixel space', () {
      final palm = PalmDetection(
        sqnRrSize: 0.5,
        rotation: 0.0,
        sqnRrCenterX: 0.5,
        sqnRrCenterY: 0.5,
        score: 0.9,
      );

      final rect = ImageUtils.palmToRect(palm, 640, 480);
      expect(rect[0], closeTo(320.0, 0.1)); // cx
      expect(rect[1], closeTo(240.0, 0.1)); // cy
      expect(rect[2], closeTo(320.0, 0.1)); // width = 0.5 * max(640,480)
      expect(rect[3], closeTo(320.0, 0.1)); // height = same as width
      expect(rect[4], closeTo(0.0, 0.001)); // angle degrees
    });

    test('converts rotation to degrees', () {
      final palm = PalmDetection(
        sqnRrSize: 0.3,
        rotation: math.pi / 2, // 90 degrees
        sqnRrCenterX: 0.5,
        sqnRrCenterY: 0.5,
        score: 0.9,
      );

      final rect = ImageUtils.palmToRect(palm, 640, 480);
      expect(rect[4], closeTo(90.0, 0.1));
    });

    test('corner position', () {
      final palm = PalmDetection(
        sqnRrSize: 0.1,
        rotation: 0.0,
        sqnRrCenterX: 0.0,
        sqnRrCenterY: 0.0,
        score: 0.9,
      );

      final rect = ImageUtils.palmToRect(palm, 640, 480);
      expect(rect[0], closeTo(0.0, 0.1)); // cx at origin
      expect(rect[1], closeTo(0.0, 0.1)); // cy at origin
    });
  });

  group('letterbox', () {
    test('produces exact target dimensions', () {
      final src = cv.Mat.zeros(100, 200, cv.MatType.CV_8UC3);
      final ratioOut = <double>[];
      final dwdhOut = <int>[];

      final result = ImageUtils.letterbox(src, 300, 300, ratioOut, dwdhOut);

      try {
        expect(result.cols, 300);
        expect(result.rows, 300);
        expect(ratioOut.length, 1);
        expect(dwdhOut.length, 2);
      } finally {
        src.dispose();
        result.dispose();
      }
    });

    test('preserves aspect ratio', () {
      final src = cv.Mat.zeros(100, 200, cv.MatType.CV_8UC3);
      final ratioOut = <double>[];
      final dwdhOut = <int>[];

      final result = ImageUtils.letterbox(src, 300, 300, ratioOut, dwdhOut);

      try {
        // Ratio should be min(300/100, 300/200) = min(3, 1.5) = 1.5
        expect(ratioOut[0], closeTo(1.5, 0.01));
        // dw = (300 - 200*1.5) / 2 = (300 - 300) / 2 = 0
        expect(dwdhOut[0], 0);
        // dh = (300 - 100*1.5) / 2 = (300 - 150) / 2 = 75
        expect(dwdhOut[1], 75);
      } finally {
        src.dispose();
        result.dispose();
      }
    });

    test('square image to larger square', () {
      final src = cv.Mat.zeros(50, 50, cv.MatType.CV_8UC3);
      final ratioOut = <double>[];
      final dwdhOut = <int>[];

      final result = ImageUtils.letterbox(src, 200, 200, ratioOut, dwdhOut);

      try {
        expect(result.cols, 200);
        expect(result.rows, 200);
        expect(ratioOut[0], closeTo(4.0, 0.01));
        expect(dwdhOut[0], 0);
        expect(dwdhOut[1], 0);
      } finally {
        src.dispose();
        result.dispose();
      }
    });

    test('clears and reuses output lists', () {
      final src = cv.Mat.zeros(100, 100, cv.MatType.CV_8UC3);
      final ratioOut = <double>[999.0, 888.0];
      final dwdhOut = <int>[77, 66, 55];

      final result = ImageUtils.letterbox(src, 200, 200, ratioOut, dwdhOut);

      try {
        // Should have cleared old values
        expect(ratioOut.length, 1);
        expect(dwdhOut.length, 2);
      } finally {
        src.dispose();
        result.dispose();
      }
    });
  });

  group('letterbox256', () {
    test('produces 256x256 output', () {
      final src = cv.Mat.zeros(100, 150, cv.MatType.CV_8UC3);
      final ratioOut = <double>[];
      final dwdhOut = <int>[];

      final result = ImageUtils.letterbox256(src, ratioOut, dwdhOut);

      try {
        expect(result.cols, 256);
        expect(result.rows, 256);
      } finally {
        src.dispose();
        result.dispose();
      }
    });
  });

  group('letterbox224', () {
    test('produces 224x224 output', () {
      final src = cv.Mat.zeros(100, 150, cv.MatType.CV_8UC3);
      final ratioOut = <double>[];
      final dwdhOut = <int>[];

      final result = ImageUtils.letterbox224(src, ratioOut, dwdhOut);

      try {
        expect(result.cols, 224);
        expect(result.rows, 224);
      } finally {
        src.dispose();
        result.dispose();
      }
    });
  });

  group('scaleFromLetterbox', () {
    test('reverses letterbox transformation', () {
      // Simulate a letterbox with ratio=2.0, dw=10, dh=20
      final xyxy = [30.0, 60.0, 130.0, 160.0];
      final result = ImageUtils.scaleFromLetterbox(xyxy, 2.0, 10, 20);

      // x1 = (30 - 10) / 2 = 10
      expect(result[0], closeTo(10.0, 0.001));
      // y1 = (60 - 20) / 2 = 20
      expect(result[1], closeTo(20.0, 0.001));
      // x2 = (130 - 10) / 2 = 60
      expect(result[2], closeTo(60.0, 0.001));
      // y2 = (160 - 20) / 2 = 70
      expect(result[3], closeTo(70.0, 0.001));
    });

    test('round-trip with letterbox', () {
      final src = cv.Mat.zeros(200, 300, cv.MatType.CV_8UC3);
      final ratioOut = <double>[];
      final dwdhOut = <int>[];
      final result = ImageUtils.letterbox(src, 256, 256, ratioOut, dwdhOut);

      // Original bbox in source image
      final originalBox = [50.0, 30.0, 150.0, 100.0];

      // Transform to letterbox space
      final ratio = ratioOut[0];
      final dw = dwdhOut[0];
      final dh = dwdhOut[1];
      final letterboxBox = [
        originalBox[0] * ratio + dw,
        originalBox[1] * ratio + dh,
        originalBox[2] * ratio + dw,
        originalBox[3] * ratio + dh,
      ];

      // Reverse transform
      final recovered =
          ImageUtils.scaleFromLetterbox(letterboxBox, ratio, dw, dh);

      expect(recovered[0], closeTo(originalBox[0], 0.5));
      expect(recovered[1], closeTo(originalBox[1], 0.5));
      expect(recovered[2], closeTo(originalBox[2], 0.5));
      expect(recovered[3], closeTo(originalBox[3], 0.5));

      src.dispose();
      result.dispose();
    });

    test('handles zero padding', () {
      final xyxy = [10.0, 20.0, 100.0, 200.0];
      final result = ImageUtils.scaleFromLetterbox(xyxy, 1.0, 0, 0);

      expect(result[0], 10.0);
      expect(result[1], 20.0);
      expect(result[2], 100.0);
      expect(result[3], 200.0);
    });

    test('handles ratio of 1.0', () {
      final xyxy = [15.0, 25.0, 115.0, 225.0];
      final result = ImageUtils.scaleFromLetterbox(xyxy, 1.0, 5, 5);

      expect(result[0], closeTo(10.0, 0.001));
      expect(result[1], closeTo(20.0, 0.001));
      expect(result[2], closeTo(110.0, 0.001));
      expect(result[3], closeTo(220.0, 0.001));
    });
  });

  group('matToFloat32Tensor', () {
    test('converts BGR to RGB and normalizes', () {
      // Create a 2x2 image with known BGR values
      final mat = cv.Mat.zeros(2, 2, cv.MatType.CV_8UC3);
      mat.setTo(cv.Scalar(255, 0, 0, 0)); // Pure blue in BGR

      try {
        final tensor = ImageUtils.matToFloat32Tensor(mat);

        expect(tensor.length, 2 * 2 * 3);
        // BGR(255,0,0) -> RGB: R=0, G=0, B=1.0
        expect(tensor[0], closeTo(0.0, 0.01)); // R
        expect(tensor[1], closeTo(0.0, 0.01)); // G
        expect(tensor[2], closeTo(1.0, 0.01)); // B
      } finally {
        mat.dispose();
      }
    });

    test('uses pre-allocated buffer when provided', () {
      final mat = cv.Mat.zeros(2, 2, cv.MatType.CV_8UC3);
      mat.setTo(cv.Scalar(0, 255, 0, 0)); // Pure green in BGR

      final buffer = Float32List(2 * 2 * 3);

      try {
        final result = ImageUtils.matToFloat32Tensor(mat, buffer: buffer);

        // Should return the same buffer
        expect(identical(result, buffer), true);
        // Green: BGR(0,255,0) -> RGB: R=0, G=1.0, B=0
        expect(buffer[0], closeTo(0.0, 0.01)); // R
        expect(buffer[1], closeTo(1.0, 0.01)); // G
        expect(buffer[2], closeTo(0.0, 0.01)); // B
      } finally {
        mat.dispose();
      }
    });

    test('normalizes 0-255 to 0.0-1.0', () {
      final mat = cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3);
      mat.setTo(cv.Scalar(128, 128, 128, 0)); // Gray

      try {
        final tensor = ImageUtils.matToFloat32Tensor(mat);

        // 128/255 ≈ 0.502
        for (int i = 0; i < 3; i++) {
          expect(tensor[i], closeTo(128.0 / 255.0, 0.01));
        }
      } finally {
        mat.dispose();
      }
    });

    test('handles white image', () {
      final mat = cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3);
      mat.setTo(cv.Scalar(255, 255, 255, 0));

      try {
        final tensor = ImageUtils.matToFloat32Tensor(mat);
        for (int i = 0; i < 3; i++) {
          expect(tensor[i], closeTo(1.0, 0.01));
        }
      } finally {
        mat.dispose();
      }
    });

    test('handles black image', () {
      final mat = cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3);

      try {
        final tensor = ImageUtils.matToFloat32Tensor(mat);
        for (int i = 0; i < 3; i++) {
          expect(tensor[i], closeTo(0.0, 0.01));
        }
      } finally {
        mat.dispose();
      }
    });
  });

  group('matToNHWC4D', () {
    test('produces correct shape [1, h, w, 3]', () {
      final mat = cv.Mat.zeros(3, 4, cv.MatType.CV_8UC3);

      try {
        final tensor = ImageUtils.matToNHWC4D(mat, 4, 3);

        expect(tensor.length, 1); // batch
        expect(tensor[0].length, 3); // height
        expect(tensor[0][0].length, 4); // width
        expect(tensor[0][0][0].length, 3); // channels
      } finally {
        mat.dispose();
      }
    });

    test('converts BGR to RGB and normalizes', () {
      final mat = cv.Mat.zeros(1, 1, cv.MatType.CV_8UC3);
      mat.setTo(cv.Scalar(255, 0, 0, 0)); // Blue in BGR

      try {
        final tensor = ImageUtils.matToNHWC4D(mat, 1, 1);

        // BGR(255,0,0) -> RGB: R=0, G=0, B=1.0
        expect(tensor[0][0][0][0], closeTo(0.0, 0.01)); // R
        expect(tensor[0][0][0][1], closeTo(0.0, 0.01)); // G
        expect(tensor[0][0][0][2], closeTo(1.0, 0.01)); // B
      } finally {
        mat.dispose();
      }
    });

    test('reuses provided buffer', () {
      final mat = cv.Mat.zeros(2, 2, cv.MatType.CV_8UC3);
      mat.setTo(cv.Scalar(0, 255, 0, 0)); // Green

      // Pre-allocate a matching tensor
      final reuse = List.generate(
        1,
        (_) => List.generate(
          2,
          (_) => List.generate(
            2,
            (_) => List<double>.filled(3, -1.0),
          ),
        ),
      );

      try {
        final result = ImageUtils.matToNHWC4D(mat, 2, 2, reuse: reuse);

        expect(identical(result, reuse), true);
        // Green: BGR(0,255,0) -> RGB: R=0, G=1.0, B=0
        expect(reuse[0][0][0][0], closeTo(0.0, 0.01));
        expect(reuse[0][0][0][1], closeTo(1.0, 0.01));
        expect(reuse[0][0][0][2], closeTo(0.0, 0.01));
      } finally {
        mat.dispose();
      }
    });

    test('handles multi-pixel image correctly', () {
      // Create a 2x1 image with different colors
      final mat = cv.Mat.zeros(1, 2, cv.MatType.CV_8UC3);
      // Set pixels manually is tricky, just verify shape
      try {
        final tensor = ImageUtils.matToNHWC4D(mat, 2, 1);

        expect(tensor.length, 1);
        expect(tensor[0].length, 1);
        expect(tensor[0][0].length, 2);
      } finally {
        mat.dispose();
      }
    });
  });

  group('reshapeToTensor4D', () {
    test('reshapes flat array to 4D', () {
      // 2x3x4x2 = 48 elements
      final flat = List<double>.generate(48, (i) => i.toDouble());
      final result = ImageUtils.reshapeToTensor4D(flat, 2, 3, 4, 2);

      expect(result.length, 2);
      expect(result[0].length, 3);
      expect(result[0][0].length, 4);
      expect(result[0][0][0].length, 2);

      // Check first element
      expect(result[0][0][0][0], 0.0);
      expect(result[0][0][0][1], 1.0);

      // Check a middle element: index = 1*3*4*2 + 1*4*2 + 2*2 + 1 = 24+8+4+1 = 37
      expect(result[1][1][2][1], 37.0);

      // Check last element: index = 47
      expect(result[1][2][3][1], 47.0);
    });

    test('reshapes 1x1x1x1', () {
      final flat = [42.0];
      final result = ImageUtils.reshapeToTensor4D(flat, 1, 1, 1, 1);

      expect(result[0][0][0][0], 42.0);
    });

    test('reshapes batch of images', () {
      // Simulate batch=2, h=2, w=2, c=3 = 24 elements
      final flat = List<double>.generate(24, (i) => i.toDouble());
      final result = ImageUtils.reshapeToTensor4D(flat, 2, 2, 2, 3);

      expect(result.length, 2);
      expect(result[0][0][0], [0.0, 1.0, 2.0]);
      expect(result[0][0][1], [3.0, 4.0, 5.0]);
      expect(result[0][1][0], [6.0, 7.0, 8.0]);
      expect(result[0][1][1], [9.0, 10.0, 11.0]);
      expect(result[1][0][0], [12.0, 13.0, 14.0]);
    });

    test('values are in row-major order', () {
      final flat = List<double>.generate(12, (i) => i.toDouble());
      final result = ImageUtils.reshapeToTensor4D(flat, 1, 2, 2, 3);

      int idx = 0;
      for (int i = 0; i < 1; i++) {
        for (int j = 0; j < 2; j++) {
          for (int k = 0; k < 2; k++) {
            for (int l = 0; l < 3; l++) {
              expect(result[i][j][k][l], idx.toDouble());
              idx++;
            }
          }
        }
      }
    });
  });
}
