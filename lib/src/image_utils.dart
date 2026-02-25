import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'palm_detector.dart';

/// Utility functions for image preprocessing and transformations using OpenCV.
///
/// Provides letterbox preprocessing, coordinate transformations, tensor
/// conversion utilities, and rotation-aware cropping for hand detection.
/// Uses native OpenCV operations for 10-50x better performance than pure Dart.
class ImageUtils {
  /// Keeps aspect ratio while resizing and centers with padding.
  ///
  /// This matches the Python keep_aspect_resize_and_pad function.
  /// Uses OpenCV's native resize for significantly better performance.
  static (cv.Mat padded, cv.Mat resized) keepAspectResizeAndPad(
    cv.Mat image,
    int resizeWidth,
    int resizeHeight,
  ) {
    final imageHeight = image.rows;
    final imageWidth = image.cols;

    final ash = resizeHeight / imageHeight;
    final asw = resizeWidth / imageWidth;

    int newWidth, newHeight;
    if (asw < ash) {
      newWidth = (imageWidth * asw).toInt();
      newHeight = (imageHeight * asw).toInt();
    } else {
      newWidth = (imageWidth * ash).toInt();
      newHeight = (imageHeight * ash).toInt();
    }

    // Resize with bilinear interpolation (INTER_LINEAR)
    final resizedImage =
        cv.resize(image, (newWidth, newHeight), interpolation: cv.INTER_LINEAR);

    // Calculate padding for each side
    final padTop = (resizeHeight - newHeight) ~/ 2;
    final padBottom = resizeHeight - newHeight - padTop;
    final padLeft = (resizeWidth - newWidth) ~/ 2;
    final padRight = resizeWidth - newWidth - padLeft;

    // Use copyMakeBorder for reliable padding (avoids ROI copy issues)
    final paddedImage = cv.copyMakeBorder(
      resizedImage,
      padTop,
      padBottom,
      padLeft,
      padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar.black,
    );

    return (paddedImage, resizedImage);
  }

  /// Crops a rotated rectangle from an image using OpenCV's warpAffine.
  ///
  /// This is used to extract hand regions with proper rotation alignment
  /// for the landmark model. Uses OpenCV's SIMD-optimized warpAffine which
  /// is 10-50x faster than pure Dart bilinear interpolation.
  ///
  /// Parameters:
  /// - [image]: Source image
  /// - [palm]: Palm detection containing rotation rectangle parameters
  /// - [padding]: Whether to handle edge cases (always handled via border mode)
  ///
  /// Returns the cropped and rotated hand image, or null if the crop is invalid.
  static cv.Mat? rotateAndCropRectangle(
    cv.Mat image,
    PalmDetection palm, {
    bool padding = true,
  }) {
    final imageWidth = image.cols;
    final imageHeight = image.rows;

    // Convert normalized coordinates to pixel coordinates
    final cx = palm.sqnRrCenterX * imageWidth;
    final cy = palm.sqnRrCenterY * imageHeight;

    // Calculate size based on image dimensions
    final size = (palm.sqnRrSize * math.max(imageWidth, imageHeight)).round();
    if (size <= 0) return null;

    // Rotation angle (positive direction to match Python, converted to degrees)
    final angleDegrees = palm.rotation * 180.0 / math.pi;

    // Get rotation matrix centered at the palm center
    final rotMat =
        cv.getRotationMatrix2D(cv.Point2f(cx, cy), angleDegrees, 1.0);

    // Adjust translation to crop around the output center
    final outCx = size / 2.0;
    final outCy = size / 2.0;

    // Modify the translation in the rotation matrix:
    // M[0,2] += outCx - cx
    // M[1,2] += outCy - cy
    final tx = rotMat.at<double>(0, 2) + outCx - cx;
    final ty = rotMat.at<double>(1, 2) + outCy - cy;
    rotMat.set<double>(0, 2, tx);
    rotMat.set<double>(1, 2, ty);

    // Apply affine transform with SIMD-optimized warpAffine
    final output = cv.warpAffine(
      image,
      rotMat,
      (size, size),
      borderMode: cv.BORDER_CONSTANT,
      borderValue: cv.Scalar.black,
    );

    rotMat.dispose();
    return output;
  }

  /// Creates a rotated rectangle crop info from palm detection.
  ///
  /// Returns the crop parameters needed for landmark extraction:
  /// [cx, cy, width, height, angleDegrees]
  static List<double> palmToRect(
      PalmDetection palm, int imageWidth, int imageHeight) {
    final cx = palm.sqnRrCenterX * imageWidth;
    final cy = palm.sqnRrCenterY * imageHeight;
    final size = palm.sqnRrSize * math.max(imageWidth, imageHeight);
    final angleDegrees = palm.rotation * 180.0 / math.pi;

    return [cx, cy, size, size, angleDegrees];
  }

  /// Applies letterbox preprocessing to fit an image into target dimensions.
  ///
  /// Scales the source image to fit within [tw]x[th] while maintaining aspect ratio,
  /// then pads with gray (114, 114, 114) to fill the target dimensions.
  ///
  /// This is critical for YOLO-style object detection models that expect fixed input sizes.
  ///
  /// Parameters:
  /// - [src]: Source image to preprocess
  /// - [tw]: Target width in pixels
  /// - [th]: Target height in pixels
  /// - [ratioOut]: Output parameter that receives the scale ratio used
  /// - [dwdhOut]: Output parameter that receives padding [dw, dh] values
  ///
  /// Returns the letterboxed image with dimensions [tw]x[th].
  static cv.Mat letterbox(
    cv.Mat src,
    int tw,
    int th,
    List<double> ratioOut,
    List<int> dwdhOut,
  ) {
    final int w = src.cols;
    final int h = src.rows;
    final double r = math.min(th / h, tw / w);
    final int nw = (w * r).round();
    final int nh = (h * r).round();
    final int dw = (tw - nw) ~/ 2;
    final int dh = (th - nh) ~/ 2;

    final resized = cv.resize(src, (nw, nh), interpolation: cv.INTER_LINEAR);

    // Calculate padding for right/bottom to ensure exact target dimensions
    final dwRight = tw - nw - dw;
    final dhBottom = th - nh - dh;

    // Use copyMakeBorder for reliable padding with gray fill (114, 114, 114)
    final canvas = cv.copyMakeBorder(
      resized,
      dh,
      dhBottom,
      dw,
      dwRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar(114, 114, 114, 0),
    );
    resized.dispose();

    ratioOut
      ..clear()
      ..add(r);
    dwdhOut
      ..clear()
      ..addAll([dw, dh]);
    return canvas;
  }

  /// Applies letterbox preprocessing to 256x256 dimensions.
  ///
  /// Convenience method that calls [letterbox] with fixed 256x256 target size.
  /// Used for hand landmark model preprocessing.
  static cv.Mat letterbox256(
    cv.Mat src,
    List<double> ratioOut,
    List<int> dwdhOut,
  ) {
    return letterbox(src, 256, 256, ratioOut, dwdhOut);
  }

  /// Applies letterbox preprocessing to 224x224 dimensions.
  ///
  /// Convenience method that calls [letterbox] with fixed 224x224 target size.
  /// Used for hand landmark model preprocessing (MediaPipe format).
  static cv.Mat letterbox224(
    cv.Mat src,
    List<double> ratioOut,
    List<int> dwdhOut,
  ) {
    return letterbox(src, 224, 224, ratioOut, dwdhOut);
  }

  /// Transforms bounding box coordinates from letterbox space back to original image space.
  ///
  /// Reverses the letterbox transformation by removing padding and unscaling coordinates.
  ///
  /// Parameters:
  /// - [xyxy]: Bounding box in letterbox space as [x1, y1, x2, y2]
  /// - [ratio]: Scale ratio from letterbox preprocessing
  /// - [dw]: Horizontal padding from letterbox preprocessing
  /// - [dh]: Vertical padding from letterbox preprocessing
  ///
  /// Returns the bounding box in original image space as [x1, y1, x2, y2].
  static List<double> scaleFromLetterbox(
    List<double> xyxy,
    double ratio,
    int dw,
    int dh,
  ) {
    final double x1 = (xyxy[0] - dw) / ratio;
    final double y1 = (xyxy[1] - dh) / ratio;
    final double x2 = (xyxy[2] - dw) / ratio;
    final double y2 = (xyxy[3] - dh) / ratio;
    return [x1, y1, x2, y2];
  }

  /// Converts a cv.Mat to a flat Float32List tensor for TensorFlow Lite.
  ///
  /// Converts pixel values from 0-255 range to normalized 0.0-1.0 range.
  /// Also converts from BGR (OpenCV format) to RGB (TFLite expected format).
  ///
  /// Parameters:
  /// - [mat]: Source image in BGR format
  /// - [buffer]: Optional pre-allocated buffer to reuse
  ///
  /// Returns a flat Float32List with normalized RGB pixel values.
  static Float32List matToFloat32Tensor(cv.Mat mat, {Float32List? buffer}) {
    final data = mat.data;
    final totalPixels = mat.rows * mat.cols;
    final size = totalPixels * 3;
    final tensor = buffer ?? Float32List(size);
    const scale = 1.0 / 255.0;

    // OpenCV uses BGR, convert to RGB and normalize
    for (int i = 0, j = 0; i < totalPixels * 3 && j < size; i += 3, j += 3) {
      tensor[j] = data[i + 2] * scale; // B -> R
      tensor[j + 1] = data[i + 1] * scale; // G -> G
      tensor[j + 2] = data[i] * scale; // R -> B
    }
    return tensor;
  }

  /// Converts an image to a 4D tensor in NHWC format for TensorFlow Lite.
  ///
  /// Converts pixel values from 0-255 range to normalized 0.0-1.0 range.
  /// Also converts from BGR (OpenCV format) to RGB (TFLite expected format).
  /// The output format is [batch, height, width, channels] where batch=1 and channels=3 (RGB).
  ///
  /// Parameters:
  /// - [mat]: Source image in BGR format
  /// - [width]: Target width (must match mat.cols)
  /// - [height]: Target height (must match mat.rows)
  /// - [reuse]: Optional tensor buffer to reuse (must match dimensions)
  ///
  /// Returns a 4D tensor [1, height, width, 3] with normalized pixel values.
  static List<List<List<List<double>>>> matToNHWC4D(
    cv.Mat mat,
    int width,
    int height, {
    List<List<List<List<double>>>>? reuse,
  }) {
    final List<List<List<List<double>>>> out = reuse ??
        List.generate(
          1,
          (_) => List.generate(
            height,
            (_) => List.generate(
              width,
              (_) => List<double>.filled(3, 0.0),
              growable: false,
            ),
            growable: false,
          ),
          growable: false,
        );

    final bytes = mat.data;
    const double scale = 1.0 / 255.0;
    int byteIndex = 0;

    // OpenCV uses BGR, convert to RGB
    for (int y = 0; y < height; y++) {
      final List<List<double>> row = out[0][y];
      for (int x = 0; x < width; x++) {
        final List<double> pixel = row[x];
        pixel[0] = bytes[byteIndex + 2] * scale; // B -> R
        pixel[1] = bytes[byteIndex + 1] * scale; // G -> G
        pixel[2] = bytes[byteIndex] * scale; // R -> B
        byteIndex += 3;
      }
    }
    return out;
  }

  /// Reshapes a flat array into a 4D tensor.
  ///
  /// Converts a 1D flattened array into a 4D nested list structure with the
  /// specified dimensions. Used for converting TensorFlow Lite output buffers
  /// into the expected tensor shape.
  ///
  /// Parameters:
  /// - [flat]: Flat array of values (length must equal dim1 * dim2 * dim3 * dim4)
  /// - [dim1]: First dimension size (batch)
  /// - [dim2]: Second dimension size (height/rows)
  /// - [dim3]: Third dimension size (width/columns)
  /// - [dim4]: Fourth dimension size (channels/features)
  ///
  /// Returns a 4D tensor [dim1, dim2, dim3, dim4] populated from [flat] in row-major order.
  static List<List<List<List<double>>>> reshapeToTensor4D(
    List<double> flat,
    int dim1,
    int dim2,
    int dim3,
    int dim4,
  ) {
    final List<List<List<List<double>>>> result = List.generate(
      dim1,
      (_) => List.generate(
        dim2,
        (_) => List.generate(
          dim3,
          (_) => List<double>.filled(dim4, 0.0),
        ),
      ),
    );

    int index = 0;
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        for (int k = 0; k < dim3; k++) {
          for (int l = 0; l < dim4; l++) {
            result[i][j][k][l] = flat[index++];
          }
        }
      }
    }

    return result;
  }
}
