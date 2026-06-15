import 'dart:math' as math;
import 'dart:typed_data';
import 'package:meta/meta.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:flutter_litert/flutter_litert.dart';
import '../models/palm_detector.dart';

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
    final params = computeLetterboxParams(
      srcWidth: image.cols,
      srcHeight: image.rows,
      targetWidth: resizeWidth,
      targetHeight: resizeHeight,
      roundDimensions: false,
    );

    final resizedImage = cv.resize(
      image,
      (params.newWidth, params.newHeight),
      interpolation: cv.INTER_LINEAR,
    );

    final paddedImage = cv.copyMakeBorder(
      resizedImage,
      params.padTop,
      params.padBottom,
      params.padLeft,
      params.padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar.black,
    );

    return (paddedImage, resizedImage);
  }

  /// Returns the center and size of a palm detection in pixel coordinates.
  static ({double cx, double cy, double size}) palmCoordinates(
    PalmDetection palm,
    int imageWidth,
    int imageHeight,
  ) {
    return (
      cx: palm.sqnRrCenterX * imageWidth,
      cy: palm.sqnRrCenterY * imageHeight,
      size: palm.sqnRrSize * math.max(imageWidth, imageHeight),
    );
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
  ///
  /// Returns the cropped and rotated hand image, or null if the crop is invalid.
  static cv.Mat? rotateAndCropRectangle(
    cv.Mat image,
    PalmDetection palm, {
    int? outSize,
  }) {
    final imageWidth = image.cols;
    final imageHeight = image.rows;

    final (:cx, :cy, size: sizeD) =
        palmCoordinates(palm, imageWidth, imageHeight);
    final size = sizeD.round();
    if (size <= 0) return null;
    final int outDim = outSize ?? size;
    if (outDim <= 0) return null;

    final angleDegrees = palm.rotation * 180.0 / math.pi;

    // Fold the crop->output scale into the rotation matrix so a single
    // warpAffine performs rotate + crop + resize straight to the model input
    // size, avoiding a large native-resolution intermediate Mat and a separate
    // cv.resize. scale == 1 (outDim == size) preserves the original behaviour.
    final double scale = outDim / sizeD;
    final rotMat =
        cv.getRotationMatrix2D(cv.Point2f(cx, cy), angleDegrees, scale);

    final outCx = outDim / 2.0;
    final outCy = outDim / 2.0;

    final tx = rotMat.at<double>(0, 2) + outCx - cx;
    final ty = rotMat.at<double>(1, 2) + outCy - cy;
    rotMat.set<double>(0, 2, tx);
    rotMat.set<double>(1, 2, ty);

    final output = cv.warpAffine(
      image,
      rotMat,
      (outDim, outDim),
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
  @visibleForTesting
  static List<double> palmToRect(
      PalmDetection palm, int imageWidth, int imageHeight) {
    final (:cx, :cy, :size) = palmCoordinates(palm, imageWidth, imageHeight);
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
  @visibleForTesting
  static cv.Mat letterbox(
    cv.Mat src,
    int tw,
    int th,
    List<double> ratioOut,
    List<int> dwdhOut,
  ) {
    final params = computeLetterboxParams(
      srcWidth: src.cols,
      srcHeight: src.rows,
      targetWidth: tw,
      targetHeight: th,
    );

    final resized = cv.resize(
      src,
      (params.newWidth, params.newHeight),
      interpolation: cv.INTER_LINEAR,
    );

    final canvas = cv.copyMakeBorder(
      resized,
      params.padTop,
      params.padBottom,
      params.padLeft,
      params.padRight,
      cv.BORDER_CONSTANT,
      value: cv.Scalar(114, 114, 114, 0),
    );
    resized.dispose();

    ratioOut
      ..clear()
      ..add(params.scale);
    dwdhOut
      ..clear()
      ..addAll([params.padLeft, params.padTop]);
    return canvas;
  }

  /// Converts a cv.Mat to a flat Float32List tensor for TensorFlow Lite.
  ///
  /// Converts pixel values from 0-255 range to normalized 0.0-1.0 range and
  /// from BGR (OpenCV format) to RGB (TFLite expected format).
  ///
  /// Uses OpenCV's SIMD-accelerated `cvtColor` + `convertTo` instead of a
  /// per-pixel Dart loop. Both ops are 5-10x faster than the pure-Dart
  /// `bgrBytesToRgbFloat32` for typical 192×192 / 224×224 inputs.
  ///
  /// Parameters:
  /// - [mat]: Source image in BGR format
  /// - [buffer]: Optional pre-allocated buffer to reuse
  ///
  /// Returns a flat Float32List with normalized RGB pixel values.
  static Float32List matToFloat32Tensor(cv.Mat mat, {Float32List? buffer}) {
    // BGR -> RGB (in-place by reusing dst)
    final cv.Mat rgb = cv.cvtColor(mat, cv.COLOR_BGR2RGB);
    // uint8 -> float32, with /255 normalization in one SIMD pass.
    final cv.Mat f32 = rgb.convertTo(cv.MatType.CV_32FC3, alpha: 1.0 / 255.0);
    rgb.dispose();

    final int totalFloats = f32.rows * f32.cols * 3;
    final Float32List dst = buffer ?? Float32List(totalFloats);
    // Mat.data is a Uint8List view of the underlying native bytes; reinterpret
    // as Float32List and copy into the (possibly tensor-backed) dst buffer.
    final Float32List src = f32.data.buffer.asFloat32List(0, totalFloats);
    dst.setRange(0, totalFloats, src);
    f32.dispose();
    return dst;
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
  @visibleForTesting
  static List<List<List<List<double>>>> matToNHWC4D(
    cv.Mat mat,
    int width,
    int height, {
    List<List<List<List<double>>>>? reuse,
  }) {
    final out = reuse ?? createNHWCTensor4D(height, width);
    fillNHWC4DFromBgrBytes(
        bytes: mat.data, tensor: out, width: width, height: height);
    return out;
  }

  /// Rebuilds a tightly packed [cv.Mat] from raw [bytes] with a single copy.
  ///
  /// `cv.Mat.fromList` walks a lazy `cast<int>` view byte-by-byte, which is
  /// slow for camera frames. Allocating the Mat first and copying into its
  /// native buffer keeps reconstruction near memcpy speed. Only use this for
  /// tightly packed, continuous pixel data.
  static cv.Mat matFromPackedBytes(
    int rows,
    int cols,
    cv.MatType type,
    Uint8List bytes,
  ) {
    final cv.Mat mat = cv.Mat.create(rows: rows, cols: cols, type: type);
    final Uint8List dst = mat.data;
    if (dst.length != bytes.length) {
      final int expected = dst.length;
      mat.dispose();
      throw ArgumentError(
        'bytes.length ${bytes.length} does not match a '
        '$rows x $cols Mat of type $type ($expected bytes)',
      );
    }
    dst.setAll(0, bytes);
    return mat;
  }

  /// Rebuilds a [cv.Mat] from a backend-neutral packed image [layout].
  static cv.Mat matFromPackedLayout(
    PackedImageLayout layout,
    Uint8List bytes,
    cv.MatType type,
  ) {
    final cv.Mat mat = cv.Mat.create(
      rows: layout.rows,
      cols: layout.cols,
      type: type,
    );
    try {
      layout.copyTo(mat.data, bytes);
      return mat;
    } catch (_) {
      mat.dispose();
      rethrow;
    }
  }

  /// Decodes a [CameraFrame] into a 3-channel BGR [cv.Mat].
  ///
  /// The layout and safe operation order come from `flutter_litert`; this
  /// method only maps that backend-neutral plan onto OpenCV primitives. Shared
  /// with the face/pose detectors so all three convert camera frames the same
  /// way.
  static cv.Mat cameraFrameToBgrMat(CameraFrame frame, {int? maxDim}) {
    final CameraFrameDecodePlan plan = frame.decodePlan();
    final cv.Mat source = matFromPackedLayout(
      plan.sourceLayout,
      frame.bytes,
      plan.sourceLayout.channels == 4 ? cv.MatType.CV_8UC4 : cv.MatType.CV_8UC1,
    );

    cv.Mat maybeResize(cv.Mat m) {
      if (maxDim == null || (m.cols <= maxDim && m.rows <= maxDim)) return m;
      final double scale = maxDim / (m.cols > m.rows ? m.cols : m.rows);
      final cv.Mat resized = cv.resize(
          m,
          (
            (m.cols * scale).toInt(),
            (m.rows * scale).toInt(),
          ),
          interpolation: cv.INTER_LINEAR);
      m.dispose();
      return resized;
    }

    int? rotateFlag() {
      return switch (plan.rotation) {
        CameraFrameRotation.cw90 => cv.ROTATE_90_CLOCKWISE,
        CameraFrameRotation.cw180 => cv.ROTATE_180,
        CameraFrameRotation.cw270 => cv.ROTATE_90_COUNTERCLOCKWISE,
        null => null,
      };
    }

    cv.Mat maybeRotate(cv.Mat m) {
      final int? flag = rotateFlag();
      if (flag == null) return m;
      final cv.Mat rotated = cv.rotate(m, flag);
      m.dispose();
      return rotated;
    }

    final int cvtCode = switch (plan.conversion) {
      CameraFrameConversion.bgra2bgr => cv.COLOR_BGRA2BGR,
      CameraFrameConversion.rgba2bgr => cv.COLOR_RGBA2BGR,
      CameraFrameConversion.yuv2bgrNv12 => cv.COLOR_YUV2BGR_NV12,
      CameraFrameConversion.yuv2bgrNv21 => cv.COLOR_YUV2BGR_NV21,
      CameraFrameConversion.yuv2bgrI420 => cv.COLOR_YUV2BGR_I420,
    };

    switch (plan.order) {
      case CameraFrameDecodeOrder.resizeRotateThenColorConvert:
        cv.Mat current = plan.hasStridePadding
            ? source.region(
                cv.Rect(0, 0, plan.visibleWidth, plan.visibleHeight),
              )
            : source;

        if (maxDim != null &&
            (current.cols > maxDim || current.rows > maxDim)) {
          final double scale = maxDim /
              (current.cols > current.rows ? current.cols : current.rows);
          final cv.Mat resized = cv.resize(
              current,
              (
                (current.cols * scale).toInt(),
                (current.rows * scale).toInt(),
              ),
              interpolation: cv.INTER_LINEAR);
          if (!identical(current, source)) current.dispose();
          current = resized;
        }

        final int? flag = rotateFlag();
        if (flag != null) {
          final cv.Mat rotated = cv.rotate(current, flag);
          if (!identical(current, source)) current.dispose();
          current = rotated;
        }

        final cv.Mat bgr = cv.cvtColor(current, cvtCode);
        if (!identical(current, source)) current.dispose();
        source.dispose();
        return bgr;

      case CameraFrameDecodeOrder.colorConvertThenResizeRotate:
        cv.Mat current = cv.cvtColor(source, cvtCode);
        source.dispose();
        current = maybeResize(current);
        current = maybeRotate(current);
        return current;
    }
  }
}
