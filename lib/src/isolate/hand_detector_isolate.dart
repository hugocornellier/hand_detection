import 'dart:typed_data';

import 'package:opencv_dart/opencv_dart.dart' as cv;

import '../hand_detector.dart';
import '../types.dart';

/// Deprecated: Use [HandDetector] instead.
///
/// [HandDetector] now runs inference in a background isolate by default.
/// This class is kept for backward compatibility and delegates all operations
/// to a [HandDetector] instance internally.
@Deprecated(
  'Use HandDetector instead. HandDetector now runs inference in a background '
  'isolate by default. Will be removed in a future release.',
)
class HandDetectorIsolate {
  HandDetectorIsolate._({required HandDetector detector})
      : _detector = detector;

  final HandDetector _detector;

  /// Spawns a new isolate with an initialized [HandDetector].
  ///
  /// Deprecated: Use [HandDetector] with [HandDetector.initialize] instead.
  static Future<HandDetectorIsolate> spawn({
    HandMode mode = HandMode.boxesAndLandmarks,
    HandLandmarkModel landmarkModel = HandLandmarkModel.full,
    double detectorConf = 0.45,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    bool enableGestures = false,
    double gestureMinConfidence = 0.5,
  }) async {
    final detector = HandDetector(
      mode: mode,
      landmarkModel: landmarkModel,
      detectorConf: detectorConf,
      maxDetections: maxDetections,
      minLandmarkScore: minLandmarkScore,
      interpreterPoolSize: interpreterPoolSize,
      performanceConfig: performanceConfig,
      enableGestures: enableGestures,
      gestureMinConfidence: gestureMinConfidence,
    );
    await detector.initialize();
    return HandDetectorIsolate._(detector: detector);
  }

  /// Returns true when the underlying [HandDetector] is initialized and ready.
  bool get isReady => _detector.isReady;

  /// Detects hands in the given encoded image in the background isolate.
  ///
  /// Parameters:
  /// - [imageBytes]: Encoded image data (JPEG, PNG, etc.) as a [List<int>] or [Uint8List]
  ///
  /// Returns a list of [Hand] objects, one per detected hand.
  Future<List<Hand>> detectHands(List<int> imageBytes) =>
      _detector.detect(imageBytes);

  /// Detects hands in a pre-decoded [cv.Mat] image in the background isolate.
  ///
  /// The original Mat is NOT disposed by this method.
  Future<List<Hand>> detectHandsFromMat(cv.Mat image) =>
      _detector.detectFromMat(image);

  /// Detects hands from raw pixel bytes in the background isolate.
  ///
  /// Parameters:
  /// - [bytes]: Raw pixel data (typically BGR format, 3 bytes per pixel)
  /// - [width]: Image width in pixels
  /// - [height]: Image height in pixels
  /// - [matType]: OpenCV MatType value (default: CV_8UC3 = 16 for BGR)
  ///
  /// Returns a list of [Hand] objects, one per detected hand.
  Future<List<Hand>> detectHandsFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
  }) =>
      _detector.detectFromMatBytes(bytes,
          width: width, height: height, matType: matType);

  /// Disposes the background isolate and releases all resources.
  Future<void> dispose() => _detector.dispose();
}
