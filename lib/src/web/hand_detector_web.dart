// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:async';
import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter_litert/flutter_litert.dart'
    show BoundingBox, PerformanceConfig;
import 'package:flutter_litert/src/web/web_detector_utils.dart'
    show decodeBitmap, WebGpuFallback;
import 'package:web/web.dart' as web;

import '../shared/hand_geometry.dart' show PalmDetection;
import '../shared/hand_types.dart';
import 'models/gesture_recognizer_web.dart';
import 'models/hand_landmark_model_web.dart';
import 'models/palm_detector_web.dart';

/// Web implementation of [HandDetector].
///
/// Mirrors the public surface of the native detector for the detect-from-bytes
/// and detect-from-video use cases, backed by LiteRT.js (auto WebGPU/WASM) and
/// Canvas preprocessing. All inference runs on the main thread (no isolates).
/// Native-only entry points (filepath, Mat, camera frames) throw
/// [UnsupportedError] on web.
class HandDetector with WebGpuFallback {
  static const String modelVersion = 'hand_detection:web:1.0.0';

  HandDetector();

  /// Creates and initializes a web hand detector in one step.
  static Future<HandDetector> create({
    HandMode mode = HandMode.boxesAndLandmarks,
    HandLandmarkModel landmarkModel = HandLandmarkModel.full,
    double detectorConf = 0.45,
    double palmNmsIou = 0.45,
    double palmRoiScale = 2.6,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    bool enableTracking = false,
    TrackingConfig trackingConfig = const TrackingConfig(),
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    bool enableGestures = false,
    double gestureMinConfidence = 0.5,
    bool useCompiledModel = false,
    String liteRtAccelerator = 'auto',
  }) async {
    final detector = HandDetector();
    await detector.initialize(
      mode: mode,
      landmarkModel: landmarkModel,
      detectorConf: detectorConf,
      palmNmsIou: palmNmsIou,
      palmRoiScale: palmRoiScale,
      maxDetections: maxDetections,
      minLandmarkScore: minLandmarkScore,
      enableTracking: enableTracking,
      trackingConfig: trackingConfig,
      enableGestures: enableGestures,
      gestureMinConfidence: gestureMinConfidence,
      liteRtAccelerator: liteRtAccelerator,
    );
    return detector;
  }

  // Reassigned in [initialize] with the caller's palm thresholds; a default
  // instance is held pre-init so [activeAccelerator] stays null-safe.
  PalmDetectorWeb _palm = PalmDetectorWeb();
  final HandLandmarkModelWeb _landmark = HandLandmarkModelWeb();
  GestureRecognizerWeb? _gesture;

  bool _palmReady = false;
  bool _landmarkReady = false;

  HandMode _mode = HandMode.boxesAndLandmarks;
  int _maxDetections = 10;
  double _minLandmarkScore = 0.5;
  bool _enableGestures = false;
  double _gestureMinConfidence = 0.5;

  bool get isReady => _palmReady && _landmarkReady;
  bool get isInitialized => isReady;

  /// Active accelerator (`'webgpu'` / `'wasm'`) across the runners, or null
  /// pre-init. May change at runtime if a GPU error triggers a WASM swap.
  @override
  String? get activeAccelerator =>
      _palm.activeAccelerator ??
      _landmark.activeAccelerator ??
      _gesture?.activeAccelerator;

  Future<void> initialize({
    HandMode mode = HandMode.boxesAndLandmarks,
    HandLandmarkModel landmarkModel = HandLandmarkModel.full,
    double detectorConf = 0.45,
    double palmNmsIou = 0.45,
    double palmRoiScale = 2.6,
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    // enableTracking / trackingConfig are accepted for cross-platform API
    // parity; the web implementation runs palm detection every frame and does
    // not implement ROI tracking yet, so trackingConfig has no effect here.
    bool enableTracking = false,
    TrackingConfig trackingConfig = const TrackingConfig(),
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    bool enableGestures = false,
    double gestureMinConfidence = 0.5,
    bool useCompiledModel = false,
    String liteRtAccelerator = 'auto',
  }) async {
    if (isReady) {
      throw StateError('HandDetector already initialized');
    }
    _mode = mode;
    _maxDetections = maxDetections;
    _minLandmarkScore = minLandmarkScore;
    _enableGestures = enableGestures;
    _gestureMinConfidence = gestureMinConfidence;

    _palm = PalmDetectorWeb(
      scoreThreshold: detectorConf,
      nmsIouThreshold: palmNmsIou,
      roiScale: palmRoiScale,
    );
    await _palm.initialize(liteRtAccelerator: liteRtAccelerator);
    _palmReady = true;
    if (mode == HandMode.boxesAndLandmarks) {
      await _landmark.initialize(liteRtAccelerator: liteRtAccelerator);
    }
    _landmarkReady = true;

    if (enableGestures && mode == HandMode.boxesAndLandmarks) {
      _gesture = GestureRecognizerWeb(minConfidence: gestureMinConfidence);
      await _gesture!.initialize(liteRtAccelerator: liteRtAccelerator);
    }
  }

  /// No-op on web (accepted for cross-platform API parity with the native
  /// implementation's MediaPipe-style tracking; see `enableTracking`).
  Future<void> resetTracking() async {}

  @override
  Future<void> swapToWasm() async {
    try {
      await _palm.dispose();
      await _landmark.dispose();
      await _gesture?.dispose();
    } catch (_) {
      // Best-effort: a runner that already errored may not dispose cleanly.
    }
    await _palm.initialize(liteRtAccelerator: 'wasm');
    if (_mode == HandMode.boxesAndLandmarks) {
      await _landmark.initialize(liteRtAccelerator: 'wasm');
    }
    if (_enableGestures && _mode == HandMode.boxesAndLandmarks) {
      _gesture = GestureRecognizerWeb(minConfidence: _gestureMinConfidence);
      await _gesture!.initialize(liteRtAccelerator: 'wasm');
    }
  }

  Future<void> dispose() async {
    await _palm.dispose();
    await _landmark.dispose();
    await _gesture?.dispose();
    _gesture = null;
    _palmReady = false;
    _landmarkReady = false;
  }

  /// Detects hands in encoded image bytes (JPEG/PNG/...).
  Future<List<Hand>> detect(Uint8List imageBytes) async {
    if (!isReady) {
      throw StateError(
        'HandDetector not initialized. Call initialize() before using.',
      );
    }
    return withFallback(() => _detectFromBytesInner(imageBytes));
  }

  Future<List<Hand>> _detectFromBytesInner(Uint8List imageBytes) async {
    final web.ImageBitmap? bitmap = await decodeBitmap(imageBytes);
    if (bitmap == null) return const <Hand>[];
    try {
      return await _runPipeline(
        bitmap,
        imageWidth: bitmap.width,
        imageHeight: bitmap.height,
      );
    } finally {
      bitmap.close();
    }
  }

  /// Detects hands from a live `<video>` element (webcam feed).
  Future<List<Hand>> detectFromVideo(web.HTMLVideoElement video) async {
    if (!isReady) {
      throw StateError(
        'HandDetector not initialized. Call initialize() before using.',
      );
    }
    final int width = video.videoWidth;
    final int height = video.videoHeight;
    if (width == 0 || height == 0) return const <Hand>[];
    return withFallback(
      () => _runPipeline(video, imageWidth: width, imageHeight: height),
    );
  }

  Future<List<Hand>> _runPipeline(
    JSObject source, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    final List<PalmDetection> palms = await _palm.detect(
      source,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
    );
    if (palms.isEmpty) return const <Hand>[];

    final limited = palms.length > _maxDetections
        ? palms.sublist(0, _maxDetections)
        : palms;

    if (_mode == HandMode.boxes) {
      return [
        for (final palm in limited)
          _buildBoxOnlyHand(palm, imageWidth, imageHeight),
      ];
    }

    final results = <Hand>[];
    for (final palm in limited) {
      final double cx = palm.sqnRrCenterX * imageWidth;
      final double cy = palm.sqnRrCenterY * imageHeight;
      final double size =
          palm.sqnRrSize * math.max(imageWidth, imageHeight).toDouble();
      final double theta = palm.rotation;
      if (size <= 0) continue;

      final lm = await _landmark.runOnCrop(
        source,
        cx: cx,
        cy: cy,
        size: size,
        theta: theta,
      );
      if (lm.score < _minLandmarkScore) continue;

      final handedness =
          lm.handedness > 0.5 ? Handedness.right : Handedness.left;
      final landmarks = _transformLandmarks(
        lm.landmarks,
        cx: cx,
        cy: cy,
        size: size,
        theta: theta,
        inW: _landmark.inputWidth,
        inH: _landmark.inputHeight,
        score: lm.score,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
      );
      final worldLandmarks = _worldLandmarks(lm.world, lm.score);

      GestureResult? gesture;
      if (_gesture != null && _gesture!.isInitialized) {
        gesture = await _gesture!.recognize(
          landmarks: landmarks,
          worldLandmarks: worldLandmarks,
          handedness: handedness,
          imageWidth: imageWidth,
          imageHeight: imageHeight,
        );
      }

      final double half = size / 2;
      results.add(Hand(
        boundingBox: _clampedBox(cx, cy, half, imageWidth, imageHeight),
        score: palm.score,
        landmarks: landmarks,
        imageWidth: imageWidth,
        imageHeight: imageHeight,
        handedness: handedness,
        rotation: theta,
        rotatedCenterX: cx,
        rotatedCenterY: cy,
        rotatedSize: size,
        gesture: gesture,
      ));
    }
    return results;
  }

  Hand _buildBoxOnlyHand(PalmDetection palm, int imageWidth, int imageHeight) {
    final double cx = palm.sqnRrCenterX * imageWidth;
    final double cy = palm.sqnRrCenterY * imageHeight;
    final double size =
        palm.sqnRrSize * math.max(imageWidth, imageHeight).toDouble();
    return Hand(
      boundingBox: _clampedBox(cx, cy, size / 2, imageWidth, imageHeight),
      score: palm.score,
      landmarks: const <HandLandmark>[],
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      handedness: null,
      rotation: palm.rotation,
      rotatedCenterX: cx,
      rotatedCenterY: cy,
      rotatedSize: size,
    );
  }

  /// Inverts the Canvas crop transform applied by [HandLandmarkModelWeb], mapping
  /// model-space landmarks (input pixels) back to original-image coordinates.
  List<HandLandmark> _transformLandmarks(
    Float32List flat, {
    required double cx,
    required double cy,
    required double size,
    required double theta,
    required int inW,
    required int inH,
    required double score,
    required int imageWidth,
    required int imageHeight,
  }) {
    final double ct = math.cos(theta);
    final double st = math.sin(theta);
    final double scale = size / inW;
    final out = <HandLandmark>[];
    for (int i = 0; i < numHandLandmarks; i++) {
      final double mx = flat[i * 3] - inW / 2.0;
      final double my = flat[i * 3 + 1] - inH / 2.0;
      final double mz = flat[i * 3 + 2];
      final double rx = ct * mx - st * my;
      final double ry = st * mx + ct * my;
      out.add(HandLandmark(
        type: HandLandmarkType.values[i],
        x: (cx + rx * scale).clamp(0.0, imageWidth.toDouble()),
        y: (cy + ry * scale).clamp(0.0, imageHeight.toDouble()),
        z: mz,
        visibility: score,
      ));
    }
    return out;
  }

  List<HandLandmark> _worldLandmarks(Float32List flat, double score) {
    final out = <HandLandmark>[];
    for (int i = 0; i < numHandLandmarks; i++) {
      final base = i * 3;
      out.add(HandLandmark(
        type: HandLandmarkType.values[i],
        x: flat[base],
        y: flat[base + 1],
        z: flat[base + 2],
        visibility: score,
      ));
    }
    return out;
  }

  BoundingBox _clampedBox(
    double cx,
    double cy,
    double half,
    int imgW,
    int imgH,
  ) {
    return BoundingBox.ltrb(
      (cx - half).clamp(0.0, imgW.toDouble()),
      (cy - half).clamp(0.0, imgH.toDouble()),
      (cx + half).clamp(0.0, imgW.toDouble()),
      (cy + half).clamp(0.0, imgH.toDouble()),
    );
  }

  // ---- API parity stubs that throw on web -----------------------------------

  Future<List<Hand>> detectFromFilepath(String path) {
    throw UnsupportedError(
      'detectFromFilepath is not supported on web. Use detect(bytes) instead.',
    );
  }

  Future<List<Hand>> detectFromMat(Object image) {
    throw UnsupportedError(
      'detectFromMat is not supported on web. Use detect(bytes) instead.',
    );
  }

  Future<List<Hand>> detectFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
  }) {
    throw UnsupportedError('detectFromMatBytes is not supported on web.');
  }

  Future<List<Hand>> detectFromCameraFrame(Object frame, {int? maxDim}) {
    throw UnsupportedError('detectFromCameraFrame is not supported on web.');
  }

  Future<List<Hand>> detectFromCameraImage(
    Object cameraImage, {
    Object? rotation,
    bool? isBgra,
    int? maxDim,
  }) {
    throw UnsupportedError(
      'detectFromCameraImage is not supported on web. Use detectFromVideo '
      'with an HTMLVideoElement instead.',
    );
  }
}
