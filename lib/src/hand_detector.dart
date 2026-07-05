import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_litert/native.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'types.dart';
import 'isolate/hand_detector_core.dart';
import 'util/image_utils.dart';

/// Startup payload transferred to the background isolate via [Isolate.spawn].
class _DetectionIsolateStartupData {
  final SendPort sendPort;
  final TransferableTypedData palmDetectionBytes;
  final TransferableTypedData handLandmarkBytes;
  final TransferableTypedData? gestureEmbedderBytes;
  final TransferableTypedData? gestureClassifierBytes;
  final String modeName;
  final String landmarkModelName;
  final double detectorConf;
  final double palmNmsIou;
  final double palmRoiScale;
  final int maxDetections;
  final double minLandmarkScore;
  final bool enableTracking;
  final TrackingConfig trackingConfig;
  final int interpreterPoolSize;
  final String performanceModeName;
  final int? numThreads;
  final bool enableGestures;
  final double gestureMinConfidence;
  final bool useCompiledModel;
  final List<int> acceleratorIndices;
  final int precisionIndex;

  _DetectionIsolateStartupData({
    required this.sendPort,
    required this.palmDetectionBytes,
    required this.handLandmarkBytes,
    this.gestureEmbedderBytes,
    this.gestureClassifierBytes,
    required this.modeName,
    required this.landmarkModelName,
    required this.detectorConf,
    required this.palmNmsIou,
    required this.palmRoiScale,
    required this.maxDetections,
    required this.minLandmarkScore,
    required this.enableTracking,
    required this.trackingConfig,
    required this.interpreterPoolSize,
    required this.performanceModeName,
    required this.numThreads,
    required this.enableGestures,
    required this.gestureMinConfidence,
    required this.useCompiledModel,
    required this.acceleratorIndices,
    required this.precisionIndex,
  });
}

/// On-device hand detection and landmark estimation using TensorFlow Lite.
///
/// Implements a two-stage pipeline based on MediaPipe:
/// 1. Palm detection using SSD-based detector with rotation rectangle output
/// 2. Hand landmark model to extract 21 keypoints per detected hand
///
/// All inference runs in a background isolate, keeping the UI thread free.
///
/// ## Usage
///
/// ```dart
/// // One-step construction
/// final detector = await HandDetector.create();
///
/// // Or two-step, if you need to configure between construction and init
/// final detector = HandDetector();
/// await detector.initialize();
///
/// final hands = await detector.detect(imageBytes);
/// await detector.dispose();
/// ```
class HandDetector {
  static const String _packageVersion = '3.0.0';
  static const String _pipelineVersion = 'pipeline_v1';

  /// Version key for the default hand detection pipeline.
  ///
  /// Downstream caches can use this to invalidate stored detections when model
  /// weights, preprocessing, post-processing, thresholds, or coordinate
  /// conventions change.
  static const String modelVersion =
      'hand_detection:$_packageVersion:mode=boxesAndLandmarks:'
      'landmarkModel=full:gestures=false:$_pipelineVersion';

  /// Builds a version key for a specific hand detector configuration.
  static String modelVersionFor({
    HandMode mode = HandMode.boxesAndLandmarks,
    HandLandmarkModel landmarkModel = HandLandmarkModel.full,
    bool enableGestures = false,
  }) {
    return 'hand_detection:$_packageVersion:mode=${mode.name}:'
        'landmarkModel=${landmarkModel.name}:gestures=$enableGestures:'
        '$_pipelineVersion';
  }

  _HandDetectorWorker? _worker;

  /// Creates a hand detector instance.
  ///
  /// The detector is not ready for use until you call [initialize].
  HandDetector();

  /// Creates and initializes a hand detector in one step.
  ///
  /// Convenience factory that combines [HandDetector.new] and [initialize].
  /// Accepts the same parameters as [initialize].
  ///
  /// Example:
  /// ```dart
  /// final detector = await HandDetector.create();
  ///
  /// // Equivalent to:
  /// final detector = HandDetector();
  /// await detector.initialize();
  /// ```
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
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
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
      interpreterPoolSize: interpreterPoolSize,
      performanceConfig: performanceConfig,
      enableGestures: enableGestures,
      gestureMinConfidence: gestureMinConfidence,
      useCompiledModel: useCompiledModel,
      liteRtAccelerator: liteRtAccelerator,
      accelerators: accelerators,
      precision: precision,
    );
    return detector;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  ///
  /// You must call [initialize] before this returns true.
  bool get isReady => _worker?.isReady ?? false;

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => isReady;

  /// Active inference backend label. On native this is always null (the engine
  /// is selected via [useCompiledModel] / [PerformanceConfig], not a LiteRT.js
  /// accelerator); the web implementation reports `'webgpu'` / `'wasm'`. Kept
  /// for cross-platform API parity so the same code compiles on every platform.
  String? get activeAccelerator => null;

  /// Initializes the hand detector by loading TensorFlow Lite models.
  ///
  /// Must be called before [detect] or [detectFromMat].
  /// Calling [initialize] twice without [dispose] throws [StateError].
  ///
  /// Parameters:
  /// - [mode]: Detection mode (boxes only or boxes + landmarks). Default: [HandMode.boxesAndLandmarks]
  /// - [landmarkModel]: Hand landmark model variant. Default: [HandLandmarkModel.full]
  /// - [detectorConf]: Palm detection confidence threshold (0.0-1.0). Default: 0.45
  /// - [palmNmsIou]: IoU threshold for palm non-maximum suppression (0.0-1.0).
  ///   Higher keeps more overlapping detections; lower merges them harder. Default: 0.45
  /// - [palmRoiScale]: Expansion factor for the palm ROI fed to the landmark
  ///   model. Larger includes more context around the palm. Default: 2.6
  /// - [maxDetections]: Maximum number of hands to detect. Default: 10
  /// - [minLandmarkScore]: Minimum landmark confidence score (0.0-1.0). Default: 0.5
  /// - [interpreterPoolSize]: Number of landmark model interpreter instances (1-10). Default: 1.
  ///   Forced to 1 when a performance delegate (XNNPACK/auto) is enabled.
  /// - [performanceConfig]: TensorFlow Lite performance configuration. Default: auto (optimal per platform)
  /// - [enableGestures]: Whether to run gesture recognition. Default: false
  /// - [gestureMinConfidence]: Minimum confidence for gesture recognition (0.0-1.0). Default: 0.5
  /// - [useCompiledModel]: Use the LiteRT Next [CompiledModel] engine (GPU with
  ///   CPU fallback) instead of the classic Interpreter engine. Default: false.
  ///   When enabled, the landmark model runs on a pool of [interpreterPoolSize]
  ///   CompiledModel instances (default 1) — one model per concurrently-detected
  ///   hand, each with its own input buffer.
  /// - [enableTracking]: Enable MediaPipe-style detection + tracking. When true,
  ///   each detected hand is followed frame-to-frame via a landmark-derived
  ///   region of interest, and the palm detector only runs to find new hands.
  ///   This greatly reduces overlay drop-outs on video. Call [resetTracking]
  ///   between unrelated inputs. Default: false (palm detection every frame).
  /// - [trackingConfig]: Tuning for the ROI carried between frames when
  ///   [enableTracking] is true (expansion, shift, association IoU, size bounds).
  ///   Ignored when tracking is off. Default: [TrackingConfig] (MediaPipe values).
  Future<void> initialize({
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
    // Web-only (LiteRT.js accelerator); accepted for cross-platform API parity
    // and ignored on native, which selects its engine via useCompiledModel.
    String liteRtAccelerator = 'auto',
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
    if (isReady) {
      throw StateError('HandDetector already initialized');
    }

    const palmPath =
        'packages/hand_detection/assets/models/hand_detection.tflite';
    const landmarkPath =
        'packages/hand_detection/assets/models/hand_landmark_full.tflite';

    final assetFutures = <Future<ByteData>>[
      rootBundle.load(palmPath),
      rootBundle.load(landmarkPath),
    ];

    if (enableGestures) {
      const embedderPath =
          'packages/hand_detection/assets/models/gesture_embedder.tflite';
      const classifierPath =
          'packages/hand_detection/assets/models/canned_gesture_classifier.tflite';
      assetFutures.add(rootBundle.load(embedderPath));
      assetFutures.add(rootBundle.load(classifierPath));
    }

    final results = await Future.wait(assetFutures);

    final palmBytes = results[0].buffer.asUint8List();
    final landmarkBytes = results[1].buffer.asUint8List();

    Uint8List? gestureEmbedderBytes;
    Uint8List? gestureClassifierBytes;
    if (enableGestures && results.length > 2) {
      gestureEmbedderBytes = results[2].buffer.asUint8List();
      gestureClassifierBytes = results[3].buffer.asUint8List();
    }

    final effectivePoolSize = performanceConfig.mode == PerformanceMode.disabled
        ? interpreterPoolSize
        : 1;

    await initializeFromBuffers(
      palmDetectionBytes: palmBytes,
      handLandmarkBytes: landmarkBytes,
      gestureEmbedderBytes: gestureEmbedderBytes,
      gestureClassifierBytes: gestureClassifierBytes,
      mode: mode,
      landmarkModel: landmarkModel,
      detectorConf: detectorConf,
      palmNmsIou: palmNmsIou,
      palmRoiScale: palmRoiScale,
      maxDetections: maxDetections,
      minLandmarkScore: minLandmarkScore,
      enableTracking: enableTracking,
      trackingConfig: trackingConfig,
      interpreterPoolSize: effectivePoolSize,
      performanceConfig: performanceConfig,
      enableGestures: enableGestures,
      gestureMinConfidence: gestureMinConfidence,
      useCompiledModel: useCompiledModel,
      accelerators: accelerators,
      precision: precision,
    );
  }

  /// Initializes the hand detector from pre-loaded model bytes.
  ///
  /// Used when asset loading from the main isolate is not available, or when
  /// bytes have already been loaded. Spawns the background isolate with the
  /// provided model data.
  ///
  /// Parameters:
  /// - [palmDetectionBytes]: Raw bytes of the palm detection TFLite model
  /// - [handLandmarkBytes]: Raw bytes of the hand landmark TFLite model
  /// - [gestureEmbedderBytes]: Raw bytes of the gesture embedder model (optional; required for gesture recognition)
  /// - [gestureClassifierBytes]: Raw bytes of the gesture classifier model (optional; required for gesture recognition)
  /// - [mode]: Detection mode. Default: [HandMode.boxesAndLandmarks]
  /// - [landmarkModel]: Hand landmark model variant. Default: [HandLandmarkModel.full]
  /// - [detectorConf]: Palm detection confidence threshold. Default: 0.45
  /// - [palmNmsIou]: IoU threshold for palm non-maximum suppression. Default: 0.45
  /// - [palmRoiScale]: Expansion factor for the palm ROI fed to the landmark model. Default: 2.6
  /// - [trackingConfig]: ROI tuning used when [enableTracking] is true. Default: [TrackingConfig]
  /// - [maxDetections]: Maximum number of hands to detect. Default: 10
  /// - [minLandmarkScore]: Minimum landmark confidence score. Default: 0.5
  /// - [interpreterPoolSize]: Number of landmark model interpreter instances. Default: 1.
  ///   Forced to 1 when a performance delegate (XNNPACK/auto) is enabled.
  /// - [performanceConfig]: TensorFlow Lite performance configuration. Default: auto
  /// - [enableGestures]: Whether to run gesture recognition. Default: false
  /// - [gestureMinConfidence]: Minimum confidence for gesture recognition. Default: 0.5
  Future<void> initializeFromBuffers({
    required Uint8List palmDetectionBytes,
    required Uint8List handLandmarkBytes,
    Uint8List? gestureEmbedderBytes,
    Uint8List? gestureClassifierBytes,
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
    Set<Accelerator> accelerators = const {Accelerator.gpu, Accelerator.cpu},
    Precision precision = Precision.fp16,
  }) async {
    if (isReady) {
      throw StateError('HandDetector already initialized');
    }

    final effectivePoolSize = performanceConfig.mode == PerformanceMode.disabled
        ? interpreterPoolSize
        : 1;

    final worker = _HandDetectorWorker();

    try {
      await worker.initialize(
        palmDetectionBytes: palmDetectionBytes,
        handLandmarkBytes: handLandmarkBytes,
        gestureEmbedderBytes: gestureEmbedderBytes,
        gestureClassifierBytes: gestureClassifierBytes,
        mode: mode,
        landmarkModel: landmarkModel,
        detectorConf: detectorConf,
        palmNmsIou: palmNmsIou,
        palmRoiScale: palmRoiScale,
        maxDetections: maxDetections,
        minLandmarkScore: minLandmarkScore,
        enableTracking: enableTracking,
        trackingConfig: trackingConfig,
        interpreterPoolSize: effectivePoolSize,
        performanceConfig: performanceConfig,
        enableGestures: enableGestures,
        gestureMinConfidence: gestureMinConfidence,
        useCompiledModel: useCompiledModel,
        accelerators: accelerators,
        precision: precision,
      );
    } catch (e) {
      if (worker.isReady) {
        await worker.dispose();
      }
      rethrow;
    }

    _worker = worker;
  }

  /// Detects hands in an image from raw bytes.
  ///
  /// Decodes the image bytes using OpenCV and performs hand detection in a
  /// background isolate.
  ///
  /// Parameters:
  /// - [imageBytes]: Raw image data in a supported format (JPEG, PNG, etc.)
  ///
  /// Returns a list of [Hand] objects, one per detected hand.
  ///
  /// Throws [StateError] if called before [initialize].
  /// Throws [FormatException] if the image bytes cannot be decoded.
  Future<List<Hand>> detect(Uint8List imageBytes) async {
    if (!isReady) {
      throw StateError(
          'HandDetector not initialized. Call initialize() first.');
    }
    final List<dynamic> result;
    try {
      result = await _worker!.sendRequest<List<dynamic>>(
        'detect',
        {
          'bytes': TransferableTypedData.fromList([imageBytes]),
        },
      );
    } catch (e) {
      rethrowOrFormatException(e, imageBytes);
    }
    return _deserializeHands(result);
  }

  /// Clears the MediaPipe-style cross-frame tracking state (see [initialize]'s
  /// `enableTracking`).
  ///
  /// Call this between unrelated inputs, for example before processing a new
  /// video or when switching to independent still images, so a stale region of
  /// interest from a previous frame is not reused. Safe to call when tracking
  /// was never enabled.
  Future<void> resetTracking() async {
    final worker = _worker;
    if (worker == null) return;
    await worker
        .sendRequest<Object?>('resetTracking', const <String, dynamic>{});
  }

  /// Detects hands in an image file at [path].
  ///
  /// Convenience wrapper that reads the file and calls [detect].
  /// Not available on Flutter Web (uses `dart:io`).
  ///
  /// Throws [StateError] if [initialize] has not been called successfully.
  /// Throws [FileSystemException] if the file cannot be read.
  Future<List<Hand>> detectFromFilepath(String path) async {
    final bytes = await File(path).readAsBytes();
    return detect(bytes);
  }

  /// Detects hands in a pre-decoded [cv.Mat] image.
  ///
  /// The Mat's raw pixel data is extracted and transferred to the background
  /// isolate using zero-copy [TransferableTypedData]. The original Mat is NOT
  /// disposed by this method; the caller is responsible for disposal.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Hand>> detectFromMat(cv.Mat image) {
    if (!isReady) {
      throw StateError(
          'HandDetector not initialized. Call initialize() first.');
    }
    // A non-continuous Mat (e.g. a region()/ROI view) yields scrambled bytes
    // from .data, which reads total*elemSize contiguous bytes and ignores row
    // stride. Clone to a continuous copy first; detectFromMatBytes copies the
    // bytes into a TransferableTypedData synchronously, so the clone can be
    // disposed immediately after.
    final cv.Mat src = image.isContinuous ? image : image.clone();
    final result = detectFromMatBytes(
      src.data,
      width: src.cols,
      height: src.rows,
      matType: src.type.value,
    );
    if (!identical(src, image)) src.dispose();
    return result;
  }

  /// Detects hands from raw pixel bytes without constructing a [cv.Mat] first.
  ///
  /// This avoids the overhead of building a Mat on the calling thread:
  /// the bytes are transferred via zero-copy [TransferableTypedData] and the
  /// Mat is reconstructed inside the background isolate.
  ///
  /// Parameters:
  /// - [bytes]: Raw pixel data (typically BGR format, 3 bytes per pixel)
  /// - [width]: Image width in pixels
  /// - [height]: Image height in pixels
  /// - [matType]: OpenCV MatType value (default: CV_8UC3 = 16 for BGR)
  ///
  /// Returns a list of [Hand] objects, one per detected hand.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Hand>> detectFromMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
  }) async {
    if (!isReady) {
      throw StateError(
          'HandDetector not initialized. Call initialize() first.');
    }
    final List<dynamic> result = await _worker!.sendRequest<List<dynamic>>(
      'detectMat',
      {
        'bytes': TransferableTypedData.fromList([bytes]),
        'width': width,
        'height': height,
        'matType': matType,
      },
    );
    return _deserializeHands(result);
  }

  /// Detects hands directly from a [CameraFrame] produced by
  /// [prepareCameraFrame].
  ///
  /// The frame's YUV→BGR colour conversion and any optional rotation happen
  /// inside the detection isolate, not on the calling thread. Use this from a
  /// `CameraController.startImageStream` callback to keep the UI thread free
  /// of OpenCV work.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Hand>> detectFromCameraFrame(
    CameraFrame frame, {
    int? maxDim,
  }) async {
    if (!isReady) {
      throw StateError(
          'HandDetector not initialized. Call initialize() first.');
    }
    final List<dynamic> result = await _worker!.sendRequest<List<dynamic>>(
      'detectCameraFrame',
      cameraFrameRpcFields(frame, {'maxDim': maxDim}),
    );
    return _deserializeHands(result);
  }

  /// One-call wrapper for live camera streams: takes a `CameraImage`-shaped
  /// object directly (any object exposing `width`, `height`, and `planes` with
  /// `bytes` / `bytesPerRow` / `bytesPerPixel`) and runs YUV packing, colour
  /// conversion, rotation, and downscale in the detection isolate, all off
  /// the UI thread.
  ///
  /// Returns an empty list (not an error) when the plane shape can't be
  /// decoded. Throws at runtime if [cameraImage] doesn't expose the expected
  /// shape.
  ///
  /// [isBgra] selects BGRA vs. RGBA for the desktop single-plane path; ignored
  /// for YUV input (Android/iOS). Defaults to `true` on macOS (BGRA) and
  /// `false` on Windows/Linux (RGBA). Only pass this explicitly if you are
  /// using a non-standard camera plugin that delivers a different format.
  ///
  /// Throws [StateError] if [initialize] has not been called.
  Future<List<Hand>> detectFromCameraImage(
    Object cameraImage, {
    CameraFrameRotation? rotation,
    bool? isBgra,
    int? maxDim,
  }) async {
    if (!isReady) {
      throw StateError(
          'HandDetector not initialized. Call initialize() first.');
    }
    final frame = prepareCameraFrameFromImage(
      cameraImage,
      rotation: rotation,
      isBgra: isBgra ?? Platform.isMacOS,
    );
    if (frame == null) return const <Hand>[];
    return detectFromCameraFrame(frame, maxDim: maxDim);
  }

  /// Detects hands in an OpenCV Mat image.
  ///
  /// Deprecated: Use [detectFromMat] instead.
  @Deprecated('Use detectFromMat instead. Will be removed in a future release.')
  Future<List<Hand>> detectOnMat(cv.Mat image) => detectFromMat(image);

  /// Detects hands from raw pixel bytes without constructing a [cv.Mat] first.
  ///
  /// Deprecated: Use [detectFromMatBytes] instead.
  @Deprecated(
      'Use detectFromMatBytes instead. Will be removed in a future release.')
  Future<List<Hand>> detectOnMatBytes(
    Uint8List bytes, {
    required int width,
    required int height,
    int matType = 16,
  }) =>
      detectFromMatBytes(bytes, width: width, height: height, matType: matType);

  /// Releases all resources used by the detector.
  Future<void> dispose() async {
    final worker = _worker;
    _worker = null;
    if (worker == null) return;

    // Graceful shutdown via the shared base: sends 'dispose' as an RPC and
    // awaits the ack before force-killing the isolate, so it can free its
    // native TFLite interpreters (~10-26MB/detector on Android) instead of
    // being reaped mid-cleanup by Isolate.kill(priority: immediate).
    await worker.disposeGracefully();
  }

  List<Hand> _deserializeHands(List<dynamic> result) => result
      .map((map) => Hand.fromMap(Map<String, dynamic>.from(map as Map)))
      .toList();

  /// Isolate entry point: initializes [HandDetectorCore] and listens for detection requests.
  @pragma('vm:entry-point')
  static void _detectionIsolateEntry(_DetectionIsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    HandDetectorCore? core;

    try {
      final palmBytes = data.palmDetectionBytes.materialize().asUint8List();
      final landmarkBytes = data.handLandmarkBytes.materialize().asUint8List();

      Uint8List? embedderBytes;
      Uint8List? classifierBytes;
      if (data.gestureEmbedderBytes != null) {
        embedderBytes = data.gestureEmbedderBytes!.materialize().asUint8List();
      }
      if (data.gestureClassifierBytes != null) {
        classifierBytes =
            data.gestureClassifierBytes!.materialize().asUint8List();
      }

      final mode = HandMode.values.firstWhere(
        (m) => m.name == data.modeName,
      );
      final performanceMode = PerformanceMode.values.firstWhere(
        (m) => m.name == data.performanceModeName,
      );

      final accelerators =
          data.acceleratorIndices.map((i) => Accelerator.values[i]).toSet();
      final precision = Precision.values[data.precisionIndex];

      core = HandDetectorCore();
      await core.initializeFromBuffers(
        palmDetectionBytes: palmBytes,
        handLandmarkBytes: landmarkBytes,
        gestureEmbedderBytes: embedderBytes,
        gestureClassifierBytes: classifierBytes,
        mode: mode,
        maxDetections: data.maxDetections,
        minLandmarkScore: data.minLandmarkScore,
        detectorConf: data.detectorConf,
        palmNmsIou: data.palmNmsIou,
        palmRoiScale: data.palmRoiScale,
        enableTracking: data.enableTracking,
        trackingConfig: data.trackingConfig,
        interpreterPoolSize: data.interpreterPoolSize,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
        enableGestures: data.enableGestures,
        gestureMinConfidence: data.gestureMinConfidence,
        useCompiledModel: data.useCompiledModel,
        accelerators: accelerators,
        precision: precision,
      );

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({
        'error': 'Hand detection isolate initialization failed: $e\n$st',
      });
      return;
    }

    Future<Object?> detectMat(cv.Mat mat) async {
      // core-null check is INSIDE the try so the finally still disposes the Mat
      // the handler already constructed (avoids a leak on the core-null edge).
      try {
        final HandDetectorCore? c = core;
        if (c == null) {
          throw StateError('HandDetectorCore not initialized in isolate');
        }
        final hands = await c.detectDirect(mat);
        return hands.map((h) => h.toMap()).toList();
      } finally {
        mat.dispose();
      }
    }

    serveIsolateRpc(
      mainSendPort: mainSendPort,
      receivePort: workerReceivePort,
      handlers: {
        'detect': (message) {
          final ByteBuffer bb =
              (message['bytes'] as TransferableTypedData).materialize();
          final mat = cv.imdecode(bb.asUint8List(), cv.IMREAD_COLOR);
          if (mat.isEmpty) {
            mat.dispose();
            throwDecodeFailure();
          }
          return detectMat(mat);
        },
        'detectMat': (message) {
          final ByteBuffer bb =
              (message['bytes'] as TransferableTypedData).materialize();
          final matType = cv.MatType(message['matType'] as int);
          return detectMat(ImageUtils.matFromPackedBytes(
            message['height'] as int,
            message['width'] as int,
            matType,
            bb.asUint8List(),
          ));
        },
        'detectCameraFrame': (message) {
          final Uint8List frameBytes =
              (message['bytes'] as TransferableTypedData)
                  .materialize()
                  .asUint8List();
          return detectMat(_matFromCameraFrameMessage(message, frameBytes));
        },
        'resetTracking': (message) {
          core?.resetTracking();
          return Future<Object?>.value(true);
        },
      },
      onDispose: () async {
        await core?.dispose();
        core = null;
      },
    );
  }

  /// Decodes a [CameraFrame] isolate message into a 3-channel BGR [cv.Mat],
  /// running all OpenCV work inside the detection isolate. Reconstructs the
  /// [CameraFrame] from the message and delegates to the shared
  /// [ImageUtils.cameraFrameToBgrMat] (the same backend-neutral decode plan +
  /// op ordering the face and pose detectors use).
  static cv.Mat _matFromCameraFrameMessage(Map message, Uint8List bytes) {
    return ImageUtils.cameraFrameToBgrMat(
      cameraFrameFromRpcMessage(message, bytes),
      maxDim: message['maxDim'] as int?,
    );
  }
}

class _HandDetectorWorker extends IsolateWorkerBase {
  @override
  String get workerDisposeOp => 'dispose';

  Future<void> initialize({
    required Uint8List palmDetectionBytes,
    required Uint8List handLandmarkBytes,
    Uint8List? gestureEmbedderBytes,
    Uint8List? gestureClassifierBytes,
    required HandMode mode,
    required HandLandmarkModel landmarkModel,
    required double detectorConf,
    required double palmNmsIou,
    required double palmRoiScale,
    required int maxDetections,
    required double minLandmarkScore,
    required bool enableTracking,
    required TrackingConfig trackingConfig,
    required int interpreterPoolSize,
    required PerformanceConfig performanceConfig,
    required bool enableGestures,
    required double gestureMinConfidence,
    required bool useCompiledModel,
    required Set<Accelerator> accelerators,
    required Precision precision,
  }) async {
    TransferableTypedData? gestureEmbedderData;
    TransferableTypedData? gestureClassifierData;
    if (gestureEmbedderBytes != null) {
      gestureEmbedderData =
          TransferableTypedData.fromList([gestureEmbedderBytes]);
    }
    if (gestureClassifierBytes != null) {
      gestureClassifierData =
          TransferableTypedData.fromList([gestureClassifierBytes]);
    }

    await initWorker(
      (sendPort) => Isolate.spawn(
        HandDetector._detectionIsolateEntry,
        _DetectionIsolateStartupData(
          sendPort: sendPort,
          palmDetectionBytes:
              TransferableTypedData.fromList([palmDetectionBytes]),
          handLandmarkBytes:
              TransferableTypedData.fromList([handLandmarkBytes]),
          gestureEmbedderBytes: gestureEmbedderData,
          gestureClassifierBytes: gestureClassifierData,
          modeName: mode.name,
          landmarkModelName: landmarkModel.name,
          detectorConf: detectorConf,
          palmNmsIou: palmNmsIou,
          palmRoiScale: palmRoiScale,
          maxDetections: maxDetections,
          minLandmarkScore: minLandmarkScore,
          enableTracking: enableTracking,
          trackingConfig: trackingConfig,
          interpreterPoolSize: interpreterPoolSize,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
          enableGestures: enableGestures,
          gestureMinConfidence: gestureMinConfidence,
          useCompiledModel: useCompiledModel,
          acceleratorIndices: accelerators.map((a) => a.index).toList(),
          precisionIndex: precision.index,
        ),
        debugName: 'HandDetector',
      ),
      timeout: const Duration(seconds: 30),
      timeoutMessage: 'Hand detection isolate initialization timed out',
    );
  }
}
