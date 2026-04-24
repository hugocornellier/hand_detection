import 'dart:async';
import 'dart:io';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'types.dart';
import 'isolate/hand_detector_core.dart';

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
  final int maxDetections;
  final double minLandmarkScore;
  final int interpreterPoolSize;
  final String performanceModeName;
  final int? numThreads;
  final bool enableGestures;
  final double gestureMinConfidence;

  _DetectionIsolateStartupData({
    required this.sendPort,
    required this.palmDetectionBytes,
    required this.handLandmarkBytes,
    this.gestureEmbedderBytes,
    this.gestureClassifierBytes,
    required this.modeName,
    required this.landmarkModelName,
    required this.detectorConf,
    required this.maxDetections,
    required this.minLandmarkScore,
    required this.interpreterPoolSize,
    required this.performanceModeName,
    required this.numThreads,
    required this.enableGestures,
    required this.gestureMinConfidence,
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
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    bool enableGestures = false,
    double gestureMinConfidence = 0.5,
  }) async {
    final detector = HandDetector();
    await detector.initialize(
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
    return detector;
  }

  /// Returns true if the detector has been initialized and is ready to use.
  ///
  /// You must call [initialize] before this returns true.
  bool get isReady => _worker?.isReady ?? false;

  /// Returns true if the detector has been initialized and is ready to use.
  bool get isInitialized => isReady;

  /// Initializes the hand detector by loading TensorFlow Lite models.
  ///
  /// Must be called before [detect] or [detectFromMat].
  /// Calling [initialize] twice without [dispose] throws [StateError].
  ///
  /// Parameters:
  /// - [mode]: Detection mode (boxes only or boxes + landmarks). Default: [HandMode.boxesAndLandmarks]
  /// - [landmarkModel]: Hand landmark model variant. Default: [HandLandmarkModel.full]
  /// - [detectorConf]: Palm detection confidence threshold (0.0-1.0). Default: 0.45
  /// - [maxDetections]: Maximum number of hands to detect. Default: 10
  /// - [minLandmarkScore]: Minimum landmark confidence score (0.0-1.0). Default: 0.5
  /// - [interpreterPoolSize]: Number of landmark model interpreter instances (1-10). Default: 1.
  ///   Forced to 1 when a performance delegate (XNNPACK/auto) is enabled.
  /// - [performanceConfig]: TensorFlow Lite performance configuration. Default: auto (optimal per platform)
  /// - [enableGestures]: Whether to run gesture recognition. Default: false
  /// - [gestureMinConfidence]: Minimum confidence for gesture recognition (0.0-1.0). Default: 0.5
  Future<void> initialize({
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
      maxDetections: maxDetections,
      minLandmarkScore: minLandmarkScore,
      interpreterPoolSize: effectivePoolSize,
      performanceConfig: performanceConfig,
      enableGestures: enableGestures,
      gestureMinConfidence: gestureMinConfidence,
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
    int maxDetections = 10,
    double minLandmarkScore = 0.5,
    int interpreterPoolSize = 1,
    PerformanceConfig performanceConfig = const PerformanceConfig(),
    bool enableGestures = false,
    double gestureMinConfidence = 0.5,
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
        maxDetections: maxDetections,
        minLandmarkScore: minLandmarkScore,
        interpreterPoolSize: effectivePoolSize,
        performanceConfig: performanceConfig,
        enableGestures: enableGestures,
        gestureMinConfidence: gestureMinConfidence,
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
    final List<dynamic> result = await _worker!.sendRequest<List<dynamic>>(
      'detect',
      {
        'bytes': TransferableTypedData.fromList([imageBytes]),
      },
    );
    return _deserializeHands(result);
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
    final int rows = image.rows;
    final int cols = image.cols;
    final int type = image.type.value;
    final Uint8List data = image.data;
    return detectFromMatBytes(data, width: cols, height: rows, matType: type);
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
      {
        'bytes': TransferableTypedData.fromList([frame.bytes]),
        'width': frame.width,
        'height': frame.height,
        'strideCols': frame.strideCols,
        'conversion': frame.conversion.index,
        'rotation': frame.rotation?.index,
        'maxDim': maxDim,
      },
    );
    return _deserializeHands(result);
  }

  /// One-call wrapper for live camera streams: takes a `CameraImage`-shaped
  /// object directly (any object exposing `width`, `height`, and `planes` with
  /// `bytes` / `bytesPerRow` / `bytesPerPixel`) and runs YUV packing, colour
  /// conversion, rotation, and downscale in the detection isolate — all off
  /// the UI thread.
  ///
  /// Returns an empty list (not an error) when the plane shape can't be
  /// decoded. Throws at runtime if [cameraImage] doesn't expose the expected
  /// shape.
  ///
  /// [isBgra] selects BGRA (macOS, default) vs. RGBA (Linux) for the desktop
  /// single-plane path; ignored for YUV input.
  ///
  /// Throws [StateError] if [initialize] has not been called.
  Future<List<Hand>> detectFromCameraImage(
    Object cameraImage, {
    CameraFrameRotation? rotation,
    bool isBgra = true,
    int? maxDim,
  }) async {
    if (!isReady) {
      throw StateError(
          'HandDetector not initialized. Call initialize() first.');
    }
    final frame = prepareCameraFrameFromImage(
      cameraImage,
      rotation: rotation,
      isBgra: isBgra,
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

    // Graceful shutdown: send 'dispose' as an RPC and await the ack before
    // letting the worker force-kill the isolate. flutter_litert's
    // IsolateWorkerBase calls Isolate.kill(priority: immediate) which races
    // past any queued 'dispose' message, so without this round-trip the
    // isolate dies before it can free its TFLite interpreters. On Android
    // each detector leaks ~10-26MB of native memory; under sequential
    // create/dispose load the low-memory killer reaps the test process.
    try {
      await worker.sendRequest<dynamic>('dispose',
          const <String, dynamic>{}).timeout(const Duration(seconds: 5));
    } catch (_) {
      // Best-effort: fall through to the force-kill below.
    }
    await worker.dispose();
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
        interpreterPoolSize: data.interpreterPoolSize,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
        enableGestures: data.enableGestures,
        gestureMinConfidence: data.gestureMinConfidence,
      );

      mainSendPort.send(workerReceivePort.sendPort);
    } catch (e, st) {
      mainSendPort.send({
        'error': 'Hand detection isolate initialization failed: $e\n$st',
      });
      return;
    }

    workerReceivePort.listen((message) async {
      if (message is! Map) return;

      final int? id = message['id'] as int?;
      final String? op = message['op'] as String?;

      if (id == null || op == null) return;

      try {
        switch (op) {
          case 'detect':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'HandDetectorCore not initialized in isolate',
              });
              return;
            }
            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final Uint8List imageBytes = bb.asUint8List();
            final cv.Mat mat = cv.imdecode(imageBytes, cv.IMREAD_COLOR);
            try {
              final hands = await core!.detectDirect(mat);
              mainSendPort.send({
                'id': id,
                'result': hands.map((h) => h.toMap()).toList(),
              });
            } finally {
              mat.dispose();
            }

          case 'detectMat':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'HandDetectorCore not initialized in isolate',
              });
              return;
            }
            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final Uint8List matBytes = bb.asUint8List();
            final int width = message['width'] as int;
            final int height = message['height'] as int;
            final int matTypeValue = message['matType'] as int;
            final matType = cv.MatType(matTypeValue);
            final mat = cv.Mat.fromList(height, width, matType, matBytes);
            try {
              final hands = await core!.detectDirect(mat);
              mainSendPort.send({
                'id': id,
                'result': hands.map((h) => h.toMap()).toList(),
              });
            } finally {
              mat.dispose();
            }

          case 'detectCameraFrame':
            if (core == null) {
              mainSendPort.send({
                'id': id,
                'error': 'HandDetectorCore not initialized in isolate',
              });
              return;
            }
            final Uint8List frameBytes =
                (message['bytes'] as TransferableTypedData)
                    .materialize()
                    .asUint8List();
            final frameMat = _matFromCameraFrameMessage(message, frameBytes);
            try {
              final hands = await core!.detectDirect(frameMat);
              mainSendPort.send({
                'id': id,
                'result': hands.map((h) => h.toMap()).toList(),
              });
            } finally {
              frameMat.dispose();
            }

          case 'dispose':
            await core?.dispose();
            core = null;
            // ACK the dispose so the main side can await it before force-
            // killing the isolate. See HandDetector.dispose().
            mainSendPort.send({'id': id, 'result': null});
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }

  /// Decodes a [CameraFrame] isolate message into a 3-channel BGR [cv.Mat],
  /// applying the conversion (YUV→BGR or BGRA/RGBA→BGR, with optional stride
  /// crop) and any requested rotation. Runs inside the detection isolate.
  ///
  /// Op ordering is tuned to keep the big allocations tiny: for BGRA frames we
  /// resize and rotate on the 4-channel buffer and defer `cvtColor` to the end
  /// (so it converts the post-resize ~maxDim buffer, not full-res). For YUV we
  /// must `cvtColor` first because the packed layout isn't resizable, but we
  /// then resize before rotating so the rotate runs on the small BGR buffer.
  /// Output is byte-identical to the rotate→resize→cvtColor order because
  /// `cv.rotate` 90/180/270 is a lossless permutation, `cv.resize`
  /// (`INTER_LINEAR`) interpolates each channel independently, and the
  /// BGRA→BGR conversion is a per-pixel alpha drop.
  static cv.Mat _matFromCameraFrameMessage(Map message, Uint8List bytes) {
    final int width = message['width'] as int;
    final int height = message['height'] as int;
    final int strideCols = message['strideCols'] as int;
    final conversion =
        CameraFrameConversion.values[message['conversion'] as int];
    final int? rotationIndex = message['rotation'] as int?;
    final int? maxDim = message['maxDim'] as int?;

    int? rotateFlag() {
      if (rotationIndex == null) return null;
      return switch (CameraFrameRotation.values[rotationIndex]) {
        CameraFrameRotation.cw90 => cv.ROTATE_90_CLOCKWISE,
        CameraFrameRotation.cw180 => cv.ROTATE_180,
        CameraFrameRotation.cw270 => cv.ROTATE_90_COUNTERCLOCKWISE,
      };
    }

    cv.Mat maybeResize(cv.Mat m) {
      if (maxDim == null || (m.cols <= maxDim && m.rows <= maxDim)) return m;
      final double scale = maxDim / (m.cols > m.rows ? m.cols : m.rows);
      final resized = cv.resize(
        m,
        ((m.cols * scale).toInt(), (m.rows * scale).toInt()),
        interpolation: cv.INTER_LINEAR,
      );
      m.dispose();
      return resized;
    }

    cv.Mat maybeRotate(cv.Mat m) {
      final flag = rotateFlag();
      if (flag == null) return m;
      final rotated = cv.rotate(m, flag);
      m.dispose();
      return rotated;
    }

    switch (conversion) {
      case CameraFrameConversion.bgra2bgr:
      case CameraFrameConversion.rgba2bgr:
        final bgraOrRgba =
            cv.Mat.fromList(height, strideCols, cv.MatType.CV_8UC4, bytes);
        cv.Mat current = strideCols != width
            ? bgraOrRgba.region(cv.Rect(0, 0, width, height))
            : bgraOrRgba;

        if (maxDim != null &&
            (current.cols > maxDim || current.rows > maxDim)) {
          final double scale = maxDim /
              (current.cols > current.rows ? current.cols : current.rows);
          final resized = cv.resize(
            current,
            ((current.cols * scale).toInt(), (current.rows * scale).toInt()),
            interpolation: cv.INTER_LINEAR,
          );
          if (!identical(current, bgraOrRgba)) current.dispose();
          current = resized;
        }

        final flag = rotateFlag();
        if (flag != null) {
          final rotated = cv.rotate(current, flag);
          if (!identical(current, bgraOrRgba)) current.dispose();
          current = rotated;
        }

        final cvtCode = conversion == CameraFrameConversion.bgra2bgr
            ? cv.COLOR_BGRA2BGR
            : cv.COLOR_RGBA2BGR;
        final bgr = cv.cvtColor(current, cvtCode);
        if (!identical(current, bgraOrRgba)) current.dispose();
        bgraOrRgba.dispose();
        return bgr;

      case CameraFrameConversion.yuv2bgrNv12:
      case CameraFrameConversion.yuv2bgrNv21:
      case CameraFrameConversion.yuv2bgrI420:
        final yuvMat = cv.Mat.fromList(
          height + height ~/ 2,
          width,
          cv.MatType.CV_8UC1,
          bytes,
        );
        final cvtCode = switch (conversion) {
          CameraFrameConversion.yuv2bgrNv12 => cv.COLOR_YUV2BGR_NV12,
          CameraFrameConversion.yuv2bgrNv21 => cv.COLOR_YUV2BGR_NV21,
          CameraFrameConversion.yuv2bgrI420 => cv.COLOR_YUV2BGR_I420,
          _ => cv.COLOR_YUV2BGR_NV12,
        };
        cv.Mat current = cv.cvtColor(yuvMat, cvtCode);
        yuvMat.dispose();
        current = maybeResize(current);
        current = maybeRotate(current);
        return current;
    }
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
    required int maxDetections,
    required double minLandmarkScore,
    required int interpreterPoolSize,
    required PerformanceConfig performanceConfig,
    required bool enableGestures,
    required double gestureMinConfidence,
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
          maxDetections: maxDetections,
          minLandmarkScore: minLandmarkScore,
          interpreterPoolSize: interpreterPoolSize,
          performanceModeName: performanceConfig.mode.name,
          numThreads: performanceConfig.numThreads,
          enableGestures: enableGestures,
          gestureMinConfidence: gestureMinConfidence,
        ),
        debugName: 'HandDetector',
      ),
      timeout: const Duration(seconds: 30),
      timeoutMessage: 'Hand detection isolate initialization timed out',
    );
  }
}
