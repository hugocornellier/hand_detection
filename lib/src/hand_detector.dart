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
class _IsolateStartupData {
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

  _IsolateStartupData({
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
  static const String _packageVersion = '2.2.0';
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

  /// Detection mode controlling pipeline behavior.
  final HandMode mode;

  /// Hand landmark model variant to use for landmark extraction.
  final HandLandmarkModel landmarkModel;

  /// Confidence threshold for palm detection (0.0 to 1.0).
  final double detectorConf;

  /// Maximum number of hands to detect per image.
  final int maxDetections;

  /// Minimum confidence score for landmark predictions (0.0 to 1.0).
  final double minLandmarkScore;

  /// Number of TensorFlow Lite interpreter instances in the landmark model pool.
  final int interpreterPoolSize;

  /// Performance configuration for TensorFlow Lite inference.
  ///
  /// By default, auto mode selects the optimal delegate per platform:
  /// - iOS: Metal GPU delegate
  /// - Android/macOS/Linux/Windows: XNNPACK (2-5x SIMD acceleration)
  final PerformanceConfig performanceConfig;

  /// Whether to run gesture recognition on detected hands.
  /// When enabled, each detected hand will include a [GestureResult].
  final bool enableGestures;

  /// Minimum confidence threshold for gesture recognition (0.0 to 1.0).
  /// Gestures with confidence below this threshold will be reported as [GestureType.unknown].
  final double gestureMinConfidence;

  _HandDetectorWorker? _worker;

  /// Creates a hand detector with the specified configuration.
  ///
  /// The detector is not ready for use until you call [initialize].
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
  HandDetector({
    this.mode = HandMode.boxesAndLandmarks,
    this.landmarkModel = HandLandmarkModel.full,
    this.detectorConf = 0.45,
    this.maxDetections = 10,
    this.minLandmarkScore = 0.5,
    this.interpreterPoolSize = 1,
    this.performanceConfig = const PerformanceConfig(),
    this.enableGestures = false,
    this.gestureMinConfidence = 0.5,
  });

  /// Creates and initializes a hand detector in one step.
  ///
  /// Convenience factory that combines [HandDetector.new] and [initialize].
  /// Accepts the same parameters as the constructor.
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
  Future<void> initialize() async {
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

    await initializeFromBuffers(
      palmDetectionBytes: palmBytes,
      handLandmarkBytes: landmarkBytes,
      gestureEmbedderBytes: gestureEmbedderBytes,
      gestureClassifierBytes: gestureClassifierBytes,
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
  Future<void> initializeFromBuffers({
    required Uint8List palmDetectionBytes,
    required Uint8List handLandmarkBytes,
    Uint8List? gestureEmbedderBytes,
    Uint8List? gestureClassifierBytes,
  }) async {
    if (isReady) {
      throw StateError('HandDetector already initialized');
    }

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
        interpreterPoolSize: interpreterPoolSize,
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
  /// Returns an empty list if image decoding fails or no hands are detected.
  ///
  /// Throws [StateError] if called before [initialize].
  Future<List<Hand>> detect(List<int> imageBytes) async {
    if (!isReady) {
      throw StateError(
          'HandDetector not initialized. Call initialize() first.');
    }
    try {
      final Uint8List bytes =
          imageBytes is Uint8List ? imageBytes : Uint8List.fromList(imageBytes);
      final List<dynamic> result = await _worker!.sendRequest<List<dynamic>>(
        'detect',
        {
          'bytes': TransferableTypedData.fromList([bytes]),
        },
      );
      return _deserializeHands(result);
    } catch (e) {
      // Intentionally broad: imdecode can throw on malformed bytes; treat any
      // decode/pipeline failure as "no hands detected" for production robustness.
      return <Hand>[];
    }
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
  /// disposed by this method — the caller is responsible for disposal.
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
  /// This avoids the overhead of building a Mat on the calling thread —
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
    await _worker?.dispose();
    _worker = null;
  }

  List<Hand> _deserializeHands(List<dynamic> result) => result
      .map((map) => Hand.fromMap(Map<String, dynamic>.from(map as Map)))
      .toList();

  /// Isolate entry point: initializes [HandDetectorCore] and listens for detection requests.
  @pragma('vm:entry-point')
  static void _isolateEntry(_IsolateStartupData data) async {
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
            if (mat.isEmpty) {
              mat.dispose();
              mainSendPort.send({'id': id, 'result': <dynamic>[]});
              return;
            }
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

          case 'dispose':
            await core?.dispose();
            core = null;
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
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
        HandDetector._isolateEntry,
        _IsolateStartupData(
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
