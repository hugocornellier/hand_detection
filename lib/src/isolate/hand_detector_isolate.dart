import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import '../hand_detector.dart';
import '../types.dart';

/// Startup payload transferred to the background isolate via [Isolate.spawn].
///
/// Carries model bytes (as [TransferableTypedData] for zero-copy transfer)
/// and all configuration needed to reconstruct a [HandDetector] inside the isolate.
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

/// A wrapper that runs the entire hand detection pipeline in a background isolate.
///
/// This class spawns a dedicated isolate containing a full [HandDetector] instance,
/// keeping all TensorFlow Lite inference off the main UI thread. This prevents
/// frame drops during live camera processing.
///
/// Image data is transferred to the worker using zero-copy
/// [TransferableTypedData], minimizing memory overhead.
///
/// ## Usage
///
/// ```dart
/// final detector = await HandDetectorIsolate.spawn(
///   mode: HandMode.boxesAndLandmarks,
///   enableGestures: true,
///   performanceConfig: const PerformanceConfig.xnnpack(),
/// );
///
/// // From encoded image bytes (JPEG, PNG, etc.)
/// final hands = await detector.detectHands(imageBytes);
///
/// // From a cv.Mat (e.g., from camera frame conversion)
/// final hands = await detector.detectHandsFromMat(mat);
///
/// await detector.dispose();
/// ```
///
/// ## Memory Considerations
///
/// The background isolate holds all TFLite models (~8MB for full pipeline).
/// Call [dispose] when finished to release these resources.
class HandDetectorIsolate extends IsolateWorkerBase {
  HandDetectorIsolate._();

  /// Spawns a new isolate with an initialized [HandDetector].
  ///
  /// The isolate loads all TFLite models during spawn, so this operation
  /// may take 100-500ms depending on the device.
  ///
  /// Parameters:
  /// - [mode]: Detection mode (boxes only or boxes + landmarks)
  /// - [landmarkModel]: Hand landmark model variant
  /// - [detectorConf]: Palm detection confidence threshold (0.0-1.0)
  /// - [maxDetections]: Maximum number of hands to detect
  /// - [minLandmarkScore]: Minimum landmark confidence score (0.0-1.0)
  /// - [interpreterPoolSize]: Number of landmark model interpreter instances
  /// - [performanceConfig]: Hardware acceleration settings
  /// - [enableGestures]: Whether to run gesture recognition
  /// - [gestureMinConfidence]: Minimum confidence for gesture recognition
  ///
  /// Example:
  /// ```dart
  /// final detector = await HandDetectorIsolate.spawn(
  ///   performanceConfig: const PerformanceConfig.xnnpack(),
  ///   enableGestures: true,
  /// );
  /// ```
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
    final instance = HandDetectorIsolate._();
    await instance._initialize(
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
    return instance;
  }

  /// Loads model assets and spawns the background isolate with an initialized [HandDetector].
  Future<void> _initialize({
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
    if (isReady) {
      throw StateError('HandDetectorIsolate already initialized');
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

    TransferableTypedData? gestureEmbedderData;
    TransferableTypedData? gestureClassifierData;
    if (enableGestures && results.length > 2) {
      final embedderBytes = results[2].buffer.asUint8List();
      final classifierBytes = results[3].buffer.asUint8List();
      gestureEmbedderData = TransferableTypedData.fromList([embedderBytes]);
      gestureClassifierData = TransferableTypedData.fromList([classifierBytes]);
    }

    await initWorker(
      (sendPort) => Isolate.spawn(
        _isolateEntry,
        _IsolateStartupData(
          sendPort: sendPort,
          palmDetectionBytes: TransferableTypedData.fromList([palmBytes]),
          handLandmarkBytes: TransferableTypedData.fromList([landmarkBytes]),
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
        debugName: 'HandDetectorIsolate',
      ),
      timeout: const Duration(seconds: 30),
      timeoutMessage: 'Hand detection isolate initialization timed out',
    );
  }

  /// Detects hands in the given encoded image in the background isolate.
  ///
  /// All processing (image decoding, palm detection, landmark extraction,
  /// and gesture recognition) runs in the background isolate.
  ///
  /// Parameters:
  /// - [imageBytes]: Encoded image data (JPEG, PNG, etc.) as a [List<int>] or [Uint8List]
  ///
  /// Returns a list of [Hand] objects, one per detected hand.
  ///
  /// Example:
  /// ```dart
  /// final hands = await detector.detectHands(imageBytes);
  /// ```
  Future<List<Hand>> detectHands(List<int> imageBytes) async {
    final Uint8List bytes =
        imageBytes is Uint8List ? imageBytes : Uint8List.fromList(imageBytes);
    final List<dynamic> result = await sendRequest<List<dynamic>>(
      'detect',
      {
        'bytes': TransferableTypedData.fromList([bytes]),
      },
    );

    return _deserializeHands(result);
  }

  /// Detects hands in a pre-decoded [cv.Mat] image in the background isolate.
  ///
  /// The raw pixel data is extracted and transferred using zero-copy
  /// [TransferableTypedData]. The original Mat is NOT disposed by this method.
  ///
  /// Example:
  /// ```dart
  /// final mat = cv.Mat.fromList(height, width, cv.MatType.CV_8UC3, bgrBytes);
  /// final hands = await detector.detectHandsFromMat(mat);
  /// mat.dispose();
  /// ```
  Future<List<Hand>> detectHandsFromMat(cv.Mat image) {
    final int rows = image.rows;
    final int cols = image.cols;
    final int type = image.type.value;
    final Uint8List data = image.data;

    return detectHandsFromMatBytes(
      data,
      width: cols,
      height: rows,
      matType: type,
    );
  }

  /// Detects hands from raw pixel bytes in the background isolate.
  ///
  /// This is a lower-level API for when you already have raw pixel data
  /// (e.g., BGR bytes from a camera frame) without an existing cv.Mat.
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
  }) async {
    final List<dynamic> result = await sendRequest<List<dynamic>>(
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

  List<Hand> _deserializeHands(List<dynamic> result) => result
      .map((map) => Hand.fromMap(Map<String, dynamic>.from(map as Map)))
      .toList();

  @override
  String get workerDisposeOp => 'dispose';

  /// Disposes the background isolate and releases all resources.
  ///
  /// After calling dispose, the instance cannot be reused. Create a new
  /// instance with [spawn] if needed.
  @override
  Future<void> dispose() => super.dispose();

  /// Isolate entry point: initializes the [HandDetector] and listens for detection requests.
  ///
  /// Sends its [SendPort] back to the main isolate on success, or an error map on failure.
  @pragma('vm:entry-point')
  static void _isolateEntry(_IsolateStartupData data) async {
    final SendPort mainSendPort = data.sendPort;
    final ReceivePort workerReceivePort = ReceivePort();

    HandDetector? detector;

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
      final landmarkModel = HandLandmarkModel.values.firstWhere(
        (m) => m.name == data.landmarkModelName,
      );
      final performanceMode = PerformanceMode.values.firstWhere(
        (m) => m.name == data.performanceModeName,
      );

      detector = HandDetector(
        mode: mode,
        landmarkModel: landmarkModel,
        detectorConf: data.detectorConf,
        maxDetections: data.maxDetections,
        minLandmarkScore: data.minLandmarkScore,
        interpreterPoolSize: data.interpreterPoolSize,
        performanceConfig: PerformanceConfig(
          mode: performanceMode,
          numThreads: data.numThreads,
        ),
        enableGestures: data.enableGestures,
        gestureMinConfidence: data.gestureMinConfidence,
      );

      await detector.initializeFromBuffers(
        palmDetectionBytes: palmBytes,
        handLandmarkBytes: landmarkBytes,
        gestureEmbedderBytes: embedderBytes,
        gestureClassifierBytes: classifierBytes,
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
            if (detector == null || !detector!.isInitialized) {
              mainSendPort.send({
                'id': id,
                'error': 'HandDetector not initialized in isolate',
              });
              return;
            }

            final ByteBuffer bb =
                (message['bytes'] as TransferableTypedData).materialize();
            final Uint8List imageBytes = bb.asUint8List();

            final hands = await detector!.detect(imageBytes);
            final serialized = hands.map((h) => h.toMap()).toList();

            mainSendPort.send({'id': id, 'result': serialized});

          case 'detectMat':
            if (detector == null || !detector!.isInitialized) {
              mainSendPort.send({
                'id': id,
                'error': 'HandDetector not initialized in isolate',
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
              final hands = await detector!.detectOnMat(mat);
              final serialized = hands.map((h) => h.toMap()).toList();
              mainSendPort.send({'id': id, 'result': serialized});
            } finally {
              mat.dispose();
            }

          case 'dispose':
            await detector?.dispose();
            detector = null;
            workerReceivePort.close();
        }
      } catch (e, st) {
        mainSendPort.send({'id': id, 'error': '$e\n$st'});
      }
    });
  }
}
