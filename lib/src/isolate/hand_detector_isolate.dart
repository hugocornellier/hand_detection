import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

import '../hand_detector.dart';
import '../types.dart';

/// Data passed to the detection isolate during startup.
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
class HandDetectorIsolate {
  HandDetectorIsolate._();

  Isolate? _isolate;
  SendPort? _sendPort;
  final ReceivePort _receivePort = ReceivePort();
  final Map<int, Completer<dynamic>> _pending = {};
  int _nextId = 0;

  bool _initialized = false;

  /// Returns true if the isolate is initialized and ready for detection.
  bool get isReady => _initialized;

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
    PerformanceConfig performanceConfig = PerformanceConfig.disabled,
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
    if (_initialized) {
      throw StateError('HandDetectorIsolate already initialized');
    }

    try {
      // Load all model assets from Flutter bundles in parallel
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
        gestureClassifierData =
            TransferableTypedData.fromList([classifierBytes]);
      }

      _isolate = await Isolate.spawn(
        _isolateEntry,
        _IsolateStartupData(
          sendPort: _receivePort.sendPort,
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
      );

      _sendPort = await _setupIsolateListener(
        receivePort: _receivePort,
        responseHandler: _handleResponse,
        timeout: const Duration(seconds: 30),
        timeoutMsg: 'Hand detection isolate initialization timed out',
      );

      _initialized = true;
    } catch (e) {
      _isolate?.kill(priority: Isolate.immediate);
      _receivePort.close();
      _initialized = false;
      rethrow;
    }
  }

  /// Sets up init handshake and message routing for the isolate.
  static Future<SendPort> _setupIsolateListener({
    required ReceivePort receivePort,
    required void Function(dynamic) responseHandler,
    required Duration timeout,
    required String timeoutMsg,
  }) async {
    final Completer<SendPort> initCompleter = Completer<SendPort>();
    late final StreamSubscription<dynamic> subscription;

    subscription = receivePort.listen((message) {
      if (!initCompleter.isCompleted) {
        if (message is SendPort) {
          initCompleter.complete(message);
        } else if (message is Map && message['error'] != null) {
          initCompleter.completeError(StateError(message['error'] as String));
        } else {
          initCompleter.completeError(
            StateError('Expected SendPort, got ${message.runtimeType}'),
          );
        }
        return;
      }
      responseHandler(message);
    });

    return initCompleter.future.timeout(
      timeout,
      onTimeout: () {
        subscription.cancel();
        throw TimeoutException(timeoutMsg);
      },
    );
  }

  void _handleResponse(dynamic message) {
    if (message is! Map) return;

    final int? id = message['id'] as int?;
    if (id == null) return;

    final Completer<dynamic>? completer = _pending.remove(id);
    if (completer == null) return;

    if (message['error'] != null) {
      completer.completeError(StateError(message['error'] as String));
    } else {
      completer.complete(message['result']);
    }
  }

  Future<T> _sendRequest<T>(
    String operation,
    Map<String, dynamic> params,
  ) async {
    if (!_initialized) {
      throw StateError(
        'HandDetectorIsolate not initialized. Use HandDetectorIsolate.spawn().',
      );
    }
    if (_sendPort == null) {
      throw StateError('Isolate SendPort not available.');
    }

    final int id = _nextId++;
    final Completer<T> completer = Completer<T>();
    _pending[id] = completer;

    try {
      _sendPort!.send({'id': id, 'op': operation, ...params});
      return await completer.future;
    } catch (e) {
      _pending.remove(id);
      rethrow;
    }
  }

  /// Detects hands in the given encoded image in the background isolate.
  ///
  /// All processing (image decoding, palm detection, landmark extraction,
  /// and gesture recognition) runs in the background isolate.
  ///
  /// Parameters:
  /// - [imageBytes]: Encoded image data (JPEG, PNG, etc.)
  ///
  /// Returns a list of [Hand] objects, one per detected hand.
  ///
  /// Example:
  /// ```dart
  /// final hands = await detector.detectHands(imageBytes);
  /// ```
  Future<List<Hand>> detectHands(Uint8List imageBytes) async {
    final List<dynamic> result = await _sendRequest<List<dynamic>>(
      'detect',
      {
        'bytes': TransferableTypedData.fromList([imageBytes]),
      },
    );

    return result
        .map((map) => Hand.fromMap(Map<String, dynamic>.from(map as Map)))
        .toList();
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
    final List<dynamic> result = await _sendRequest<List<dynamic>>(
      'detectMat',
      {
        'bytes': TransferableTypedData.fromList([bytes]),
        'width': width,
        'height': height,
        'matType': matType,
      },
    );

    return result
        .map((map) => Hand.fromMap(Map<String, dynamic>.from(map as Map)))
        .toList();
  }

  /// Disposes the background isolate and releases all resources.
  ///
  /// After calling dispose, the instance cannot be reused. Create a new
  /// instance with [spawn] if needed.
  Future<void> dispose() async {
    for (final completer in _pending.values) {
      if (!completer.isCompleted) {
        completer.completeError(StateError('HandDetectorIsolate disposed'));
      }
    }
    _pending.clear();

    if (_sendPort != null) {
      try {
        _sendPort!.send({'id': -1, 'op': 'dispose'});
      } catch (_) {}
    }

    _isolate?.kill(priority: Isolate.immediate);
    _receivePort.close();

    _isolate = null;
    _sendPort = null;
    _initialized = false;
  }

  /// Isolate entry point - handles hand detection.
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
