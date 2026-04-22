import 'dart:async';
import 'dart:io';
import 'dart:math';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show DeviceOrientation;
import 'package:image_picker/image_picker.dart';
import 'package:file_selector/file_selector.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:camera/camera.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:sensors_plus/sensors_plus.dart';

/// Calculates the 4 corner points of a rotated rectangle.
///
/// Matches OpenCV's cv.boxPoints() behavior for drawing rotated bounding boxes.
/// Parameters:
/// - [cx], [cy]: Center coordinates of the rectangle
/// - [width], [height]: Dimensions of the rectangle
/// - [rotation]: Rotation angle in radians
///
/// Returns a list of 4 Offset points representing the corners of the rotated rectangle.
List<Offset> getRotatedRectPoints(
  double cx,
  double cy,
  double width,
  double height,
  double rotation,
) {
  final b = cos(rotation) * 0.5;
  final a = sin(rotation) * 0.5;

  return [
    Offset(cx - a * height - b * width, cy + b * height - a * width),
    Offset(cx + a * height - b * width, cy - b * height - a * width),
    Offset(cx + a * height + b * width, cy - b * height + a * width),
    Offset(cx - a * height + b * width, cy + b * height + a * width),
  ];
}

void main() {
  runApp(const HandDetectionApp());
}

class HandDetectionApp extends StatelessWidget {
  const HandDetectionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Hand Detection Demo',
      theme: ThemeData(
        colorSchemeSeed: Colors.blue,
        useMaterial3: true,
      ),
      home: const HomeScreen(),
    );
  }
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Hand Detection Demo'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.pan_tool, size: 100, color: Colors.blue[300]),
            const SizedBox(height: 48),
            Text(
              'Choose Detection Mode',
              style: Theme.of(context).textTheme.headlineMedium,
            ),
            const SizedBox(height: 48),
            _buildModeCard(
              context,
              icon: Icons.image,
              title: 'Still Image',
              description: 'Detect hands in photos from gallery or camera',
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const StillImageScreen(),
                  ),
                );
              },
            ),
            const SizedBox(height: 24),
            _buildModeCard(
              context,
              icon: Icons.videocam,
              title: 'Live Camera',
              description: 'Real-time hand detection from camera feed',
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => const CameraScreen(),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModeCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String description,
    required VoidCallback onTap,
  }) {
    return SizedBox(
      width: 400,
      child: Card(
        elevation: 4,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Row(
              children: [
                Icon(icon, size: 64, color: Colors.blue),
                const SizedBox(width: 24),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        title,
                        style: Theme.of(context).textTheme.titleLarge,
                      ),
                      const SizedBox(height: 8),
                      Text(
                        description,
                        style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                              color: Colors.grey[600],
                            ),
                      ),
                    ],
                  ),
                ),
                const Icon(Icons.arrow_forward_ios),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class StillImageScreen extends StatefulWidget {
  const StillImageScreen({super.key});

  @override
  State<StillImageScreen> createState() => _StillImageScreenState();
}

class _StillImageScreenState extends State<StillImageScreen> {
  final HandDetector _handDetector = HandDetector(
    mode: HandMode.boxesAndLandmarks,
    landmarkModel: HandLandmarkModel.full,
    detectorConf: 0.6,
    maxDetections: 10,
    minLandmarkScore: 0.5,
    performanceConfig: PerformanceConfig
        .disabled, // Disabled XNNPACK to fix initialization error
  );
  final ImagePicker _picker = ImagePicker();

  bool _isInitialized = false;
  bool _isProcessing = false;
  File? _imageFile;
  List<Hand> _results = [];
  String? _errorMessage;

  @override
  void initState() {
    super.initState();
    _initializeDetectors();
  }

  Future<void> _initializeDetectors() async {
    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    try {
      await _handDetector.initialize();
      setState(() {
        _isInitialized = true;
        _isProcessing = false;
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Failed to initialize: $e';
      });
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);
      if (pickedFile == null) return;

      setState(() {
        _imageFile = File(pickedFile.path);
        _results = [];
        _isProcessing = true;
        _errorMessage = null;
      });

      final Uint8List bytes = await _imageFile!.readAsBytes();
      final List<Hand> results = await _handDetector.detect(bytes);

      setState(() {
        _results = results;
        _isProcessing = false;
        if (results.isEmpty) _errorMessage = 'No hands detected in image';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  Future<void> _pickFileFromSystem() async {
    try {
      const XTypeGroup typeGroup = XTypeGroup(
        label: 'images',
        extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
      );
      final XFile? file = await openFile(acceptedTypeGroups: [typeGroup]);

      if (file == null) return;

      setState(() {
        _imageFile = File(file.path);
        _results = [];
        _isProcessing = true;
        _errorMessage = null;
      });

      final Uint8List bytes = await _imageFile!.readAsBytes();
      final List<Hand> results = await _handDetector.detect(bytes);

      setState(() {
        _results = results;
        _isProcessing = false;
        if (results.isEmpty) _errorMessage = 'No hands detected in image';
      });
    } catch (e) {
      setState(() {
        _isProcessing = false;
        _errorMessage = 'Error: $e';
      });
    }
  }

  void _showImageSourceDialog() {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Select Image Source'),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.folder_open),
                title: const Text('Browse Files'),
                onTap: () {
                  Navigator.pop(context);
                  _pickFileFromSystem();
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_library),
                title: const Text('Gallery'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.gallery);
                },
              ),
              ListTile(
                leading: const Icon(Icons.camera_alt),
                title: const Text('Camera'),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage(ImageSource.camera);
                },
              ),
            ],
          ),
        );
      },
    );
  }

  @override
  void dispose() {
    _handDetector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Hand Detection Demo'),
        actions: [
          if (_isInitialized && _imageFile != null)
            IconButton(
              icon: const Icon(Icons.info_outline),
              onPressed: _showHandInfo,
            ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _isInitialized && !_isProcessing
          ? FloatingActionButton.extended(
              onPressed: _showImageSourceDialog,
              icon: const Icon(Icons.add_photo_alternate),
              label: const Text('Select Image'),
            )
          : null,
    );
  }

  Widget _buildBody() {
    if (!_isInitialized && _isProcessing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing hand detector...'),
          ],
        ),
      );
    }

    if (_errorMessage != null && _imageFile == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.red),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _initializeDetectors,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (_imageFile == null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.pan_tool_outlined, size: 100, color: Colors.grey[400]),
            const SizedBox(height: 24),
            Text('Select an image to detect hands',
                style: TextStyle(fontSize: 18, color: Colors.grey[600])),
            const SizedBox(height: 16),
            ElevatedButton.icon(
              onPressed: _showImageSourceDialog,
              icon: const Icon(Icons.add_photo_alternate),
              label: const Text('Select Image'),
            ),
          ],
        ),
      );
    }

    return SingleChildScrollView(
      child: Column(
        children: [
          HandVisualizerWidget(
            imageFile: _imageFile!,
            results: _results,
          ),
          if (_isProcessing)
            const Padding(
              padding: EdgeInsets.all(16),
              child: Column(
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 8),
                  Text('Detecting hands...'),
                ],
              ),
            ),
          if (_errorMessage != null && !_isProcessing)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Card(
                color: Colors.red[50],
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline, color: Colors.red),
                      const SizedBox(width: 8),
                      Expanded(child: Text(_errorMessage!)),
                    ],
                  ),
                ),
              ),
            ),
          if (_results.isNotEmpty)
            Padding(
              padding: const EdgeInsets.all(16),
              child: Card(
                child: Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Detections: ${_results.length}',
                          style: Theme.of(context)
                              .textTheme
                              .titleLarge
                              ?.copyWith(
                                  color: Colors.green,
                                  fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }

  void _showHandInfo() {
    if (_results.isEmpty) return;
    final Hand first = _results.first;

    showModalBottomSheet(
      context: context,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.7,
        minChildSize: 0.5,
        maxChildSize: 0.95,
        expand: false,
        builder: (context, scrollController) => ListView(
          controller: scrollController,
          padding: const EdgeInsets.all(16),
          children: [
            Text('Landmark Details (first hand)',
                style: Theme.of(context).textTheme.headlineSmall),
            const SizedBox(height: 16),
            ..._buildLandmarkListFor(first),
          ],
        ),
      ),
    );
  }

  List<Widget> _buildLandmarkListFor(Hand result) {
    final List<HandLandmark> lm = result.landmarks;
    return lm.map((landmark) {
      final Point pixel =
          landmark.toPixel(result.imageWidth, result.imageHeight);
      return Card(
        margin: const EdgeInsets.only(bottom: 8),
        child: ListTile(
          leading: CircleAvatar(
            backgroundColor:
                landmark.visibility > 0.5 ? Colors.green : Colors.orange,
            child: Text(landmark.type.index.toString(),
                style: const TextStyle(fontSize: 12)),
          ),
          title: Text(_landmarkName(landmark.type),
              style: const TextStyle(fontWeight: FontWeight.w500)),
          subtitle: Text(''
              'Position: (${pixel.x}, ${pixel.y})\n'
              'Visibility: ${(landmark.visibility * 100).toStringAsFixed(0)}%'),
          isThreeLine: true,
        ),
      );
    }).toList();
  }

  String _landmarkName(HandLandmarkType type) {
    return type
        .toString()
        .split('.')
        .last
        .replaceAllMapped(
          RegExp(r'[A-Z]'),
          (match) => ' ${match.group(0)}',
        )
        .trim();
  }
}

class HandVisualizerWidget extends StatelessWidget {
  final File imageFile;
  final List<Hand> results;

  const HandVisualizerWidget(
      {super.key, required this.imageFile, required this.results});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(builder: (context, constraints) {
      return Stack(
        children: [
          Image.file(imageFile, fit: BoxFit.contain),
          Positioned.fill(
              child:
                  CustomPaint(painter: MultiOverlayPainter(results: results))),
        ],
      );
    });
  }
}

class MultiOverlayPainter extends CustomPainter {
  final List<Hand> results;

  MultiOverlayPainter({required this.results});

  @override
  void paint(Canvas canvas, Size size) {
    if (results.isEmpty) return;

    final int iw = results.first.imageWidth;
    final int ih = results.first.imageHeight;

    final double imageAspect = iw / ih;
    final double canvasAspect = size.width / size.height;
    double scaleX, scaleY;
    double offsetX = 0, offsetY = 0;

    if (canvasAspect > imageAspect) {
      scaleY = size.height / ih;
      scaleX = scaleY;
      offsetX = (size.width - iw * scaleX) / 2;
    } else {
      scaleX = size.width / iw;
      scaleY = scaleX;
      offsetY = (size.height - ih * scaleY) / 2;
    }

    for (final r in results) {
      _drawBbox(canvas, r, scaleX, scaleY, offsetX, offsetY);
      if (r.hasLandmarks) {
        _drawConnections(canvas, r, scaleX, scaleY, offsetX, offsetY);
        _drawLandmarks(canvas, r, scaleX, scaleY, offsetX, offsetY);
      }
    }
  }

  void _drawConnections(Canvas canvas, Hand result, double scaleX,
      double scaleY, double offsetX, double offsetY) {
    final Paint paint = Paint()
      ..color = Colors.green.withValues(alpha: 0.8)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    // Use the predefined skeleton connections from the package
    for (final List<HandLandmarkType> c in handLandmarkConnections) {
      final HandLandmark? start = result.getLandmark(c[0]);
      final HandLandmark? end = result.getLandmark(c[1]);
      if (start != null &&
          end != null &&
          start.visibility > 0.5 &&
          end.visibility > 0.5) {
        canvas.drawLine(
          Offset(start.x * scaleX + offsetX, start.y * scaleY + offsetY),
          Offset(end.x * scaleX + offsetX, end.y * scaleY + offsetY),
          paint,
        );
      }
    }
  }

  void _drawLandmarks(Canvas canvas, Hand result, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    for (final HandLandmark l in result.landmarks) {
      if (l.visibility > 0.5) {
        final Offset center =
            Offset(l.x * scaleX + offsetX, l.y * scaleY + offsetY);
        final Paint glow = Paint()..color = Colors.blue.withValues(alpha: 0.3);
        final Paint point = Paint()..color = Colors.red;
        final Paint centerDot = Paint()..color = Colors.white;
        canvas.drawCircle(center, 8, glow);
        canvas.drawCircle(center, 5, point);
        canvas.drawCircle(center, 2, centerDot);
      }
    }
  }

  void _drawBbox(Canvas canvas, Hand r, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    // Draw rotated rectangle (red) if rotation data exists
    if (r.rotation != null &&
        r.rotatedCenterX != null &&
        r.rotatedCenterY != null &&
        r.rotatedSize != null) {
      final Paint rotatedPaint = Paint()
        ..color = Colors.red.withValues(alpha: 0.9)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;

      final points = getRotatedRectPoints(
        r.rotatedCenterX! * scaleX + offsetX,
        r.rotatedCenterY! * scaleY + offsetY,
        r.rotatedSize! * scaleX,
        r.rotatedSize! * scaleY,
        r.rotation!,
      );

      final path = Path()..addPolygon(points, true);
      canvas.drawPath(path, rotatedPaint);
    }

    // Draw regular axis-aligned bbox (orange)
    final Paint boxPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final Paint fillPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.08)
      ..style = PaintingStyle.fill;

    final double x1 = r.boundingBox.left * scaleX + offsetX;
    final double y1 = r.boundingBox.top * scaleY + offsetY;
    final double x2 = r.boundingBox.right * scaleX + offsetX;
    final double y2 = r.boundingBox.bottom * scaleY + offsetY;
    final Rect rect = Rect.fromLTRB(x1, y1, x2, y2);
    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, boxPaint);
  }

  @override
  bool shouldRepaint(MultiOverlayPainter oldDelegate) => true;
}

class CameraScreen extends StatefulWidget {
  const CameraScreen({super.key});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  CameraController? _cameraController;
  bool _isImageStreamStarted = false;
  int _cameraGeneration = 0;
  bool _isDisposing = false;
  HandDetectorIsolate? _handDetectorIsolate;
  int _maxHands = 2;
  bool _enableGestures = true;

  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _isSwitchingCamera = false;
  List<Hand> _currentHands = [];
  String? _errorMessage;
  int _frameCount = 0;
  static const int _frameSkip =
      1; // Process every frame (matches Python default)
  Size? _imageSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;
  List<CameraDescription> _availableCameras = const [];
  String _deviceOrientation = 'Portrait Up';
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;

  // FPS / detection-time display (inline, matches face_detection_tflite)
  int _fps = 0;
  DateTime? _lastFpsUpdate;
  int _framesSinceLastUpdate = 0;
  int _detectionTimeMs = 0;

  @override
  void initState() {
    super.initState();
    _initializeHandDetector();
    _initCamera();

    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      _accelerometerSub = accelerometerEventStream().listen((event) {
        final next = event.x.abs() > event.y.abs()
            ? (event.x > 0 ? 'Landscape Left' : 'Landscape Right')
            : (event.y > 0 ? 'Portrait Up' : 'Portrait Down');
        if (next == 'Portrait Down' &&
            (_deviceOrientation == 'Landscape Left' ||
                _deviceOrientation == 'Landscape Right')) {
          return;
        }
        if (next != _deviceOrientation && mounted) {
          setState(() => _deviceOrientation = next);
        }
      });
    }
  }

  Future<void> _initializeHandDetector() async {
    try {
      _handDetectorIsolate = await HandDetectorIsolate.spawn(
        mode: HandMode.boxesAndLandmarks,
        landmarkModel: HandLandmarkModel.full,
        detectorConf: 0.6,
        maxDetections: _maxHands,
        minLandmarkScore: 0.5,
        performanceConfig: const PerformanceConfig.xnnpack(),
        enableGestures: _enableGestures,
        gestureMinConfidence: 0.5,
      );
      if (mounted) {
        setState(() {
          _isInitialized = true;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorMessage = 'Failed to initialize hand detector: $e';
        });
      }
    }
  }

  Future<void> _updateMaxHands(int newMax) async {
    if (newMax == _maxHands) return;

    setState(() {
      _isInitialized = false;
      _maxHands = newMax;
    });

    // Dispose old detector and create new one
    _handDetectorIsolate?.dispose();
    _handDetectorIsolate = null;
    await _initializeHandDetector();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          setState(() => _errorMessage = 'No cameras available');
        }
        return;
      }
      _availableCameras = cameras;

      final camera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.front,
        orElse: () => cameras.first,
      );

      await _startControllerFor(camera);
    } catch (e, st) {
      debugPrint('Camera init failed: $e');
      debugPrint('$st');
      if (mounted) {
        setState(() => _errorMessage = 'Camera init failed: $e');
      }
    }
  }

  Future<void> _startControllerFor(CameraDescription camera) async {
    final generation = ++_cameraGeneration;
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );
    var imageStreamStarted = false;

    try {
      await controller.initialize();
      if (!_isCurrentCameraGeneration(generation)) {
        await _disposeController(controller, stopStream: false);
        return;
      }

      await controller.startImageStream(_processCameraImage);
      imageStreamStarted = true;
      if (!_isCurrentCameraGeneration(generation)) {
        await _disposeController(controller, stopStream: true);
        return;
      }

      setState(() {
        _cameraController = controller;
        _isImageStreamStarted = true;
        _sensorOrientation = controller.description.sensorOrientation;
        _isFrontCamera =
            controller.description.lensDirection == CameraLensDirection.front;
      });
    } catch (_) {
      await _disposeController(controller, stopStream: imageStreamStarted);
      rethrow;
    }
  }

  bool _isCurrentCameraGeneration(int generation) {
    return mounted && !_isDisposing && generation == _cameraGeneration;
  }

  Future<void> _disposeController(
    CameraController controller, {
    required bool stopStream,
  }) async {
    if (stopStream) {
      try {
        await controller.stopImageStream();
      } catch (_) {}
    }
    try {
      await controller.dispose();
    } catch (_) {}
  }

  bool get _canSwitchCamera {
    if (kIsWeb) return false;
    if (!(Platform.isAndroid || Platform.isIOS)) return false;
    final hasFront = _availableCameras
        .any((c) => c.lensDirection == CameraLensDirection.front);
    final hasBack = _availableCameras
        .any((c) => c.lensDirection == CameraLensDirection.back);
    return hasFront && hasBack;
  }

  Future<void> _switchCamera() async {
    if (_isSwitchingCamera) return;
    if (!_canSwitchCamera) return;

    final target =
        _isFrontCamera ? CameraLensDirection.back : CameraLensDirection.front;
    final next = _availableCameras.firstWhere(
      (c) => c.lensDirection == target,
      orElse: () => _availableCameras.first,
    );

    final prev = _cameraController;
    final wasStreaming = _isImageStreamStarted;
    ++_cameraGeneration;
    setState(() {
      _isSwitchingCamera = true;
      _cameraController = null;
      _isImageStreamStarted = false;
      _currentHands = [];
      _imageSize = null;
    });
    try {
      if (prev != null) {
        await _disposeController(prev, stopStream: wasStreaming);
      }

      await _startControllerFor(next);
    } catch (e, st) {
      debugPrint('Camera switch failed: $e');
      debugPrint('$st');
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error switching camera: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isSwitchingCamera = false);
    }
  }

  void _updateFps() {
    _framesSinceLastUpdate++;
    final now = DateTime.now();
    if (_lastFpsUpdate != null) {
      final diff = now.difference(_lastFpsUpdate!).inMilliseconds;
      if (diff >= 1000 && mounted) {
        setState(() {
          _fps = (_framesSinceLastUpdate * 1000 / diff).round();
          _framesSinceLastUpdate = 0;
          _lastFpsUpdate = now;
        });
      }
    } else {
      _lastFpsUpdate = now;
    }
  }

  int get _barQuarterTurns {
    switch (_deviceOrientation) {
      case 'Landscape Left':
        return 1;
      case 'Landscape Right':
        return 3;
      default:
        return 0;
    }
  }

  DeviceOrientation _effectiveDeviceOrientation(BuildContext context) {
    final controller = _cameraController;
    if (controller != null) {
      return controller.value.deviceOrientation;
    }
    return MediaQuery.of(context).orientation == Orientation.portrait
        ? DeviceOrientation.portraitUp
        : DeviceOrientation.landscapeLeft;
  }

  int? _rotationFlagForFrame({
    required int width,
    required int height,
  }) {
    if (!mounted) return null;
    final int? sensor = _sensorOrientation;
    if (sensor == null) return null;

    // iOS: the camera plugin pre-rotates the image stream per
    // AVCaptureConnection.videoOrientation, so the historical portrait-only
    // rotation path still applies. Landscape iOS is handled in step 3 of
    // the rotation plan (pending empirical verification on device via the
    // one-shot probe logged from _processCameraImage).
    if (Platform.isIOS) {
      final DeviceOrientation orientation =
          _effectiveDeviceOrientation(context);
      final bool isPortrait = orientation == DeviceOrientation.portraitUp ||
          orientation == DeviceOrientation.portraitDown;
      if (!isPortrait) return null;
      if (height >= width) return null;
      if (sensor == 90) return cv.ROTATE_90_CLOCKWISE;
      if (sensor == 270) return cv.ROTATE_90_COUNTERCLOCKWISE;
      return null;
    }

    // Android: combined formula covering all four device orientations.
    // `sensorOrientation` is the clockwise rotation needed to display the
    // raw sensor buffer upright in the device's natural orientation;
    // `deviceRotation` is how far the device is rotated clockwise from
    // natural (portraitUp=0, landscapeLeft=90, portraitDown=180,
    // landscapeRight=270; per Flutter's DeviceOrientation enum).
    if (Platform.isAndroid) {
      final DeviceOrientation d = _effectiveDeviceOrientation(context);
      final int deviceRotation = switch (d) {
        DeviceOrientation.portraitUp => 0,
        DeviceOrientation.landscapeLeft => 90,
        DeviceOrientation.portraitDown => 180,
        DeviceOrientation.landscapeRight => 270,
      };

      final int total = _isFrontCamera
          ? (sensor + deviceRotation) % 360
          : (sensor - deviceRotation + 360) % 360;

      return switch (total) {
        90 => cv.ROTATE_90_CLOCKWISE,
        180 => cv.ROTATE_180,
        270 => cv.ROTATE_90_COUNTERCLOCKWISE,
        _ => null,
      };
    }

    // Desktop / web: camera_desktop delivers already-upright frames.
    return null;
  }

  /// Converts a CameraImage to BGR cv.Mat for OpenCV processing.
  ///
  /// Handles:
  /// - Desktop BGRA (macOS via camera_desktop): single plane, BGRA byte order
  /// - Desktop RGBA (Linux via camera_desktop): single plane, RGBA byte order
  /// - iOS NV12: 2 planes, YUV420
  /// - Android I420: 3 planes, YUV420
  Future<cv.Mat?> _convertCameraImageToMat(CameraImage image) async {
    try {
      final int width = image.width;
      final int height = image.height;

      // Desktop: camera_desktop provides single-plane 4-channel packed format
      if (image.planes.length == 1 &&
          (image.planes[0].bytesPerPixel ?? 1) >= 4) {
        final bytes = image.planes[0].bytes;
        final stride = image.planes[0].bytesPerRow;

        // Create a 4-channel Mat directly from camera bytes (handles stride)
        final matCols = stride ~/ 4;
        final bgraOrRgba =
            cv.Mat.fromList(height, matCols, cv.MatType.CV_8UC4, bytes);
        // Crop out stride padding if present
        final cropped = matCols != width
            ? bgraOrRgba.region(cv.Rect(0, 0, width, height))
            : bgraOrRgba;

        // Native SIMD-accelerated color conversion
        final colorCode =
            Platform.isMacOS ? cv.COLOR_BGRA2BGR : cv.COLOR_RGBA2BGR;
        cv.Mat mat = cv.cvtColor(cropped, colorCode);

        if (!identical(cropped, bgraOrRgba)) cropped.dispose();
        bgraOrRgba.dispose();

        final rotationFlag =
            _rotationFlagForFrame(width: width, height: height);
        if (rotationFlag != null) {
          final rotated = cv.rotate(mat, rotationFlag);
          mat.dispose();
          return rotated;
        }
        return mat;
      }

      // Mobile: YUV420. Pack Y+UV into a contiguous buffer via flutter_litert's
      // shared `packYuv420`, then hand to OpenCV for native cvtColor. The Dart
      // per-pixel loop this replaced was ~500ms/frame on Android; cvtColor
      // runs in single-digit ms.
      final p0 = image.planes[0];
      final p1 = image.planes.length > 1 ? image.planes[1] : null;
      final p2 = image.planes.length > 2 ? image.planes[2] : null;
      if (p1 == null) return null;

      final packed = packYuv420(
        width: width,
        height: height,
        y: (
          bytes: p0.bytes,
          rowStride: p0.bytesPerRow,
          pixelStride: p0.bytesPerPixel ?? 1,
        ),
        u: (
          bytes: p1.bytes,
          rowStride: p1.bytesPerRow,
          pixelStride: p1.bytesPerPixel ?? 1,
        ),
        v: p2 == null
            ? null
            : (
                bytes: p2.bytes,
                rowStride: p2.bytesPerRow,
                pixelStride: p2.bytesPerPixel ?? 1,
              ),
      );
      if (packed == null) return null;

      final int cvtCode = switch (packed.layout) {
        YuvLayout.nv12 => cv.COLOR_YUV2BGR_NV12,
        YuvLayout.nv21 => cv.COLOR_YUV2BGR_NV21,
        YuvLayout.i420 => cv.COLOR_YUV2BGR_I420,
      };
      final cv.Mat yuvMat = cv.Mat.fromList(
        packed.height + packed.height ~/ 2,
        packed.width,
        cv.MatType.CV_8UC1,
        packed.bytes,
      );
      cv.Mat mat = cv.cvtColor(yuvMat, cvtCode);
      yuvMat.dispose();

      final rotationFlag = _rotationFlagForFrame(width: width, height: height);
      if (rotationFlag != null) {
        final rotated = cv.rotate(mat, rotationFlag);
        mat.dispose();
        return rotated;
      }
      return mat;
    } catch (e) {
      return null;
    }
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (!mounted || _cameraController == null) return;
    final generation = _cameraGeneration;

    _updateFps();

    // Skip frames for performance
    _frameCount++;
    if (_frameCount % _frameSkip != 0) {
      return;
    }

    // Skip if already processing
    if (_isProcessing || !_isInitialized || _handDetectorIsolate == null) {
      return;
    }

    _isProcessing = true;

    try {
      cv.Mat? mat = await _convertCameraImageToMat(image);
      if (mat == null || _handDetectorIsolate == null) {
        _isProcessing = false;
        return;
      }

      // Downscale for performance, the palm detection model internally
      // resizes to 192×192, so full-res frames just waste IPC bandwidth.
      const int maxDim = 640;
      if (mat.cols > maxDim || mat.rows > maxDim) {
        final double scale =
            maxDim / (mat.cols > mat.rows ? mat.cols : mat.rows);
        final cv.Mat resized = cv.resize(
          mat,
          ((mat.cols * scale).toInt(), (mat.rows * scale).toInt()),
          interpolation: cv.INTER_LINEAR,
        );
        mat.dispose();
        mat = resized;
      }

      // Track detection image size for overlay coordinate mapping.
      final Size detectionSize = Size(mat.cols.toDouble(), mat.rows.toDouble());

      final stopwatch = Stopwatch()..start();

      // Run hand detection in background isolate
      final List<Hand> hands =
          await _handDetectorIsolate!.detectHandsFromMat(mat);

      stopwatch.stop();

      // Clean up
      mat.dispose();

      if (_isCurrentCameraGeneration(generation)) {
        setState(() {
          _currentHands = hands;
          _detectionTimeMs = stopwatch.elapsedMilliseconds;
          _imageSize = detectionSize;
        });
      }
    } catch (_) {
      // Silently ignore errors
    } finally {
      _isProcessing = false;
    }
  }

  @override
  void dispose() {
    _isDisposing = true;
    ++_cameraGeneration;
    _accelerometerSub?.cancel();
    final controller = _cameraController;
    final wasStreaming = _isImageStreamStarted;
    _cameraController = null;
    _isImageStreamStarted = false;
    if (controller != null) {
      unawaited(_disposeController(controller, stopStream: wasStreaming));
    }
    _handDetectorIsolate?.dispose();
    super.dispose();
  }

  Widget _buildCameraTopBar() {
    final canPop = Navigator.of(context).canPop();
    final isMobile = !kIsWeb && (Platform.isAndroid || Platform.isIOS);

    final fpsText = SizedBox(
      width: 70,
      child: Text(
        'FPS: $_fps',
        style: const TextStyle(color: Colors.white, fontSize: 14),
        textAlign: isMobile ? TextAlign.left : TextAlign.right,
      ),
    );
    const separator = Text(
      ' | ',
      style: TextStyle(color: Colors.white, fontSize: 14),
    );
    final msText = SizedBox(
      width: 70,
      child: Text(
        '${_detectionTimeMs}ms',
        style: const TextStyle(color: Colors.white, fontSize: 14),
      ),
    );

    return Material(
      color: Colors.black.withAlpha(179),
      elevation: 4,
      child: SizedBox(
        height: kToolbarHeight,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 4),
          child: Row(
            children: [
              if (canPop)
                IconButton(
                  tooltip: 'Back',
                  color: Colors.white,
                  icon: const Icon(Icons.arrow_back),
                  onPressed: () => Navigator.of(context).maybePop(),
                ),
              if (isMobile) ...[
                const SizedBox(width: 8),
                fpsText,
                separator,
                msText,
                const Spacer(),
              ] else
                const Expanded(
                  child: Padding(
                    padding: EdgeInsets.symmetric(horizontal: 8),
                    child: Text(
                      'Live Hand Detection',
                      style: TextStyle(color: Colors.white, fontSize: 18),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ),
              if (_canSwitchCamera)
                IconButton(
                  tooltip: _isFrontCamera
                      ? 'Switch to back camera'
                      : 'Switch to front camera',
                  color: Colors.white,
                  icon: Icon(Platform.isIOS
                      ? Icons.flip_camera_ios
                      : Icons.flip_camera_android),
                  onPressed: _isSwitchingCamera ? null : _switchCamera,
                ),
              PopupMenuButton<void>(
                tooltip: 'Settings',
                icon: const Icon(Icons.settings, color: Colors.white),
                color: Colors.blueGrey[900],
                padding: EdgeInsets.zero,
                itemBuilder: (context) => [
                  PopupMenuItem<void>(
                    enabled: false,
                    padding: EdgeInsets.zero,
                    child: StatefulBuilder(
                      builder: (context, setMenuState) {
                        return _buildSettingsMenuContent(setMenuState);
                      },
                    ),
                  ),
                ],
              ),
              if (!isMobile) ...[
                const SizedBox(width: 8),
                fpsText,
                separator,
                msText,
              ],
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSettingsMenuContent(StateSetter setMenuState) {
    void update(VoidCallback fn) {
      setState(fn);
      setMenuState(() {});
    }

    const sectionLabelStyle = TextStyle(
      color: Colors.white60,
      fontSize: 10,
      fontWeight: FontWeight.w600,
      letterSpacing: 1.2,
    );

    return SizedBox(
      width: 260,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('MAX HANDS', style: sectionLabelStyle),
            const SizedBox(height: 4),
            Row(
              children: [
                Expanded(
                  child: Slider(
                    value: _maxHands.toDouble(),
                    min: 1,
                    max: 10,
                    divisions: 9,
                    activeColor: Colors.blue,
                    inactiveColor: Colors.white24,
                    label: '$_maxHands',
                    onChanged: (value) =>
                        update(() => _maxHands = value.toInt()),
                    onChangeEnd: (value) => _updateMaxHands(value.toInt()),
                  ),
                ),
                SizedBox(
                  width: 28,
                  child: Text(
                    '$_maxHands',
                    style: const TextStyle(color: Colors.white70, fontSize: 14),
                    textAlign: TextAlign.right,
                  ),
                ),
              ],
            ),
            const Divider(color: Colors.white24, height: 24),
            const Text('GESTURES', style: sectionLabelStyle),
            const SizedBox(height: 4),
            Row(
              children: [
                const Expanded(
                  child: Text(
                    'Detect gestures',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                ),
                Switch(
                  value: _enableGestures,
                  activeTrackColor: Colors.blue,
                  onChanged: (value) {
                    update(() => _enableGestures = value);
                    _updateMaxHands(_maxHands);
                  },
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _positionedTopBar(int turns) {
    final bar = _buildCameraTopBar();
    final padding = MediaQuery.of(context).padding;
    if (turns == 0) {
      return Positioned(
        top: padding.top,
        left: padding.left,
        right: padding.right,
        child: bar,
      );
    }
    return Positioned(
      top: padding.top,
      bottom: padding.bottom,
      left: turns == 3 ? padding.left : null,
      right: turns == 1 ? padding.right : null,
      width: kToolbarHeight,
      child: RotatedBox(quarterTurns: turns, child: bar),
    );
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized || _cameraController == null) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Live Hand Detection'),
        ),
        body: _buildBody(),
      );
    }

    final int turns = _barQuarterTurns;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          _buildBody(),
          _positionedTopBar(turns),
        ],
      ),
    );
  }

  Widget _buildBody() {
    if (_errorMessage != null && !_isInitialized) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              textAlign: TextAlign.center,
              style: const TextStyle(color: Colors.red),
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: () {
                setState(() {
                  _errorMessage = null;
                });
                _initializeHandDetector();
              },
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }

    if (!_isInitialized) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing hand detector...'),
          ],
        ),
      );
    }

    final controller = _cameraController;
    if (controller == null || !controller.value.isInitialized) {
      return const Center(child: CircularProgressIndicator());
    }

    final cameraAspectRatio = controller.value.aspectRatio;
    final effectiveOrientation = _effectiveDeviceOrientation(context);
    final bool isPortrait =
        effectiveOrientation == DeviceOrientation.portraitUp ||
            effectiveOrientation == DeviceOrientation.portraitDown;
    final double displayAspectRatio =
        isPortrait ? 1.0 / cameraAspectRatio : cameraAspectRatio;

    return Stack(
      fit: StackFit.expand,
      children: [
        // Camera preview + hand overlay inside a correctly-sized AspectRatio box
        Center(
          child: AspectRatio(
            aspectRatio: displayAspectRatio,
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(controller),
                if (_currentHands.isNotEmpty && _imageSize != null)
                  CustomPaint(
                    painter: CameraHandOverlayPainter(
                      hands: _currentHands,
                      imageSize: _imageSize!,
                      mirrorHorizontally: Platform.isAndroid && _isFrontCamera,
                    ),
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class CameraHandOverlayPainter extends CustomPainter {
  final List<Hand> hands;
  final Size imageSize;
  final bool mirrorHorizontally;

  CameraHandOverlayPainter({
    required this.hands,
    required this.imageSize,
    required this.mirrorHorizontally,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (hands.isEmpty) return;

    // Use the post-rotation image size for correct coordinate mapping.
    // Matches CameraPreview's cover behavior to avoid stretched/squashed overlays.
    final double sourceWidth = imageSize.width;
    final double sourceHeight = imageSize.height;

    final double sourceAspectRatio = sourceWidth / sourceHeight;
    final double viewportAspectRatio = size.width / size.height;

    double scaleX, scaleY;
    double offsetX = 0, offsetY = 0;

    if (sourceAspectRatio > viewportAspectRatio) {
      // Source is wider: fit height and crop left/right.
      scaleY = size.height / sourceHeight;
      scaleX = scaleY;
      offsetX = (size.width - sourceWidth * scaleX) / 2;
    } else {
      // Source is taller: fit width and crop top/bottom.
      scaleX = size.width / sourceWidth;
      scaleY = scaleX;
      offsetY = (size.height - sourceHeight * scaleY) / 2;
    }

    double tx(double x) => mirrorHorizontally
        ? (sourceWidth - x) * scaleX + offsetX
        : x * scaleX + offsetX;

    for (final hand in hands) {
      _drawBbox(canvas, hand, tx, scaleX, scaleY, offsetX, offsetY);
      if (hand.hasLandmarks) {
        _drawConnections(canvas, hand, tx, scaleX, scaleY, offsetX, offsetY);
        _drawLandmarks(canvas, hand, tx, scaleX, scaleY, offsetX, offsetY);
      }
      if (hand.hasGesture) {
        _drawGesture(canvas, hand, tx, scaleX, scaleY, offsetX, offsetY);
      }
    }
  }

  void _drawConnections(Canvas canvas, Hand hand, double Function(double) tx,
      double scaleX, double scaleY, double offsetX, double offsetY) {
    final Paint paint = Paint()
      ..color = Colors.green.withValues(alpha: 0.8)
      ..strokeWidth = 3
      ..strokeCap = StrokeCap.round;

    // Use the predefined skeleton connections from the package
    for (final List<HandLandmarkType> c in handLandmarkConnections) {
      final HandLandmark? start = hand.getLandmark(c[0]);
      final HandLandmark? end = hand.getLandmark(c[1]);
      if (start != null &&
          end != null &&
          start.visibility > 0.5 &&
          end.visibility > 0.5) {
        canvas.drawLine(
          Offset(tx(start.x), start.y * scaleY + offsetY),
          Offset(tx(end.x), end.y * scaleY + offsetY),
          paint,
        );
      }
    }
  }

  void _drawLandmarks(Canvas canvas, Hand hand, double Function(double) tx,
      double scaleX, double scaleY, double offsetX, double offsetY) {
    for (final HandLandmark l in hand.landmarks) {
      if (l.visibility > 0.5) {
        final Offset center = Offset(tx(l.x), l.y * scaleY + offsetY);
        final Paint glow = Paint()..color = Colors.blue.withValues(alpha: 0.3);
        final Paint point = Paint()..color = Colors.red;
        final Paint centerDot = Paint()..color = Colors.white;
        canvas.drawCircle(center, 8, glow);
        canvas.drawCircle(center, 5, point);
        canvas.drawCircle(center, 2, centerDot);
      }
    }
  }

  void _drawBbox(Canvas canvas, Hand hand, double Function(double) tx,
      double scaleX, double scaleY, double offsetX, double offsetY) {
    // Draw rotated rectangle (red) if rotation data exists
    if (hand.rotation != null &&
        hand.rotatedCenterX != null &&
        hand.rotatedCenterY != null &&
        hand.rotatedSize != null) {
      final Paint rotatedPaint = Paint()
        ..color = Colors.red.withValues(alpha: 0.9)
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2;

      final points = getRotatedRectPoints(
        tx(hand.rotatedCenterX!),
        hand.rotatedCenterY! * scaleY + offsetY,
        hand.rotatedSize! * scaleX,
        hand.rotatedSize! * scaleY,
        hand.rotation!,
      );

      final path = Path()..addPolygon(points, true);
      canvas.drawPath(path, rotatedPaint);
    }

    // Draw regular axis-aligned bbox (orange)
    final Paint boxPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.9)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;

    final Paint fillPaint = Paint()
      ..color = Colors.orangeAccent.withValues(alpha: 0.08)
      ..style = PaintingStyle.fill;

    final double x1 = tx(hand.boundingBox.left);
    final double y1 = hand.boundingBox.top * scaleY + offsetY;
    final double x2 = tx(hand.boundingBox.right);
    final double y2 = hand.boundingBox.bottom * scaleY + offsetY;
    final Rect rect = Rect.fromLTRB(min(x1, x2), y1, max(x1, x2), y2);
    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, boxPaint);
  }

  void _drawGesture(Canvas canvas, Hand hand, double Function(double) tx,
      double scaleX, double scaleY, double offsetX, double offsetY) {
    final gesture = hand.gesture;
    if (gesture == null || gesture.type == GestureType.unknown) return;

    // Only show gestures with confidence > 0.6
    if (gesture.confidence < 0.6) return;

    final emoji = _gestureToEmoji(gesture.type);
    if (emoji.isEmpty) return;

    // Position the emoji above the hand bounding box
    final double x = tx((hand.boundingBox.left + hand.boundingBox.right) / 2);
    final double y = hand.boundingBox.top * scaleY + offsetY - 20;

    // Draw background circle
    final Paint bgPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.9)
      ..style = PaintingStyle.fill;
    canvas.drawCircle(Offset(x, y), 28, bgPaint);

    // Draw border
    final Paint borderPaint = Paint()
      ..color = Colors.blue.withValues(alpha: 0.8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;
    canvas.drawCircle(Offset(x, y), 28, borderPaint);

    // Draw emoji text
    final textPainter = TextPainter(
      text: TextSpan(
        text: emoji,
        style: const TextStyle(fontSize: 32),
      ),
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(x - textPainter.width / 2, y - textPainter.height / 2),
    );

    // Draw confidence label below
    final confPainter = TextPainter(
      text: TextSpan(
        text: '${(gesture.confidence * 100).toInt()}%',
        style: TextStyle(
          fontSize: 12,
          color: Colors.white,
          fontWeight: FontWeight.bold,
          shadows: [
            Shadow(
              offset: const Offset(1, 1),
              blurRadius: 2,
              color: Colors.black.withValues(alpha: 0.8),
            ),
          ],
        ),
      ),
      textDirection: TextDirection.ltr,
    );
    confPainter.layout();
    confPainter.paint(
      canvas,
      Offset(x - confPainter.width / 2, y + 32),
    );
  }

  String _gestureToEmoji(GestureType gesture) {
    switch (gesture) {
      case GestureType.thumbUp:
        return '\u{1F44D}'; // 👍
      case GestureType.thumbDown:
        return '\u{1F44E}'; // 👎
      case GestureType.victory:
        return '\u{270C}'; // ✌️
      case GestureType.openPalm:
        return '\u{1F590}'; // 🖐️
      case GestureType.closedFist:
        return '\u{270A}'; // ✊
      case GestureType.pointingUp:
        return '\u{261D}'; // ☝️
      case GestureType.iLoveYou:
        return '\u{1F91F}'; // 🤟
      case GestureType.unknown:
        return '';
    }
  }

  @override
  bool shouldRepaint(CameraHandOverlayPainter oldDelegate) => true;
}
