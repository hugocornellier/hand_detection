import 'dart:io';
import 'dart:typed_data';
import 'dart:math';
import 'dart:collection';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_selector/file_selector.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:camera_macos/camera_macos_controller.dart';
import 'package:camera_macos/camera_macos_view.dart';
import 'package:camera_macos/camera_macos_arguments.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

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

/// FPS calculator class ported from Python's CvFpsCalc
class FpsCalculator {
  final int bufferLen;
  final Queue<double> _diffTimes;
  DateTime _startTime;

  FpsCalculator({this.bufferLen = 10})
      : _diffTimes = Queue<double>(),
        _startTime = DateTime.now();

  double get() {
    final currentTime = DateTime.now();
    final differentTime = currentTime.difference(_startTime).inMicroseconds /
        1000.0; // milliseconds
    _startTime = currentTime;

    _diffTimes.add(differentTime);
    if (_diffTimes.length > bufferLen) {
      _diffTimes.removeFirst();
    }

    if (_diffTimes.isEmpty) return 0.0;

    final avgDiffTime = _diffTimes.reduce((a, b) => a + b) / _diffTimes.length;
    final fps = 1000.0 / avgDiffTime;
    return double.parse(fps.toStringAsFixed(2));
  }
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
  CameraMacOSController? _cameraController;
  late HandDetector _handDetector;
  int _maxHands = 2;

  bool _isInitialized = false;
  bool _isProcessing = false;
  List<Hand> _currentHands = [];
  String? _errorMessage;
  int _frameCount = 0;
  static const int _frameSkip =
      1; // Process every frame (matches Python default)
  Size? _cameraSize;

  // FPS calculation
  final FpsCalculator _fpsCalculator = FpsCalculator(bufferLen: 10);
  double _currentFps = 0.0;

  @override
  void initState() {
    super.initState();
    _createHandDetector();
    _initializeHandDetector();
  }

  void _createHandDetector() {
    _handDetector = HandDetector(
      mode: HandMode.boxesAndLandmarks,
      landmarkModel: HandLandmarkModel.full,
      detectorConf: 0.6,
      maxDetections: _maxHands,
      minLandmarkScore: 0.5,
      performanceConfig: const PerformanceConfig.xnnpack(),
      enableGestures: true,
      gestureMinConfidence: 0.5,
    );
  }

  Future<void> _initializeHandDetector() async {
    try {
      // Initialize hand detector
      await _handDetector.initialize();
      setState(() {
        _isInitialized = true;
      });
    } catch (e) {
      setState(() {
        _errorMessage = 'Failed to initialize hand detector: $e';
      });
    }
  }

  Future<void> _updateMaxHands(int newMax) async {
    if (newMax == _maxHands) return;

    setState(() {
      _isInitialized = false;
      _maxHands = newMax;
    });

    // Dispose old detector and create new one
    _handDetector.dispose();
    _createHandDetector();
    await _initializeHandDetector();
  }

  void _onCameraInitialized(CameraMacOSController controller) {
    setState(() {
      _cameraController = controller;
    });

    // Start image stream
    controller.startImageStream((CameraImageData? imageData) {
      if (imageData != null) {
        _processCameraImage(imageData);
      }
    });
  }

  Future<void> _processCameraImage(CameraImageData imageData) async {
    // Calculate FPS
    final fps = _fpsCalculator.get();

    // Skip frames for performance
    _frameCount++;
    if (_frameCount % _frameSkip != 0) {
      return;
    }

    // Skip if already processing
    if (_isProcessing || !_isInitialized) {
      return;
    }

    _isProcessing = true;

    try {
      // Convert ARGB (macOS camera format) to BGR cv.Mat
      // macOS camera outputs ARGB, OpenCV expects BGR
      final mat =
          cv.Mat.zeros(imageData.height, imageData.width, cv.MatType.CV_8UC3);
      final bytes = imageData.bytes;
      final stride = imageData.bytesPerRow;
      final matData = mat.data;

      int dstIdx = 0;
      for (int y = 0; y < imageData.height; y++) {
        final rowStart = y * stride;
        for (int x = 0; x < imageData.width; x++) {
          final srcIdx = rowStart + x * 4;
          // Input: ARGB (macOS camera format)
          // Output: BGR (OpenCV format)
          matData[dstIdx++] = bytes[srcIdx + 3]; // B
          matData[dstIdx++] = bytes[srcIdx + 2]; // G
          matData[dstIdx++] = bytes[srcIdx + 1]; // R
        }
      }

      // Store camera size from image data
      if (_cameraSize == null) {
        setState(() {
          _cameraSize =
              Size(imageData.width.toDouble(), imageData.height.toDouble());
        });
      }

      // Match Python's 640x480 resolution (no aggressive downscaling)
      cv.Mat processedMat = mat;
      const int maxDim = 640;
      if (mat.cols > maxDim || mat.rows > maxDim) {
        final double scale =
            maxDim / (mat.cols > mat.rows ? mat.cols : mat.rows);
        processedMat = cv.resize(
          mat,
          ((mat.cols * scale).toInt(), (mat.rows * scale).toInt()),
          interpolation: cv.INTER_LINEAR,
        );
        mat.dispose();
      }

      // Run hand detection directly on cv.Mat
      final List<Hand> hands = await _handDetector.detectOnMat(processedMat);

      // Clean up
      processedMat.dispose();

      // Update UI with results
      if (mounted) {
        setState(() {
          _currentHands = hands;
          _currentFps = fps;
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
    _cameraController?.destroy();
    _handDetector.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Live Hand Detection'),
        actions: [
          // Max hands slider
          if (_isInitialized)
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const Text('Max: '),
                SizedBox(
                  width: 120,
                  child: Slider(
                    value: _maxHands.toDouble(),
                    min: 1,
                    max: 10,
                    divisions: 9,
                    label: '$_maxHands',
                    onChanged: (value) => _updateMaxHands(value.toInt()),
                  ),
                ),
                Text('$_maxHands'),
                const SizedBox(width: 16),
              ],
            ),
          if (_isInitialized && _cameraController != null)
            Padding(
              padding: const EdgeInsets.all(8.0),
              child: Center(
                child: Text(
                  '${_currentHands.length} hand(s)',
                  style: const TextStyle(fontSize: 16),
                ),
              ),
            ),
        ],
      ),
      body: _buildBody(),
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

    return Stack(
      fit: StackFit.expand,
      children: [
        // Camera preview
        CameraMacOSView(
          onCameraInizialized: _onCameraInitialized,
          cameraMode: CameraMacOSMode.photo,
          enableAudio: false,
        ),

        // Hand overlay
        if (_currentHands.isNotEmpty && _cameraSize != null)
          CustomPaint(
            painter: CameraHandOverlayPainter(
              hands: _currentHands,
              cameraSize: _cameraSize!,
            ),
          ),

        // FPS display (Python style: black outline + white text at top-left)
        Positioned(
          top: 10,
          left: 10,
          child: Stack(
            children: [
              // Black outline
              Text(
                'FPS:${_currentFps.toStringAsFixed(2)}',
                style: TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  foreground: Paint()
                    ..style = PaintingStyle.stroke
                    ..strokeWidth = 4
                    ..color = Colors.black,
                ),
              ),
              // White text
              Text(
                'FPS:${_currentFps.toStringAsFixed(2)}',
                style: const TextStyle(
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}

class CameraHandOverlayPainter extends CustomPainter {
  final List<Hand> hands;
  final Size cameraSize;

  CameraHandOverlayPainter({
    required this.hands,
    required this.cameraSize,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (hands.isEmpty) return;

    // Get image dimensions from first hand
    final int imageWidth = hands.first.imageWidth;
    final int imageHeight = hands.first.imageHeight;

    // Calculate scaling to map from image coordinates to preview coordinates
    final double imageAspect = imageWidth / imageHeight;
    final double canvasAspect = size.width / size.height;

    double scaleX, scaleY;
    double offsetX = 0, offsetY = 0;

    if (canvasAspect > imageAspect) {
      scaleY = size.height / imageHeight;
      scaleX = scaleY;
      offsetX = (size.width - imageWidth * scaleX) / 2;
    } else {
      scaleX = size.width / imageWidth;
      scaleY = scaleX;
      offsetY = (size.height - imageHeight * scaleY) / 2;
    }

    for (final hand in hands) {
      _drawBbox(canvas, hand, scaleX, scaleY, offsetX, offsetY);
      if (hand.hasLandmarks) {
        _drawConnections(canvas, hand, scaleX, scaleY, offsetX, offsetY);
        _drawLandmarks(canvas, hand, scaleX, scaleY, offsetX, offsetY);
      }
      if (hand.hasGesture) {
        _drawGesture(canvas, hand, scaleX, scaleY, offsetX, offsetY);
      }
    }
  }

  void _drawConnections(Canvas canvas, Hand hand, double scaleX, double scaleY,
      double offsetX, double offsetY) {
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
          Offset(start.x * scaleX + offsetX, start.y * scaleY + offsetY),
          Offset(end.x * scaleX + offsetX, end.y * scaleY + offsetY),
          paint,
        );
      }
    }
  }

  void _drawLandmarks(Canvas canvas, Hand hand, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    for (final HandLandmark l in hand.landmarks) {
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

  void _drawBbox(Canvas canvas, Hand hand, double scaleX, double scaleY,
      double offsetX, double offsetY) {
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
        hand.rotatedCenterX! * scaleX + offsetX,
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

    final double x1 = hand.boundingBox.left * scaleX + offsetX;
    final double y1 = hand.boundingBox.top * scaleY + offsetY;
    final double x2 = hand.boundingBox.right * scaleX + offsetX;
    final double y2 = hand.boundingBox.bottom * scaleY + offsetY;
    final Rect rect = Rect.fromLTRB(x1, y1, x2, y2);
    canvas.drawRect(rect, fillPaint);
    canvas.drawRect(rect, boxPaint);
  }

  void _drawGesture(Canvas canvas, Hand hand, double scaleX, double scaleY,
      double offsetX, double offsetY) {
    final gesture = hand.gesture;
    if (gesture == null || gesture.type == GestureType.unknown) return;

    // Only show gestures with confidence > 0.6
    if (gesture.confidence < 0.6) return;

    final emoji = _gestureToEmoji(gesture.type);
    if (emoji.isEmpty) return;

    // Position the emoji above the hand bounding box
    final double x =
        (hand.boundingBox.left + hand.boundingBox.right) / 2 * scaleX + offsetX;
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
