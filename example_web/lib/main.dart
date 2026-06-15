import 'dart:typed_data';

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:hand_detection/hand_detection.dart';

void main() {
  runApp(const HandDetectionWebApp());
}

class HandDetectionWebApp extends StatelessWidget {
  const HandDetectionWebApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Hand Detection (Web)',
      theme: ThemeData(colorSchemeSeed: Colors.blue, useMaterial3: true),
      home: const WebDemoScreen(),
    );
  }
}

/// Web demo: pick an image, run the LiteRT.js + Canvas hand pipeline, and draw
/// boxes / skeleton / gesture labels. Lets you switch the LiteRT.js accelerator
/// (auto -> WebGPU with WASM fallback, or force WASM).
class WebDemoScreen extends StatefulWidget {
  const WebDemoScreen({super.key});

  @override
  State<WebDemoScreen> createState() => _WebDemoScreenState();
}

class _WebDemoScreenState extends State<WebDemoScreen> {
  HandDetector? _detector;
  bool _initializing = true;
  String? _error;

  /// Requested accelerator: 'auto' | 'webgpu' | 'wasm'.
  String _accelerator = 'auto';
  bool _enableGestures = true;

  Uint8List? _imageBytes;
  List<Hand> _hands = const [];
  bool _detecting = false;
  int? _inferenceMs;

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<void> _initDetector() async {
    setState(() {
      _initializing = true;
      _error = null;
    });
    try {
      final detector = await HandDetector.create(
        mode: HandMode.boxesAndLandmarks,
        detectorConf: 0.5,
        maxDetections: 10,
        minLandmarkScore: 0.5,
        enableGestures: _enableGestures,
        gestureMinConfidence: 0.5,
        liteRtAccelerator: _accelerator,
      );
      if (!mounted) {
        await detector.dispose();
        return;
      }
      setState(() {
        _detector = detector;
        _initializing = false;
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _initializing = false;
          _error = 'Failed to initialize: $e';
        });
      }
    }
  }

  Future<void> _reinitDetector() async {
    final old = _detector;
    _detector = null;
    setState(() {
      _hands = const [];
      _inferenceMs = null;
    });
    await old?.dispose();
    await _initDetector();
  }

  Future<void> _pickImage() async {
    final detector = _detector;
    if (detector == null) return;
    const typeGroup = XTypeGroup(
      label: 'images',
      extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'],
    );
    final XFile? file = await openFile(acceptedTypeGroups: [typeGroup]);
    if (file == null) return;

    final bytes = await file.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _hands = const [];
      _detecting = true;
      _error = null;
    });

    try {
      final sw = Stopwatch()..start();
      final hands = await detector.detect(bytes);
      sw.stop();
      if (!mounted) return;
      setState(() {
        _hands = hands;
        _inferenceMs = sw.elapsedMilliseconds;
        _detecting = false;
      });
    } catch (e) {
      if (mounted) {
        setState(() {
          _detecting = false;
          _error = 'Detection failed: $e';
        });
      }
    }
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Hand Detection (Web)'),
        actions: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8),
            child: Row(
              children: [
                const Icon(Icons.memory, size: 18),
                const SizedBox(width: 6),
                DropdownButtonHideUnderline(
                  child: DropdownButton<String>(
                    value: _accelerator,
                    items: const [
                      DropdownMenuItem(value: 'auto', child: Text('auto')),
                      DropdownMenuItem(value: 'webgpu', child: Text('webgpu')),
                      DropdownMenuItem(value: 'wasm', child: Text('wasm')),
                    ],
                    onChanged: (_initializing || _detecting)
                        ? null
                        : (v) {
                            if (v == null || v == _accelerator) return;
                            setState(() => _accelerator = v);
                            _reinitDetector();
                          },
                  ),
                ),
                const SizedBox(width: 8),
                const Text('Gestures'),
                Switch(
                  value: _enableGestures,
                  onChanged: (_initializing || _detecting)
                      ? null
                      : (v) {
                          setState(() => _enableGestures = v);
                          _reinitDetector();
                        },
                ),
              ],
            ),
          ),
        ],
      ),
      floatingActionButton: (_detector != null && !_detecting)
          ? FloatingActionButton.extended(
              onPressed: _pickImage,
              icon: const Icon(Icons.add_photo_alternate),
              label: const Text('Pick Image'),
            )
          : null,
      body: _buildBody(),
    );
  }

  Widget _buildBody() {
    if (_initializing) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Loading LiteRT.js and models...'),
          ],
        ),
      );
    }
    if (_error != null && _imageBytes == null) {
      return Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, size: 56, color: Colors.red),
              const SizedBox(height: 12),
              Text(
                _error!,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.red),
              ),
              const SizedBox(height: 12),
              ElevatedButton(
                onPressed: _initDetector,
                child: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    return Column(
      children: [
        _statusBar(),
        const Divider(height: 1),
        Expanded(
          child: _imageBytes == null
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.pan_tool_outlined,
                        size: 96,
                        color: Colors.grey[400],
                      ),
                      const SizedBox(height: 16),
                      Text(
                        'Pick an image to detect hands',
                        style: TextStyle(fontSize: 18, color: Colors.grey[600]),
                      ),
                    ],
                  ),
                )
              : Stack(
                  children: [
                    Positioned.fill(
                      child: Image.memory(_imageBytes!, fit: BoxFit.contain),
                    ),
                    Positioned.fill(
                      child: CustomPaint(
                        painter: MultiOverlayPainter(results: _hands),
                      ),
                    ),
                    if (_detecting)
                      const Positioned.fill(
                        child: ColoredBox(
                          color: Color(0x33000000),
                          child: Center(child: CircularProgressIndicator()),
                        ),
                      ),
                  ],
                ),
        ),
        if (_hands.isNotEmpty) _gesturePanel(),
      ],
    );
  }

  Widget _statusBar() {
    final acc = _detector?.activeAccelerator ?? _accelerator;
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
      child: Row(
        children: [
          _chip(Icons.bolt, 'accelerator: $acc'),
          const SizedBox(width: 8),
          _chip(Icons.back_hand, 'hands: ${_hands.length}'),
          if (_inferenceMs != null) ...[
            const SizedBox(width: 8),
            _chip(Icons.timer, '${_inferenceMs}ms'),
          ],
        ],
      ),
    );
  }

  Widget _chip(IconData icon, String label) {
    return Chip(
      avatar: Icon(icon, size: 16),
      label: Text(label),
      visualDensity: VisualDensity.compact,
    );
  }

  Widget _gesturePanel() {
    return Container(
      width: double.infinity,
      color: Theme.of(context).colorScheme.surfaceContainerHighest,
      padding: const EdgeInsets.all(12),
      child: Wrap(
        spacing: 8,
        runSpacing: 8,
        children: [
          for (int i = 0; i < _hands.length; i++) _handChip(i, _hands[i]),
        ],
      ),
    );
  }

  Widget _handChip(int index, Hand hand) {
    final String handed;
    if (hand.handedness == null) {
      handed = '?';
    } else {
      handed = hand.handedness == Handedness.right ? 'Right' : 'Left';
    }
    final g = hand.gesture;
    final String gesture;
    if (g != null && g.type != GestureType.unknown) {
      final pct = (g.confidence * 100).toStringAsFixed(0);
      gesture = '${g.type.name} ($pct%)';
    } else {
      gesture = 'none';
    }
    return Chip(label: Text('Hand ${index + 1}: $handed · $gesture'));
  }
}
