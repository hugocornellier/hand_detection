import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:file_selector/file_selector.dart';
import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_colorpicker/flutter_colorpicker.dart';
import 'package:camera/camera.dart';
import 'package:hand_detection/hand_detection_native.dart';
import 'package:flutter_litert/flutter_litert.dart'
    show FrameThrottle, OneEuroFilter;
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:path_provider/path_provider.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:video_player/video_player.dart';

/// Centers [child] in the available space; when the viewport is too small to
/// fit it, the content scrolls vertically instead of overflowing.
class _ScrollableCentered extends StatelessWidget {
  final Widget child;
  const _ScrollableCentered({required this.child});

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) => SingleChildScrollView(
        child: ConstrainedBox(
          constraints: BoxConstraints(minHeight: constraints.maxHeight),
          child: Center(child: child),
        ),
      ),
    );
  }
}

/// Compact labeled swatch that opens a color picker dialog on tap.
class _ColorPickerButton extends StatelessWidget {
  final String label;
  final Color color;
  final ValueChanged<Color> onColorChanged;

  const _ColorPickerButton({
    required this.label,
    required this.color,
    required this.onColorChanged,
  });

  void _pick(BuildContext context) {
    Color tempColor = color;
    showDialog<void>(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('Pick $label Color'),
        content: SingleChildScrollView(
          child: ColorPicker(
            pickerColor: color,
            onColorChanged: (c) => tempColor = c,
            pickerAreaHeightPercent: 0.8,
            displayThumbColor: true,
            enableAlpha: true,
            labelTypes: const [ColorLabelType.hex],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              onColorChanged(tempColor);
              Navigator.of(context).pop();
            },
            child: const Text('Select'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => _pick(context),
      borderRadius: BorderRadius.circular(6),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
        decoration: BoxDecoration(
          border: Border.all(color: Colors.grey.shade300),
          borderRadius: BorderRadius.circular(6),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 18,
              height: 18,
              decoration: BoxDecoration(
                color: color,
                border: Border.all(color: Colors.grey.shade400),
                borderRadius: BorderRadius.circular(3),
              ),
            ),
            const SizedBox(width: 6),
            Text(label, style: const TextStyle(fontSize: 12)),
            const SizedBox(width: 2),
            const Icon(Icons.arrow_drop_down, size: 16),
          ],
        ),
      ),
    );
  }
}

/// Compact checkbox with an inline label, sized for dense settings panels.
class CompactCheckbox extends StatelessWidget {
  final String label;
  final bool value;
  final ValueChanged<bool?> onChanged;

  const CompactCheckbox({
    super.key,
    required this.label,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: () => onChanged(!value),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: 24,
            height: 24,
            child: Checkbox(
              value: value,
              onChanged: onChanged,
              materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
              visualDensity: VisualDensity.compact,
            ),
          ),
          const SizedBox(width: 4),
          Text(label, style: const TextStyle(fontSize: 12)),
        ],
      ),
    );
  }
}

/// Compact slider with a fixed-width leading label, sized for dense settings
/// panels.
class CompactSlider extends StatelessWidget {
  final String label;
  final double value;
  final double min;
  final double max;
  final ValueChanged<double> onChanged;

  const CompactSlider({
    super.key,
    required this.label,
    required this.value,
    required this.min,
    required this.max,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2.0),
      child: Row(
        children: [
          SizedBox(
            width: 70,
            child: Text(label, style: const TextStyle(fontSize: 12)),
          ),
          Expanded(
            child: SliderTheme(
              data: SliderTheme.of(context).copyWith(
                trackHeight: 2.0,
                thumbShape:
                    const RoundSliderThumbShape(enabledThumbRadius: 6.0),
                overlayShape:
                    const RoundSliderOverlayShape(overlayRadius: 12.0),
              ),
              child: Slider(
                value: value,
                min: min,
                max: max,
                divisions: ((max - min) * 10).round(),
                label: value.toStringAsFixed(1),
                onChanged: onChanged,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Classify detection-time in milliseconds into a display-friendly bucket
/// (`label`, `color`, `icon`) for overlay status indicators.
({String label, Color color, IconData icon}) performanceLevel(int ms) {
  if (ms < 200) {
    return (label: 'Excellent', color: Colors.green, icon: Icons.speed);
  } else if (ms < 500) {
    return (label: 'Good', color: Colors.lightGreen, icon: Icons.thumb_up);
  } else if (ms < 1000) {
    return (label: 'Fair', color: Colors.orange, icon: Icons.warning_amber);
  } else {
    return (label: 'Slow', color: Colors.red, icon: Icons.hourglass_bottom);
  }
}

/// Compact tappable badge that displays the total processing time plus a
/// color-coded performance indicator. Tapping opens a dialog with detection
/// timing and detected-hand details.
///
/// Designed as a drop-in overlay for the still-image hand detection flow.
class TimingBadge extends StatelessWidget {
  final int totalMs;
  final int? detectionMs;
  final int handCount;
  final bool gesturesEnabled;

  const TimingBadge({
    super.key,
    required this.totalMs,
    this.detectionMs,
    this.handCount = 0,
    this.gesturesEnabled = false,
  });

  @override
  Widget build(BuildContext context) {
    final perf = performanceLevel(totalMs);
    return GestureDetector(
      onTap: () => _showDetails(context),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: Colors.black.withAlpha(179),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(perf.icon, size: 14, color: perf.color),
            const SizedBox(width: 6),
            Text(
              '${totalMs}ms',
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
                fontSize: 12,
              ),
            ),
            const SizedBox(width: 4),
            Text(perf.label, style: TextStyle(color: perf.color, fontSize: 12)),
            const SizedBox(width: 4),
            const Icon(Icons.info_outline, size: 12, color: Colors.white54),
          ],
        ),
      ),
    );
  }

  void _showDetails(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Row(
          children: const [
            Icon(Icons.timer, color: Colors.blue),
            SizedBox(width: 8),
            Text('Processing Details'),
          ],
        ),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            _TimingRow(
              label: 'Hands detected',
              value: '$handCount',
              color: Colors.green,
            ),
            _TimingRow(
              label: 'Gestures',
              value: gesturesEnabled ? 'On' : 'Off',
              color: gesturesEnabled ? Colors.green : Colors.grey,
            ),
            const Divider(height: 16),
            if (detectionMs != null)
              _TimingRow(
                label: 'Detection',
                value: '${detectionMs}ms',
                color: Colors.green,
              ),
            _TimingRow(
              label: 'Total',
              value: '${totalMs}ms',
              color: Colors.blue,
              isBold: true,
            ),
            const SizedBox(height: 12),
            _PerformanceIndicator(totalMs: totalMs),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}

class _TimingRow extends StatelessWidget {
  final String label;
  final String value;
  final Color color;
  final bool isBold;

  const _TimingRow({
    required this.label,
    required this.value,
    required this.color,
    this.isBold = false,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(color: color, shape: BoxShape.circle),
              ),
              const SizedBox(width: 8),
              Text(
                label,
                style: TextStyle(
                  fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
                  fontSize: isBold ? 15 : 14,
                ),
              ),
            ],
          ),
          Text(
            value,
            style: TextStyle(
              fontWeight: isBold ? FontWeight.bold : FontWeight.normal,
              fontSize: isBold ? 15 : 14,
              color: color,
            ),
          ),
        ],
      ),
    );
  }
}

class _PerformanceIndicator extends StatelessWidget {
  final int totalMs;

  const _PerformanceIndicator({required this.totalMs});

  @override
  Widget build(BuildContext context) {
    final perf = performanceLevel(totalMs);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: perf.color.withAlpha(26),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: perf.color.withAlpha(77)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(perf.icon, size: 16, color: perf.color),
          const SizedBox(width: 6),
          Text(
            perf.label,
            style: TextStyle(
              color: perf.color,
              fontWeight: FontWeight.bold,
              fontSize: 14,
            ),
          ),
        ],
      ),
    );
  }
}

String _gestureLabel(GestureType g) {
  switch (g) {
    case GestureType.thumbUp:
      return 'Thumb Up';
    case GestureType.thumbDown:
      return 'Thumb Down';
    case GestureType.victory:
      return 'Victory';
    case GestureType.openPalm:
      return 'Open Palm';
    case GestureType.closedFist:
      return 'Closed Fist';
    case GestureType.pointingUp:
      return 'Pointing Up';
    case GestureType.iLoveYou:
      return 'I Love You';
    case GestureType.unknown:
      return '';
  }
}

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(MaterialApp(
    debugShowCheckedModeBanner: false,
    title: 'Hand Detection Demo',
    theme: ThemeData(
      colorSchemeSeed: Colors.blue,
      useMaterial3: true,
    ),
    home: const HomeScreen(),
  ));
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Hand Detection Demo'),
      ),
      body: _ScrollableCentered(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 720),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 24),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 16),
                  child: Text(
                    'Choose a Demo',
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                ),
                const SizedBox(height: 28),
                _buildSection(
                  context,
                  'Hand Detection / Landmarks',
                  [
                    _buildModeCard(
                      context,
                      icon: Icons.videocam,
                      title: 'Live Camera',
                      description: 'Real-time hand detection from camera feed',
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => const LiveCameraScreen()),
                        );
                      },
                    ),
                    _buildModeCard(
                      context,
                      icon: Icons.image,
                      title: 'Still Image',
                      description:
                          'Detect hands in photos from gallery or camera',
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => const Example()),
                        );
                      },
                    ),
                    _buildModeCard(
                      context,
                      icon: Icons.movie_creation_outlined,
                      title: 'Video File',
                      description:
                          'Process an MP4 frame-by-frame with smoothed '
                          'hand detection',
                      onTap: () {
                        Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (context) => const VideoFileScreen()),
                        );
                      },
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildSection(
    BuildContext context,
    String title,
    List<Widget> cards,
  ) {
    final List<Widget> row = [];
    for (int i = 0; i < cards.length; i++) {
      if (i > 0) row.add(const SizedBox(width: 12));
      row.add(cards[i]);
    }
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: Text(
            title,
            style: Theme.of(context).textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: Colors.grey[700],
                ),
          ),
        ),
        const SizedBox(height: 12),
        SingleChildScrollView(
          scrollDirection: Axis.horizontal,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: IntrinsicHeight(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: row,
            ),
          ),
        ),
      ],
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
      width: 190,
      child: Card(
        elevation: 4,
        clipBehavior: Clip.antiAlias,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(12),
          child: Padding(
            padding: const EdgeInsets.all(16),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, size: 40, color: Colors.blue),
                const SizedBox(height: 12),
                Text(
                  title,
                  style: Theme.of(context).textTheme.titleMedium,
                  textAlign: TextAlign.center,
                ),
                const SizedBox(height: 6),
                Text(
                  description,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.grey[600],
                      ),
                  textAlign: TextAlign.center,
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

// ─────────────────────────── Still Image Screen ───────────────────────────

class Example extends StatefulWidget {
  const Example({super.key});
  @override
  State<Example> createState() => _ExampleState();
}

class _ExampleState extends State<Example> {
  HandDetector? _handDetector;
  Uint8List? _imageBytes;
  List<Hand> _hands = [];
  Size? _originalSize;

  bool _isLoading = false;
  bool _showBoundingBoxes = true;
  bool _showSkeleton = true;
  bool _showLandmarks = true;
  bool _showHandedness = true;
  bool _showGestures = true;
  bool _showLandmarkLabels = false;

  int? _detectionTimeMs;
  int? _totalTimeMs;

  Color _boundingBoxColor = const Color(0xFFFF9800);
  Color _landmarkColor = const Color(0xFFFF3D00);
  Color _skeletonColor = const Color(0xFF00E676);

  double _boundingBoxThickness = 2.0;
  double _landmarkSize = 3.0;
  double _skeletonThickness = 3.0;

  int _maxHands = 4;
  bool _enableGestures = true;

  @override
  void initState() {
    super.initState();
    _initHandDetector();
  }

  Future<void> _initHandDetector() async {
    try {
      await _handDetector?.dispose();
      _handDetector = await HandDetector.create(
        mode: HandMode.boxesAndLandmarks,
        landmarkModel: HandLandmarkModel.full,
        detectorConf: 0.6,
        maxDetections: _maxHands,
        minLandmarkScore: 0.5,
        performanceConfig: const PerformanceConfig.xnnpack(),
        enableGestures: _enableGestures,
        gestureMinConfidence: 0.5,
      );
    } catch (_) {}
    if (mounted) setState(() {});
  }

  @override
  void dispose() {
    _handDetector?.dispose();
    super.dispose();
  }

  Future<void> _pickAndRun() async {
    final ImagePicker picker = ImagePicker();
    final XFile? picked =
        await picker.pickImage(source: ImageSource.gallery, imageQuality: 100);
    if (picked == null) return;

    setState(() {
      _imageBytes = null;
      _hands = [];
      _originalSize = null;
      _isLoading = true;
      _detectionTimeMs = null;
      _totalTimeMs = null;
    });

    final Uint8List bytes = await picked.readAsBytes();

    if (_handDetector == null || !_handDetector!.isReady) {
      setState(() => _isLoading = false);
      return;
    }

    await _processImage(bytes);
  }

  Future<void> _processImage(Uint8List bytes) async {
    setState(() => _isLoading = true);

    final DateTime totalStart = DateTime.now();
    final DateTime detectionStart = DateTime.now();
    final List<Hand> hands = await _handDetector!.detect(bytes);
    final DateTime detectionEnd = DateTime.now();

    Size decodedSize;
    if (hands.isNotEmpty) {
      decodedSize = Size(
        hands.first.imageWidth.toDouble(),
        hands.first.imageHeight.toDouble(),
      );
    } else {
      final codec = await ui.instantiateImageCodec(bytes);
      final frame = await codec.getNextFrame();
      decodedSize =
          Size(frame.image.width.toDouble(), frame.image.height.toDouble());
      frame.image.dispose();
    }

    if (!mounted) return;

    final DateTime totalEnd = DateTime.now();
    setState(() {
      _imageBytes = bytes;
      _originalSize = decodedSize;
      _hands = hands;
      _isLoading = false;
      _detectionTimeMs = detectionEnd.difference(detectionStart).inMilliseconds;
      _totalTimeMs = totalEnd.difference(totalStart).inMilliseconds;
    });
  }

  void _showSettingsSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.3,
        maxChildSize: 0.9,
        builder: (context, scrollController) => StatefulBuilder(
          builder: (context, setSheetState) {
            void updateState(VoidCallback fn) {
              fn();
              setSheetState(() {});
              setState(() {});
            }

            // Settings that change the detector pipeline require a re-init and
            // a re-run on the current image (max hands, gesture recognition).
            Future<void> onDetectorSettingChange(VoidCallback fn) async {
              fn();
              setSheetState(() {});
              setState(() {});
              await _initHandDetector();
              if (_imageBytes != null) {
                await _processImage(_imageBytes!);
              }
            }

            Widget cb(String label, bool v, void Function(bool) set) =>
                CompactCheckbox(
                    label: label,
                    value: v,
                    onChanged: (x) => updateState(() => set(x ?? false)));
            Widget col(String label, Color c, void Function(Color) set) =>
                _ColorPickerButton(
                    label: label,
                    color: c,
                    onColorChanged: (x) => updateState(() => set(x)));
            Widget sl(String label, double v, double mn, double mx,
                    void Function(double) set) =>
                CompactSlider(
                    label: label,
                    value: v,
                    min: mn,
                    max: mx,
                    onChanged: (x) => updateState(() => set(x)));

            return Container(
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
              ),
              child: Column(
                children: [
                  Container(
                    margin: const EdgeInsets.symmetric(vertical: 8),
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.grey[300],
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  Expanded(
                    child: ListView(
                      controller: scrollController,
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      children: [
                        ExpansionTile(
                          title: const Text('Display Options',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          initiallyExpanded: true,
                          children: [
                            Wrap(
                              spacing: 8,
                              runSpacing: 4,
                              children: [
                                cb('Bounding Boxes', _showBoundingBoxes,
                                    (v) => _showBoundingBoxes = v),
                                cb('Skeleton', _showSkeleton,
                                    (v) => _showSkeleton = v),
                                cb('Landmarks', _showLandmarks,
                                    (v) => _showLandmarks = v),
                                cb('Handedness', _showHandedness,
                                    (v) => _showHandedness = v),
                                cb('Gestures', _showGestures,
                                    (v) => _showGestures = v),
                                cb('Landmark Labels', _showLandmarkLabels,
                                    (v) => _showLandmarkLabels = v),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Detection',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          children: [
                            Padding(
                              padding:
                                  const EdgeInsets.symmetric(vertical: 4.0),
                              child: Row(
                                children: [
                                  const SizedBox(
                                      width: 70,
                                      child: Text('Max Hands',
                                          style: TextStyle(fontSize: 12))),
                                  Expanded(
                                    child: Slider(
                                      value: _maxHands.toDouble(),
                                      min: 1,
                                      max: 10,
                                      divisions: 9,
                                      label: '$_maxHands',
                                      onChanged: (v) => setSheetState(
                                          () => _maxHands = v.toInt()),
                                      onChangeEnd: (v) =>
                                          onDetectorSettingChange(
                                              () => _maxHands = v.toInt()),
                                    ),
                                  ),
                                  SizedBox(
                                      width: 24,
                                      child: Text('$_maxHands',
                                          textAlign: TextAlign.right)),
                                ],
                              ),
                            ),
                            SwitchListTile(
                              dense: true,
                              contentPadding: EdgeInsets.zero,
                              title: const Text('Gesture recognition'),
                              value: _enableGestures,
                              onChanged: (v) => onDetectorSettingChange(
                                  () => _enableGestures = v),
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Colors',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          children: [
                            Wrap(
                              spacing: 6,
                              runSpacing: 6,
                              children: [
                                col('BBox', _boundingBoxColor,
                                    (c) => _boundingBoxColor = c),
                                col('Landmarks', _landmarkColor,
                                    (c) => _landmarkColor = c),
                                col('Skeleton', _skeletonColor,
                                    (c) => _skeletonColor = c),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Sizes',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          children: [
                            sl('BBox', _boundingBoxThickness, 0.5, 10.0,
                                (v) => _boundingBoxThickness = v),
                            sl('Landmark', _landmarkSize, 0.5, 15.0,
                                (v) => _landmarkSize = v),
                            sl('Skeleton', _skeletonThickness, 0.5, 10.0,
                                (v) => _skeletonThickness = v),
                            const SizedBox(height: 8),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final bool hasImage = _imageBytes != null && _originalSize != null;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Still Image Detection'),
        actions: [
          IconButton(
            onPressed: _pickAndRun,
            icon: const Icon(Icons.add_photo_alternate),
            tooltip: 'Pick Image',
          ),
          IconButton(
            onPressed: _showSettingsSheet,
            icon: const Icon(Icons.tune),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: Stack(
        children: [
          Center(
            child: hasImage
                ? LayoutBuilder(
                    builder: (context, constraints) {
                      final fitted = applyBoxFit(
                        BoxFit.contain,
                        _originalSize!,
                        Size(constraints.maxWidth, constraints.maxHeight),
                      );
                      final Size renderSize = fitted.destination;
                      final Rect imageRect = Alignment.center.inscribe(
                        renderSize,
                        Offset.zero &
                            Size(constraints.maxWidth, constraints.maxHeight),
                      );

                      return Stack(
                        children: [
                          Positioned.fromRect(
                            rect: imageRect,
                            child: SizedBox.fromSize(
                              size: renderSize,
                              child: Image.memory(
                                _imageBytes!,
                                fit: BoxFit.fill,
                              ),
                            ),
                          ),
                          Positioned(
                            left: imageRect.left,
                            top: imageRect.top,
                            width: imageRect.width,
                            height: imageRect.height,
                            child: CustomPaint(
                              size: Size(imageRect.width, imageRect.height),
                              painter: HandDetectionsPainter(
                                hands: _hands,
                                imageRectOnCanvas: Rect.fromLTWH(
                                    0, 0, imageRect.width, imageRect.height),
                                originalImageSize: _originalSize!,
                                showBoundingBoxes: _showBoundingBoxes,
                                showSkeleton: _showSkeleton,
                                showLandmarks: _showLandmarks,
                                showLandmarkLabels: _showLandmarkLabels,
                                showHandedness: _showHandedness,
                                showGestures: _showGestures,
                                boundingBoxColor: _boundingBoxColor,
                                landmarkColor: _landmarkColor,
                                skeletonColor: _skeletonColor,
                                boundingBoxThickness: _boundingBoxThickness,
                                landmarkSize: _landmarkSize,
                                skeletonThickness: _skeletonThickness,
                              ),
                            ),
                          ),
                        ],
                      );
                    },
                  )
                : _ScrollableCentered(
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.add_photo_alternate,
                            size: 80, color: Colors.grey[300]),
                        const SizedBox(height: 16),
                        Text(
                          'No image selected',
                          style:
                              TextStyle(fontSize: 18, color: Colors.grey[600]),
                        ),
                        const SizedBox(height: 8),
                        Text(
                          'Tap the + icon to pick an image',
                          style:
                              TextStyle(fontSize: 14, color: Colors.grey[500]),
                        ),
                      ],
                    ),
                  ),
          ),
          if (hasImage && _totalTimeMs != null)
            Positioned(
              top: 12,
              left: 12,
              child: TimingBadge(
                totalMs: _totalTimeMs!,
                detectionMs: _detectionTimeMs,
                handCount: _hands.length,
                gesturesEnabled: _enableGestures,
              ),
            ),
          if (_isLoading)
            Container(
              color: Colors.black54,
              child: const Center(
                child: CircularProgressIndicator(),
              ),
            ),
        ],
      ),
    );
  }
}

/// Paints hand detection results over a still image, mapping original-image
/// pixel coordinates onto the displayed image rect. All overlays (boxes,
/// skeleton, landmarks, labels) are individually toggleable and styleable.
class HandDetectionsPainter extends CustomPainter {
  final List<Hand> hands;
  final Rect imageRectOnCanvas;
  final Size originalImageSize;
  final bool showBoundingBoxes;
  final bool showSkeleton;
  final bool showLandmarks;
  final bool showLandmarkLabels;
  final bool showHandedness;
  final bool showGestures;
  final Color boundingBoxColor;
  final Color landmarkColor;
  final Color skeletonColor;
  final double boundingBoxThickness;
  final double landmarkSize;
  final double skeletonThickness;

  HandDetectionsPainter({
    required this.hands,
    required this.imageRectOnCanvas,
    required this.originalImageSize,
    required this.showBoundingBoxes,
    required this.showSkeleton,
    required this.showLandmarks,
    required this.showLandmarkLabels,
    required this.showHandedness,
    required this.showGestures,
    required this.boundingBoxColor,
    required this.landmarkColor,
    required this.skeletonColor,
    required this.boundingBoxThickness,
    required this.landmarkSize,
    required this.skeletonThickness,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (hands.isEmpty) return;

    final Paint boxPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = boundingBoxThickness
      ..color = boundingBoxColor;

    final Paint lmPaint = Paint()
      ..style = PaintingStyle.fill
      ..color = landmarkColor;

    final Paint skPaint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = skeletonThickness
      ..strokeCap = StrokeCap.round
      ..color = skeletonColor;

    final double ox = imageRectOnCanvas.left;
    final double oy = imageRectOnCanvas.top;
    final double scaleX = imageRectOnCanvas.width / originalImageSize.width;
    final double scaleY = imageRectOnCanvas.height / originalImageSize.height;

    Offset map(double x, double y) => Offset(ox + x * scaleX, oy + y * scaleY);

    for (final Hand hand in hands) {
      if (showSkeleton && hand.hasLandmarks) {
        for (final c in handLandmarkConnections) {
          final HandLandmark? a = hand.getLandmark(c[0]);
          final HandLandmark? b = hand.getLandmark(c[1]);
          if (a != null &&
              b != null &&
              a.visibility > 0.5 &&
              b.visibility > 0.5) {
            canvas.drawLine(map(a.x, a.y), map(b.x, b.y), skPaint);
          }
        }
      }

      if (showLandmarks && hand.hasLandmarks) {
        for (final HandLandmark lm in hand.landmarks) {
          if (lm.visibility <= 0.5) continue;
          final Offset center = map(lm.x, lm.y);
          canvas.drawCircle(center, landmarkSize, lmPaint);
          if (showLandmarkLabels) {
            _drawText(
                canvas, '${lm.type.index}', center + const Offset(5, -5), 9);
          }
        }
      }

      if (showBoundingBoxes) {
        final BoundingBox bb = hand.boundingBox;
        final Rect rect = Rect.fromLTRB(
          ox + bb.left * scaleX,
          oy + bb.top * scaleY,
          ox + bb.right * scaleX,
          oy + bb.bottom * scaleY,
        );
        canvas.drawRect(rect, boxPaint);

        final List<String> parts = ['${(hand.score * 100).round()}%'];
        if (showHandedness && hand.handedness != null) {
          parts.add(hand.handedness == Handedness.right ? 'Right' : 'Left');
        }
        if (showGestures &&
            hand.gesture != null &&
            hand.gesture!.type != GestureType.unknown) {
          parts.add(_gestureLabel(hand.gesture!.type));
        }
        _drawLabelChip(canvas, parts.join('  •  '), Offset(rect.left, rect.top),
            boundingBoxColor);
      }
    }
  }

  void _drawText(Canvas canvas, String text, Offset at, double fontSize) {
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: Colors.white,
          fontSize: fontSize,
          shadows: const [Shadow(color: Colors.black, blurRadius: 2)],
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    tp.paint(canvas, at);
  }

  void _drawLabelChip(Canvas canvas, String text, Offset topLeft, Color color) {
    if (text.isEmpty) return;
    final tp = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(
          color: Colors.black,
          fontSize: 11,
          fontWeight: FontWeight.bold,
        ),
      ),
      textDirection: TextDirection.ltr,
    )..layout();

    const double padX = 4, padY = 2;
    final double w = tp.width + padX * 2;
    final double h = tp.height + padY * 2;
    final double top = (topLeft.dy - h).clamp(0.0, double.infinity);
    final Rect chip = Rect.fromLTWH(topLeft.dx, top, w, h);
    canvas.drawRect(chip, Paint()..color = color);
    tp.paint(canvas, Offset(topLeft.dx + padX, top + padY));
  }

  @override
  bool shouldRepaint(covariant HandDetectionsPainter old) {
    return old.hands != hands ||
        old.imageRectOnCanvas != imageRectOnCanvas ||
        old.originalImageSize != originalImageSize ||
        old.showBoundingBoxes != showBoundingBoxes ||
        old.showSkeleton != showSkeleton ||
        old.showLandmarks != showLandmarks ||
        old.showLandmarkLabels != showLandmarkLabels ||
        old.showHandedness != showHandedness ||
        old.showGestures != showGestures ||
        old.boundingBoxColor != boundingBoxColor ||
        old.landmarkColor != landmarkColor ||
        old.skeletonColor != skeletonColor ||
        old.boundingBoxThickness != boundingBoxThickness ||
        old.landmarkSize != landmarkSize ||
        old.skeletonThickness != skeletonThickness;
  }
}

// ─────────────────────────── Live Camera Screen ───────────────────────────

class LiveCameraScreen extends StatefulWidget {
  const LiveCameraScreen({super.key});

  @override
  State<LiveCameraScreen> createState() => _LiveCameraScreenState();
}

class _LiveCameraScreenState extends State<LiveCameraScreen> {
  CameraController? _cameraController;
  List<CameraDescription> _availableCameras = const [];
  HandDetector? _handDetector;
  List<Hand> _hands = [];
  Size? _imageSize;
  int? _sensorOrientation;
  bool _isFrontCamera = false;
  bool _isSwitchingCamera = false;
  final FrameThrottle _throttle = FrameThrottle();
  bool _isInitialized = false;
  DeviceOrientation _deviceOrientation = DeviceOrientation.portraitUp;
  StreamSubscription<AccelerometerEvent>? _accelerometerSub;
  int _detectionTimeMs = 0;
  final FpsCounter _fpsCounter = FpsCounter();
  int _fps = 0;
  bool _isImageStreamStarted = false;

  int _maxHands = 2;
  bool _enableGestures = true;
  bool _enableTracking = false;

  // Live backend benchmarking: default to CompiledModel, with a one-tap
  // XNNPACK fallback for immediate A/B checks in the camera view.
  bool _useCompiledModel = true;
  final List<int> _recentInferenceMs = [];
  int _detThisSec = 0;

  @override
  void initState() {
    super.initState();
    _initCamera();

    if (!kIsWeb && (Platform.isAndroid || Platform.isIOS)) {
      _accelerometerSub = accelerometerEventStream().listen((event) {
        final next = event.x.abs() > event.y.abs()
            ? (event.x > 0
                ? DeviceOrientation.landscapeLeft
                : DeviceOrientation.landscapeRight)
            : (event.y > 0
                ? DeviceOrientation.portraitUp
                : DeviceOrientation.portraitDown);
        if (next == DeviceOrientation.portraitDown &&
            (_deviceOrientation == DeviceOrientation.landscapeLeft ||
                _deviceOrientation == DeviceOrientation.landscapeRight)) {
          return;
        }
        if (next != _deviceOrientation && mounted) {
          setState(() => _deviceOrientation = next);
        }
      });
    }
  }

  Future<HandDetector> _createDetector(bool useCompiledModel) {
    return HandDetector.create(
      mode: HandMode.boxesAndLandmarks,
      landmarkModel: HandLandmarkModel.full,
      detectorConf: 0.6,
      maxDetections: _maxHands,
      minLandmarkScore: 0.5,
      enableTracking: _enableTracking,
      performanceConfig: const PerformanceConfig.xnnpack(),
      enableGestures: _enableGestures,
      gestureMinConfidence: 0.5,
      useCompiledModel: useCompiledModel,
    );
  }

  /// (Re)creates the detector, falling back to XNNPACK if CompiledModel init
  /// fails on this device.
  Future<void> _reinitDetector() async {
    final old = _handDetector;
    _handDetector = null;
    await old?.dispose();
    try {
      _handDetector = await _createDetector(_useCompiledModel);
      return;
    } catch (e) {
      if (!_useCompiledModel) rethrow;
      debugPrint(
          'Live camera CompiledModel init failed; falling back to XNNPACK: $e');
      if (mounted) {
        setState(() => _useCompiledModel = false);
      } else {
        _useCompiledModel = false;
      }
      _handDetector = await _createDetector(false);
    }
  }

  Future<void> _toggleAccelerator() async {
    setState(() {
      _isInitialized = false;
      _useCompiledModel = !_useCompiledModel;
      _recentInferenceMs.clear();
    });
    // ignore: avoid_print
    print('[live-bench] switching backend -> '
        '${_useCompiledModel ? 'compiledmodel' : 'xnnpack'}');
    await _reinitDetector();
    if (mounted) setState(() => _isInitialized = true);
  }

  Future<void> _updateDetectorSettings(VoidCallback fn) async {
    setState(() {
      _isInitialized = false;
      fn();
    });
    await _reinitDetector();
    if (mounted) setState(() => _isInitialized = true);
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
              TextButton(
                onPressed: _isInitialized ? _toggleAccelerator : null,
                style: TextButton.styleFrom(
                  minimumSize: const Size(48, 36),
                  padding: const EdgeInsets.symmetric(horizontal: 8),
                ),
                child: Text(
                  _useCompiledModel ? 'CM' : 'XNN',
                  style: const TextStyle(
                    color: Colors.amberAccent,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
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
                        setMenuState(() => _maxHands = value.toInt()),
                    onChangeEnd: (value) => _updateDetectorSettings(
                        () => _maxHands = value.toInt()),
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
                    setMenuState(() => _enableGestures = value);
                    _updateDetectorSettings(() => _enableGestures = value);
                  },
                ),
              ],
            ),
            const Divider(color: Colors.white24, height: 24),
            const Text('TRACKING', style: sectionLabelStyle),
            const SizedBox(height: 4),
            Row(
              children: [
                const Expanded(
                  child: Text(
                    'Detection + tracking (MediaPipe)',
                    style: TextStyle(color: Colors.white70, fontSize: 14),
                  ),
                ),
                Switch(
                  value: _enableTracking,
                  activeTrackColor: Colors.blue,
                  onChanged: (value) {
                    setMenuState(() => _enableTracking = value);
                    _updateDetectorSettings(() => _enableTracking = value);
                  },
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Future<void> _initCamera() async {
    try {
      try {
        await _reinitDetector();
      } catch (e) {
        debugPrint('Detector init failed: $e');
        _handDetector = await _createDetector(false);
        _useCompiledModel = false;
      }
      if (mounted) setState(() => _isInitialized = true);

      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('No cameras available')),
          );
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
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error initializing camera: $e')),
        );
      }
    }
  }

  Future<void> _startControllerFor(CameraDescription camera) async {
    final controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup
          .yuv420, // prevents JPEG fallback on Android; ignored on desktop
    );

    await controller.initialize();

    if (!mounted) {
      await controller.dispose();
      return;
    }

    setState(() {
      _cameraController = controller;
      _sensorOrientation = controller.description.sensorOrientation;
      _isFrontCamera =
          controller.description.lensDirection == CameraLensDirection.front;
    });

    await controller.startImageStream(_processCameraImage);
    _isImageStreamStarted = true;
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
    setState(() {
      _isSwitchingCamera = true;
      _cameraController = null;
      _hands = [];
      _imageSize = null;
    });
    try {
      if (prev != null) {
        if (_isImageStreamStarted) {
          try {
            await prev.stopImageStream();
          } catch (_) {}
          _isImageStreamStarted = false;
        }
        await prev.dispose();
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

  DeviceOrientation _effectiveDeviceOrientation(BuildContext context) {
    final controller = _cameraController;
    if (controller != null) {
      return controller.value.deviceOrientation;
    }

    return MediaQuery.of(context).orientation == Orientation.portrait
        ? DeviceOrientation.portraitUp
        : DeviceOrientation.landscapeLeft;
  }

  Future<void> _processCameraImage(CameraImage image) async {
    if (_fpsCounter.tick() && mounted) {
      setState(() => _fps = _fpsCounter.fps);
      final n = _recentInferenceMs.length;
      final meanMs =
          n == 0 ? 0 : (_recentInferenceMs.reduce((a, b) => a + b) / n).round();
      final backend = _useCompiledModel ? 'compiledmodel' : 'xnnpack';
      // ignore: avoid_print
      print('[live-bench] backend=$backend '
          'cameraFps=$_fps detPerSec=$_detThisSec meanInferMs=$meanMs '
          'lastMs=$_detectionTimeMs hands=${_hands.length}');
      _recentInferenceMs.clear();
      _detThisSec = 0;
    }

    await _throttle.run(() async {
      try {
        if (_handDetector == null || !_isInitialized || !mounted) return;
        final startTime = DateTime.now();
        final sensor = _sensorOrientation;
        final CameraFrameRotation? rotation = sensor == null
            ? null
            : rotationForFrame(
                width: image.width,
                height: image.height,
                sensorOrientation: sensor,
                isFrontCamera: _isFrontCamera,
                deviceOrientation: _effectiveDeviceOrientation(context),
              );

        const int maxDim = 640;
        final Size size = detectionSize(
          width: image.width,
          height: image.height,
          rotation: rotation,
          maxDim: maxDim,
        );

        final List<Hand> hands = await _handDetector!.detectFromCameraImage(
          image,
          rotation: rotation,
          isBgra: Platform.isMacOS,
          maxDim: maxDim,
        );

        final endTime = DateTime.now();
        final detectionTime = endTime.difference(startTime).inMilliseconds;
        _recentInferenceMs.add(detectionTime);
        _detThisSec++;

        if (mounted) {
          setState(() {
            _hands = hands;
            _imageSize = size;
            _detectionTimeMs = detectionTime;
          });
        }
      } catch (_) {
        // Silently handle errors during processing to keep the stream alive.
      }
    });
  }

  @override
  void dispose() {
    _accelerometerSub?.cancel();
    if (_isImageStreamStarted) {
      _cameraController?.stopImageStream();
    }
    _cameraController?.dispose();
    _handDetector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized || _cameraController == null) {
      return Scaffold(
        appBar: AppBar(
          title: const Text('Live Hand Detection'),
        ),
        body: const Center(
          child: CircularProgressIndicator(),
        ),
      );
    }

    final cameraAspectRatio = _cameraController!.value.aspectRatio;
    final effectiveOrientation = _effectiveDeviceOrientation(context);
    final bool isPortrait =
        effectiveOrientation == DeviceOrientation.portraitUp ||
            effectiveOrientation == DeviceOrientation.portraitDown;

    final double displayAspectRatio =
        isPortrait ? 1.0 / cameraAspectRatio : cameraAspectRatio;

    final int turns = barQuarterTurns(_deviceOrientation);
    final bool mirrorOverlayHorizontally =
        (Platform.isAndroid && _isFrontCamera) || Platform.isWindows;

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          HandDetectionCameraOverlay(
            cameraPreview: CameraPreview(_cameraController!),
            displayAspectRatio: displayAspectRatio,
            mirrorHorizontally: mirrorOverlayHorizontally,
            hands: _hands,
            imageSize: _imageSize,
          ),
          _positionedTopBar(turns),
        ],
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
}

/// Aspect-fitted camera preview with a hand overlay painted on top. Mirrors
/// the structure of the package's still-image/camera painters; uses
/// [CameraHandOverlayPainter] for the live overlay (boxes, skeleton,
/// landmarks, gesture emoji).
class HandDetectionCameraOverlay extends StatelessWidget {
  final Widget cameraPreview;
  final double displayAspectRatio;
  final bool mirrorHorizontally;
  final List<Hand> hands;
  final Size? imageSize;

  const HandDetectionCameraOverlay({
    super.key,
    required this.cameraPreview,
    required this.displayAspectRatio,
    required this.mirrorHorizontally,
    required this.hands,
    required this.imageSize,
  });

  @override
  Widget build(BuildContext context) {
    return Center(
      child: AspectRatio(
        aspectRatio: displayAspectRatio,
        child: Stack(
          fit: StackFit.expand,
          children: [
            cameraPreview,
            if (imageSize != null)
              CustomPaint(
                painter: CameraHandOverlayPainter(
                  hands: hands,
                  imageSize: imageSize!,
                  mirrorHorizontally: mirrorHorizontally,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────── Video File Screen ────────────────────────────

class VideoFileScreen extends StatefulWidget {
  const VideoFileScreen({super.key});

  @override
  State<VideoFileScreen> createState() => _VideoFileScreenState();
}

class _VideoFileScreenState extends State<VideoFileScreen> {
  HandDetector? _detector;
  bool _isInitialized = false;
  bool _isProcessing = false;
  bool _cancelRequested = false;
  bool _useCompiledModel = true;
  bool _enableTracking = false;
  String? _errorMessage;
  String? _statusMessage;

  String? _inputPath;
  String? _outputPath;
  int _totalFrames = 0;
  int _processedFrames = 0;
  double _videoFps = 0;
  int _videoWidth = 0;
  int _videoHeight = 0;
  Duration _elapsed = Duration.zero;
  final Stopwatch _wallClock = Stopwatch();

  VideoPlayerController? _playerController;
  bool _playerReady = false;
  String? _playerError;

  bool _smoothingEnabled = true;
  final HandSmoother _smoother = HandSmoother(enabled: true);

  // Paint options, mirroring the Still Image screen. Style options (colors,
  // sizes, toggles) are read per frame; gesture recognition is captured when a
  // run starts (it is an init-time detector setting).
  bool _enableGestures = true;
  bool _showBoundingBoxes = true;
  bool _showSkeleton = true;
  bool _showLandmarks = true;
  bool _showHandedness = true;
  bool _showGestureLabels = true;

  Color _boundingBoxColor = const Color(0xFFFF9800);
  Color _landmarkColor = const Color(0xFFFF3D00);
  Color _skeletonColor = const Color(0xFF00E676);

  double _boundingBoxThickness = 2.0;
  double _landmarkSize = 3.0;
  double _skeletonThickness = 3.0;

  bool get _supportsInAppPlayer {
    if (kIsWeb) return true;
    return Platform.isAndroid || Platform.isIOS || Platform.isMacOS;
  }

  @override
  void initState() {
    super.initState();
    _initDetector();
  }

  Future<HandDetector> _createDetector(bool useCompiledModel) {
    return HandDetector.create(
      mode: HandMode.boxesAndLandmarks,
      landmarkModel: HandLandmarkModel.full,
      detectorConf: 0.6,
      maxDetections: 10,
      minLandmarkScore: 0.5,
      enableTracking: _enableTracking,
      performanceConfig: const PerformanceConfig.xnnpack(),
      enableGestures: _enableGestures,
      gestureMinConfidence: 0.5,
      useCompiledModel: useCompiledModel,
    );
  }

  Future<void> _initDetector() async {
    try {
      final detector = await _createDetector(_useCompiledModel);
      if (!mounted) {
        await detector.dispose();
        return;
      }
      setState(() {
        _detector = detector;
        _isInitialized = true;
      });
      return;
    } catch (e) {
      if (!_useCompiledModel) {
        if (mounted) {
          setState(() => _errorMessage = 'Failed to initialize detector: $e');
        }
        return;
      }
      debugPrint('CompiledModel init failed; falling back to XNNPACK: $e');
      _useCompiledModel = false;
    }
    try {
      final detector = await _createDetector(false);
      if (!mounted) {
        await detector.dispose();
        return;
      }
      setState(() {
        _detector = detector;
        _isInitialized = true;
      });
    } catch (e) {
      if (mounted) {
        setState(() => _errorMessage = 'Failed to initialize detector: $e');
      }
    }
  }

  Future<void> _reinitDetector() async {
    setState(() => _isInitialized = false);
    final old = _detector;
    _detector = null;
    await old?.dispose();
    await _initDetector();
  }

  Future<void> _toggleAccelerator() async {
    if (!_isInitialized || _isProcessing) return;
    _useCompiledModel = !_useCompiledModel;
    await _reinitDetector();
  }

  @override
  void dispose() {
    _cancelRequested = true;
    _detector?.dispose();
    _playerController?.dispose();
    super.dispose();
  }

  Future<void> _disposePlayer() async {
    final c = _playerController;
    _playerController = null;
    _playerReady = false;
    _playerError = null;
    await c?.dispose();
  }

  Future<void> _initPlayerForOutput(String path) async {
    await _disposePlayer();
    if (!_supportsInAppPlayer) return;
    final controller = VideoPlayerController.file(File(path));
    _playerController = controller;
    try {
      await controller.initialize();
      await controller.setLooping(true);
      if (!mounted) {
        await controller.dispose();
        _playerController = null;
        return;
      }
      setState(() => _playerReady = true);
      await controller.play();
    } catch (e) {
      if (!mounted) return;
      setState(() => _playerError = 'Could not load video: $e');
    }
  }

  Future<void> _pickVideo() async {
    const typeGroup = XTypeGroup(
      label: 'Videos',
      extensions: ['mp4', 'mov', 'm4v'],
    );
    final XFile? file = await openFile(acceptedTypeGroups: [typeGroup]);
    if (file == null) return;
    await _processVideo(file.path);
  }

  Future<void> _processVideo(String path) async {
    final detector = _detector;
    if (detector == null) return;

    final inputFile = File(path);
    if (!await inputFile.exists()) {
      setState(() => _errorMessage = 'File does not exist: $path');
      return;
    }

    final cap = cv.VideoCapture.fromFile(path);
    if (!cap.isOpened) {
      cap.release();
      String hint = '';
      if (Platform.isLinux) {
        hint = '\n\nLinux requires GStreamer plugins. Try:\n'
            '  sudo apt install gstreamer1.0-libav '
            'gstreamer1.0-plugins-good gstreamer1.0-plugins-bad';
      }
      setState(
        () => _errorMessage =
            'Could not open video.\nFormat may not be supported by the OS '
                'video backend.$hint',
      );
      return;
    }

    final fps = cap.get(cv.CAP_PROP_FPS);
    final width = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
    final height = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();
    final total = cap.get(cv.CAP_PROP_FRAME_COUNT).toInt();

    final docs = await getApplicationDocumentsDirectory();
    final outName = 'hand_${DateTime.now().millisecondsSinceEpoch}.mp4';
    final outPath = '${docs.path}/$outName';

    final writer = cv.VideoWriter.fromFile(outPath, 'avc1', fps, (
      width,
      height,
    ));
    if (!writer.isOpened) {
      cap.release();
      setState(
        () => _errorMessage =
            'Could not open writer for $outPath. The "avc1" (H.264) codec '
                'may not be available on this OS backend.',
      );
      return;
    }

    if (!mounted) {
      cap.release();
      writer.release();
      return;
    }
    await _disposePlayer();
    setState(() {
      _inputPath = path;
      _outputPath = outPath;
      _videoFps = fps;
      _videoWidth = width;
      _videoHeight = height;
      _totalFrames = total;
      _processedFrames = 0;
      _isProcessing = true;
      _cancelRequested = false;
      _errorMessage = null;
      _statusMessage = 'Processing...';
      _elapsed = Duration.zero;
    });
    _wallClock
      ..reset()
      ..start();

    cv.Mat? frame;
    _smoother.reset();
    // New video: drop any tracked ROI carried over from a previous run.
    await detector.resetTracking();
    try {
      int idx = 0;
      while (mounted && !_cancelRequested) {
        final result = cap.read(m: frame);
        final ok = result.$1;
        frame = result.$2;
        if (!ok || frame.isEmpty) break;

        final List<Hand> raw = await detector.detectFromMat(frame);
        final double tSec = fps > 0 ? idx / fps : idx / 30.0;
        final List<Hand> hands = _smoother.apply(raw, tSec);
        _drawHandsOnMat(frame, hands);
        writer.write(frame);

        idx++;
        if (idx % 4 == 0) {
          if (!mounted) break;
          setState(() {
            _processedFrames = idx;
            _elapsed = _wallClock.elapsed;
          });
          await Future<void>.delayed(Duration.zero);
        }
      }
      if (mounted) {
        setState(() {
          _processedFrames = idx;
          _elapsed = _wallClock.elapsed;
          _statusMessage = _cancelRequested
              ? 'Cancelled after $idx frames.'
              : 'Done. Wrote $idx frames to:\n$outPath';
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() => _errorMessage = 'Error during processing: $e');
      }
    } finally {
      _wallClock.stop();
      cap.release();
      writer.release();
      frame?.dispose();
      if (mounted) setState(() => _isProcessing = false);
      if (mounted && !_cancelRequested && _outputPath != null) {
        await _initPlayerForOutput(_outputPath!);
      }
    }
  }

  /// Converts a Flutter [Color] to an OpenCV BGR scalar (alpha ignored).
  cv.Scalar _bgr(Color c) => cv.Scalar(
        (c.b * 255).roundToDouble(),
        (c.g * 255).roundToDouble(),
        (c.r * 255).roundToDouble(),
      );

  /// Draws the enabled overlays onto [mat] with OpenCV, mirroring what
  /// [HandDetectionsPainter] draws on screen for the Still Image mode.
  void _drawHandsOnMat(cv.Mat mat, List<Hand> hands) {
    if (hands.isEmpty) return;
    final black = cv.Scalar(0, 0, 0);

    final w = mat.cols;
    final h = mat.rows;

    for (final hand in hands) {
      if (_showSkeleton && hand.hasLandmarks) {
        final skeletonColor = _bgr(_skeletonColor);
        for (final connection in handLandmarkConnections) {
          final a = hand.getLandmark(connection[0]);
          final b = hand.getLandmark(connection[1]);
          if (a == null || b == null) continue;
          if (a.visibility <= 0.5 || b.visibility <= 0.5) continue;
          cv.line(
            mat,
            cv.Point(a.x.toInt(), a.y.toInt()),
            cv.Point(b.x.toInt(), b.y.toInt()),
            skeletonColor,
            thickness: math.max(1, _skeletonThickness.round()),
          );
        }
      }

      if (_showLandmarks && hand.hasLandmarks) {
        final lmColor = _bgr(_landmarkColor);
        for (final lm in hand.landmarks) {
          if (lm.visibility <= 0.5) continue;
          cv.circle(
            mat,
            cv.Point(lm.x.toInt(), lm.y.toInt()),
            math.max(1, _landmarkSize.round()),
            lmColor,
            thickness: -1,
          );
        }
      }

      if (_showBoundingBoxes) {
        final boxColor = _bgr(_boundingBoxColor);
        final bb = hand.boundingBox;
        final l = bb.left.toInt().clamp(0, w - 1);
        final t = bb.top.toInt().clamp(0, h - 1);
        final r = bb.right.toInt().clamp(0, w - 1);
        final b = bb.bottom.toInt().clamp(0, h - 1);
        cv.rectangle(
          mat,
          cv.Rect(l, t, (r - l).clamp(1, w), (b - t).clamp(1, h)),
          boxColor,
          thickness: math.max(1, _boundingBoxThickness.round()),
        );

        final parts = <String>['${(hand.score * 100).toStringAsFixed(0)}%'];
        if (_showHandedness && hand.handedness != null) {
          parts.add(hand.handedness == Handedness.right ? 'R' : 'L');
        }
        if (_showGestureLabels &&
            hand.gesture != null &&
            hand.gesture!.type != GestureType.unknown) {
          parts.add(_gestureLabel(hand.gesture!.type));
        }
        final label = parts.join('  ');
        final (sz, _) = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2);
        final labelTop = (t - sz.height - 8).clamp(0, h - 1);
        final labelW = (sz.width + 8).clamp(1, w - l);
        final labelH = (sz.height + 8).clamp(1, h - labelTop);
        cv.rectangle(
          mat,
          cv.Rect(l, labelTop, labelW, labelH),
          boxColor,
          thickness: -1,
        );
        cv.putText(
          mat,
          label,
          cv.Point(l + 4, labelTop + sz.height + 2),
          cv.FONT_HERSHEY_SIMPLEX,
          0.6,
          black,
          thickness: 2,
        );
      }
    }
  }

  Future<void> _openOutputFile() async {
    final path = _outputPath;
    if (path == null) return;
    try {
      if (Platform.isMacOS) {
        await Process.run('open', [path]);
      } else if (Platform.isLinux) {
        await Process.run('xdg-open', [path]);
      } else if (Platform.isWindows) {
        await Process.run('cmd', ['/c', 'start', '', path]);
      } else if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Saved to: $path')));
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(
          context,
        ).showSnackBar(SnackBar(content: Text('Could not open: $e')));
      }
    }
  }

  void _showVideoSettings() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => DraggableScrollableSheet(
        initialChildSize: 0.6,
        minChildSize: 0.3,
        maxChildSize: 0.9,
        builder: (context, scrollController) => StatefulBuilder(
          builder: (context, setSheetState) {
            void updateState(VoidCallback fn) {
              fn();
              setSheetState(() {});
              setState(() {});
            }

            Widget cb(String label, bool v, void Function(bool) set) =>
                CompactCheckbox(
                    label: label,
                    value: v,
                    onChanged: (x) => updateState(() => set(x ?? false)));
            Widget col(String label, Color c, void Function(Color) set) =>
                _ColorPickerButton(
                    label: label,
                    color: c,
                    onColorChanged: (x) => updateState(() => set(x)));
            Widget sl(String label, double v, double mn, double mx,
                    void Function(double) set) =>
                CompactSlider(
                    label: label,
                    value: v,
                    min: mn,
                    max: mx,
                    onChanged: (x) => updateState(() => set(x)));

            return Container(
              decoration: const BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.vertical(top: Radius.circular(16)),
              ),
              child: Column(
                children: [
                  Container(
                    margin: const EdgeInsets.symmetric(vertical: 8),
                    width: 40,
                    height: 4,
                    decoration: BoxDecoration(
                      color: Colors.grey[300],
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  Expanded(
                    child: ListView(
                      controller: scrollController,
                      padding: const EdgeInsets.symmetric(horizontal: 16),
                      children: [
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 4),
                          child: Text(
                            'Gesture recognition applies when processing '
                            'starts; styles apply to the remaining frames.',
                            style: TextStyle(fontSize: 12, color: Colors.grey),
                          ),
                        ),
                        ExpansionTile(
                          title: const Text('Display Options',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          initiallyExpanded: true,
                          children: [
                            Wrap(
                              spacing: 8,
                              runSpacing: 4,
                              children: [
                                cb('Bounding Boxes', _showBoundingBoxes,
                                    (v) => _showBoundingBoxes = v),
                                cb('Skeleton', _showSkeleton,
                                    (v) => _showSkeleton = v),
                                cb('Landmarks', _showLandmarks,
                                    (v) => _showLandmarks = v),
                                cb('Handedness', _showHandedness,
                                    (v) => _showHandedness = v),
                                cb('Gesture Labels', _showGestureLabels,
                                    (v) => _showGestureLabels = v),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Detection',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          children: [
                            SwitchListTile(
                              dense: true,
                              contentPadding: EdgeInsets.zero,
                              title: const Text('Gesture recognition'),
                              subtitle: const Text(
                                  'Recompute on the next processing run'),
                              value: _enableGestures,
                              onChanged: _isProcessing
                                  ? null
                                  : (v) {
                                      updateState(() => _enableGestures = v);
                                      _reinitDetector();
                                    },
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Colors',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          children: [
                            Wrap(
                              spacing: 6,
                              runSpacing: 6,
                              children: [
                                col('BBox', _boundingBoxColor,
                                    (c) => _boundingBoxColor = c),
                                col('Landmarks', _landmarkColor,
                                    (c) => _landmarkColor = c),
                                col('Skeleton', _skeletonColor,
                                    (c) => _skeletonColor = c),
                              ],
                            ),
                            const SizedBox(height: 8),
                          ],
                        ),
                        ExpansionTile(
                          title: const Text('Sizes',
                              style: TextStyle(fontWeight: FontWeight.bold)),
                          children: [
                            sl('BBox', _boundingBoxThickness, 0.5, 10.0,
                                (v) => _boundingBoxThickness = v),
                            sl('Landmark', _landmarkSize, 0.5, 15.0,
                                (v) => _landmarkSize = v),
                            sl('Skeleton', _skeletonThickness, 0.5, 10.0,
                                (v) => _skeletonThickness = v),
                            const SizedBox(height: 8),
                          ],
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Video File - Hand Detection'),
        actions: [
          TextButton(
            onPressed:
                _isInitialized && !_isProcessing ? _toggleAccelerator : null,
            child: Text(
              _useCompiledModel ? 'CM' : 'XNN',
              style: const TextStyle(
                color: Colors.amber,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          IconButton(
            onPressed: _showVideoSettings,
            icon: const Icon(Icons.tune),
            tooltip: 'Settings',
          ),
        ],
      ),
      body: _buildBody(),
      floatingActionButton: _isInitialized && !_isProcessing
          ? FloatingActionButton.extended(
              onPressed: _pickVideo,
              icon: const Icon(Icons.video_file),
              label: const Text('Pick Video'),
            )
          : (_isProcessing
              ? FloatingActionButton.extended(
                  onPressed: () => setState(() => _cancelRequested = true),
                  icon: const Icon(Icons.cancel),
                  label: const Text('Cancel'),
                  backgroundColor: Colors.red,
                )
              : null),
    );
  }

  Widget _buildBody() {
    if (!_isInitialized && _errorMessage == null) {
      return const _ScrollableCentered(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Initializing detector...'),
          ],
        ),
      );
    }

    final progress = (_totalFrames > 0)
        ? (_processedFrames / _totalFrames).clamp(0.0, 1.0)
        : 0.0;
    final processedFps = (_elapsed.inMilliseconds > 0)
        ? _processedFrames * 1000.0 / _elapsed.inMilliseconds
        : 0.0;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          if (_errorMessage != null)
            Card(
              color: Colors.red[50],
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    const Icon(Icons.error_outline, color: Colors.red),
                    const SizedBox(width: 12),
                    Expanded(child: Text(_errorMessage!)),
                  ],
                ),
              ),
            ),
          if (_inputPath != null) ...[
            const SizedBox(height: 8),
            _infoRow('Input', _inputPath!),
            if (_videoWidth > 0)
              _infoRow(
                'Source',
                '$_videoWidth×$_videoHeight @ '
                    '${_videoFps.toStringAsFixed(2)} fps · '
                    '$_totalFrames frames',
              ),
          ],
          if (!_isProcessing)
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              dense: true,
              title: const Text('Smoothing (One-Euro filter)'),
              subtitle: Text(
                _smoothingEnabled
                    ? 'On: landmarks filtered across frames'
                    : 'Off: raw per-frame detections',
              ),
              value: _smoothingEnabled,
              onChanged: (v) {
                setState(() {
                  _smoothingEnabled = v;
                  _smoother.enabled = v;
                  _smoother.reset();
                });
              },
            ),
          if (!_isProcessing)
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              dense: true,
              title: const Text('Detection + tracking (MediaPipe)'),
              subtitle: Text(
                _enableTracking
                    ? 'On: hands followed frame-to-frame via landmark ROI'
                    : 'Off: palm detector re-runs on every frame',
              ),
              value: _enableTracking,
              onChanged: (v) {
                setState(() => _enableTracking = v);
                _reinitDetector();
              },
            ),
          if (!_isProcessing && _inputPath != null) ...[
            const SizedBox(height: 8),
            Align(
              alignment: Alignment.centerLeft,
              child: OutlinedButton.icon(
                onPressed: () => _processVideo(_inputPath!),
                icon: const Icon(Icons.refresh),
                label: const Text('Re-run with current settings'),
              ),
            ),
          ],
          const SizedBox(height: 16),
          if (_isProcessing) ...[
            LinearProgressIndicator(value: _totalFrames > 0 ? progress : null),
            const SizedBox(height: 8),
            Text(
              'Frame $_processedFrames / $_totalFrames · '
              '${(progress * 100).toStringAsFixed(1)}% · '
              '${processedFps.toStringAsFixed(1)} fps · '
              'elapsed ${_formatDuration(_elapsed)}',
              style: const TextStyle(
                fontFeatures: [FontFeature.tabularFigures()],
              ),
            ),
          ] else if (_outputPath != null && _statusMessage != null)
            VideoResultCard(
              statusMessage: _statusMessage!,
              summary: 'Total time: ${_formatDuration(_elapsed)} '
                  '(${processedFps.toStringAsFixed(1)} fps avg)',
              preview: _buildOutputPreview(),
              onOpenOutput:
                  (Platform.isMacOS || Platform.isLinux || Platform.isWindows)
                      ? _openOutputFile
                      : null,
            )
          else
            Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const SizedBox(height: 32),
                  Icon(
                    Icons.movie_creation_outlined,
                    size: 96,
                    color: Colors.grey[400],
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Pick an MP4 to run hand detection on every frame.\n'
                    'Output is written to the app documents directory.',
                    textAlign: TextAlign.center,
                    style: TextStyle(color: Colors.grey[700]),
                  ),
                ],
              ),
            ),
        ],
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 70,
            child: Text(
              '$label:',
              style: const TextStyle(fontWeight: FontWeight.w600),
            ),
          ),
          Expanded(child: SelectableText(value)),
        ],
      ),
    );
  }

  String _formatDuration(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    if (d.inHours > 0) {
      return '${d.inHours}:$m:$s';
    }
    return '$m:$s';
  }

  Widget _buildOutputPreview() {
    if (!_supportsInAppPlayer) return const SizedBox.shrink();
    if (_playerError != null) {
      return Text(_playerError!, style: const TextStyle(color: Colors.red));
    }
    final controller = _playerController;
    if (controller == null || !_playerReady) {
      return const SizedBox(
        height: 64,
        child: Center(
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
              SizedBox(width: 12),
              Flexible(child: Text('Loading preview...')),
            ],
          ),
        ),
      );
    }
    return _OutputVideoPlayer(controller: controller);
  }
}

// ─────────────────────────── Video Result Card ────────────────────────────

/// Result card shown after a video finishes processing.
class VideoResultCard extends StatelessWidget {
  final String statusMessage;
  final String summary;
  final Widget preview;
  final VoidCallback? onOpenOutput;

  const VideoResultCard({
    super.key,
    required this.statusMessage,
    required this.summary,
    required this.preview,
    this.onOpenOutput,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.green),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    statusMessage,
                    style: const TextStyle(fontWeight: FontWeight.w500),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(summary),
            const SizedBox(height: 12),
            preview,
            if (onOpenOutput != null) ...[
              const SizedBox(height: 12),
              Align(
                alignment: Alignment.centerLeft,
                child: ElevatedButton.icon(
                  onPressed: onOpenOutput,
                  icon: const Icon(Icons.play_circle_outline),
                  label: const Text('Open output video'),
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }
}

/// Layout chrome for the output video preview.
class VideoPlayerChrome extends StatelessWidget {
  final double aspectRatio;
  final Widget video;
  final Widget progress;
  final bool isPlaying;
  final String positionLabel;
  final VoidCallback onTogglePlay;

  const VideoPlayerChrome({
    super.key,
    required this.aspectRatio,
    required this.video,
    required this.progress,
    required this.isPlaying,
    required this.positionLabel,
    required this.onTogglePlay,
  });

  @override
  Widget build(BuildContext context) {
    final double maxPreviewHeight =
        math.max(120.0, MediaQuery.sizeOf(context).height * 0.45);
    return Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: ConstrainedBox(
            constraints: BoxConstraints(maxHeight: maxPreviewHeight),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: AspectRatio(
                aspectRatio: aspectRatio,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    Container(color: Colors.black),
                    video,
                  ],
                ),
              ),
            ),
          ),
        ),
        const SizedBox(height: 8),
        LayoutBuilder(
          builder: (context, constraints) {
            final bool showTime = constraints.maxWidth >= 180;
            return Row(
              children: [
                IconButton(
                  icon: Icon(isPlaying ? Icons.pause : Icons.play_arrow),
                  onPressed: onTogglePlay,
                ),
                Expanded(child: progress),
                if (showTime) ...[
                  const SizedBox(width: 8),
                  Text(
                    positionLabel,
                    style: const TextStyle(
                      fontFeatures: [FontFeature.tabularFigures()],
                    ),
                  ),
                ],
              ],
            );
          },
        ),
      ],
    );
  }
}

// ─────────────────────────── Output Video Player ──────────────────────────

class _OutputVideoPlayer extends StatefulWidget {
  final VideoPlayerController controller;
  const _OutputVideoPlayer({required this.controller});

  @override
  State<_OutputVideoPlayer> createState() => _OutputVideoPlayerState();
}

class _OutputVideoPlayerState extends State<_OutputVideoPlayer> {
  void _onTick() {
    if (mounted) setState(() {});
  }

  @override
  void initState() {
    super.initState();
    widget.controller.addListener(_onTick);
  }

  @override
  void didUpdateWidget(covariant _OutputVideoPlayer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.controller != widget.controller) {
      oldWidget.controller.removeListener(_onTick);
      widget.controller.addListener(_onTick);
    }
  }

  @override
  void dispose() {
    widget.controller.removeListener(_onTick);
    super.dispose();
  }

  String _fmt(Duration d) {
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }

  @override
  Widget build(BuildContext context) {
    final c = widget.controller;
    final value = c.value;
    return VideoPlayerChrome(
      aspectRatio: value.aspectRatio == 0 ? 16 / 9 : value.aspectRatio,
      video: VideoPlayer(c),
      progress: VideoProgressIndicator(
        c,
        allowScrubbing: true,
        padding: const EdgeInsets.symmetric(vertical: 12),
      ),
      isPlaying: value.isPlaying,
      positionLabel: '${_fmt(value.position)} / ${_fmt(value.duration)}',
      onTogglePlay: () {
        if (value.isPlaying) {
          c.pause();
        } else {
          c.play();
        }
      },
    );
  }
}

// ─────────────────────────── Hand Smoother ────────────────────────────────

/// Per-track One-Euro temporal smoothing of hand landmarks across video
/// frames. Matches detections to tracks by bounding-box IoU, then filters each
/// hand's 21 landmark x/y through per-track [OneEuroFilter]s to remove jitter.
/// Tracks are dropped after a few missed frames.
class HandSmoother {
  bool enabled;
  static const int _maxMissed = 5;
  static const double _minIou = 0.2;
  final List<_HandTrack> _tracks = [];

  HandSmoother({this.enabled = true});

  void reset() => _tracks.clear();

  List<Hand> apply(List<Hand> hands, double tSec) {
    if (!enabled || hands.isEmpty) {
      if (!enabled) _tracks.clear();
      return hands;
    }

    final unmatched = List<int>.generate(_tracks.length, (i) => i);
    final matchedTrack = List<int?>.filled(hands.length, null);

    for (int p = 0; p < hands.length; p++) {
      double bestIou = _minIou;
      int bestT = -1;
      for (final t in unmatched) {
        if (!_tracks[t].hasBox) continue;
        final iou = _iou(hands[p], _tracks[t]);
        if (iou > bestIou) {
          bestIou = iou;
          bestT = t;
        }
      }
      if (bestT >= 0) {
        matchedTrack[p] = bestT;
        unmatched.remove(bestT);
      }
    }

    final out = <Hand>[];
    for (int p = 0; p < hands.length; p++) {
      _HandTrack track;
      if (matchedTrack[p] != null) {
        track = _tracks[matchedTrack[p]!];
        track.missedFrames = 0;
      } else {
        track = _HandTrack();
        _tracks.add(track);
      }
      final bb = hands[p].boundingBox;
      track.lastLeft = bb.left;
      track.lastTop = bb.top;
      track.lastRight = bb.right;
      track.lastBottom = bb.bottom;
      track.hasBox = true;
      out.add(_smoothHand(hands[p], track, tSec));
    }

    for (final t in unmatched) {
      _tracks[t].missedFrames++;
    }
    _tracks.removeWhere((t) => t.missedFrames > _maxMissed);

    return out;
  }

  Hand _smoothHand(Hand hand, _HandTrack track, double tSec) {
    if (hand.landmarks.isEmpty) return hand;
    final smoothed = <HandLandmark>[];
    for (int i = 0; i < hand.landmarks.length; i++) {
      final lm = hand.landmarks[i];
      var fs = track.filters[i];
      if (fs == null) {
        fs = [
          OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
          OneEuroFilter(minCutoff: 1.0, beta: 0.1, dCutoff: 1.0),
        ];
        track.filters[i] = fs;
      }
      smoothed.add(HandLandmark(
        type: lm.type,
        x: fs[0].filter(lm.x, tSec),
        y: fs[1].filter(lm.y, tSec),
        z: lm.z,
        visibility: lm.visibility,
      ));
    }
    return Hand(
      boundingBox: hand.boundingBox,
      score: hand.score,
      landmarks: smoothed,
      imageWidth: hand.imageWidth,
      imageHeight: hand.imageHeight,
      handedness: hand.handedness,
      rotation: hand.rotation,
      rotatedCenterX: hand.rotatedCenterX,
      rotatedCenterY: hand.rotatedCenterY,
      rotatedSize: hand.rotatedSize,
      gesture: hand.gesture,
    );
  }

  double _iou(Hand a, _HandTrack b) {
    final box = a.boundingBox;
    final l = math.max(box.left, b.lastLeft);
    final t = math.max(box.top, b.lastTop);
    final r = math.min(box.right, b.lastRight);
    final bo = math.min(box.bottom, b.lastBottom);
    final iw = math.max(0.0, r - l);
    final ih = math.max(0.0, bo - t);
    final inter = iw * ih;
    final aa = math.max(0.0, box.right - box.left) *
        math.max(0.0, box.bottom - box.top);
    final bb = math.max(0.0, b.lastRight - b.lastLeft) *
        math.max(0.0, b.lastBottom - b.lastTop);
    final union = aa + bb - inter;
    if (union <= 0) return 0;
    return inter / union;
  }
}

class _HandTrack {
  final Map<int, List<OneEuroFilter>> filters = {};
  double lastLeft = 0, lastTop = 0, lastRight = 0, lastBottom = 0;
  bool hasBox = false;
  int missedFrames = 0;
}
