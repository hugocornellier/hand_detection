import 'dart:async';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';
import 'package:opencv_dart/opencv_dart.dart' as cv;
import 'package:meta/meta.dart';
import 'package:flutter_litert/flutter_litert.dart';
import 'image_utils.dart';
import 'types.dart';

/// SSD Anchor configuration options for palm detection.
///
/// Mirrors the Python SSDAnchorOptions namedtuple. Used internally for
/// generating anchor boxes in the SSD-style palm detection model.
class SSDAnchorOptions {
  /// Number of feature map layers (typically 4 for palm detection).
  final int numLayers;

  /// Minimum anchor scale (0.0-1.0).
  final double minScale;

  /// Maximum anchor scale (0.0-1.0).
  final double maxScale;

  /// Input image height in pixels.
  final int inputSizeHeight;

  /// Input image width in pixels.
  final int inputSizeWidth;

  /// X offset for anchor centers (typically 0.5).
  final double anchorOffsetX;

  /// Y offset for anchor centers (typically 0.5).
  final double anchorOffsetY;

  /// Feature map strides for each layer.
  final List<int> strides;

  /// Aspect ratios for anchor boxes.
  final List<double> aspectRatios;

  /// Whether to reduce boxes in the lowest layer.
  final bool reduceBoxesInLowestLayer;

  /// Interpolated scale aspect ratio (0 to disable).
  final double interpolatedScaleAspectRatio;

  /// Whether to use fixed anchor size (1x1).
  final bool fixedAnchorSize;

  /// Creates SSD anchor options.
  const SSDAnchorOptions({
    required this.numLayers,
    required this.minScale,
    required this.maxScale,
    required this.inputSizeHeight,
    required this.inputSizeWidth,
    required this.anchorOffsetX,
    required this.anchorOffsetY,
    required this.strides,
    required this.aspectRatios,
    required this.reduceBoxesInLowestLayer,
    required this.interpolatedScaleAspectRatio,
    required this.fixedAnchorSize,
  });
}

/// A detected palm with rotation rectangle parameters.
///
/// Used to crop and rotate hand regions for landmark extraction.
class PalmDetection {
  /// Size of the square rotation rectangle (normalized).
  final double sqnRrSize;

  /// Rotation angle in radians.
  final double rotation;

  /// Center X coordinate (normalized 0-1).
  final double sqnRrCenterX;

  /// Center Y coordinate (normalized 0-1).
  final double sqnRrCenterY;

  /// Detection confidence score (0.0 to 1.0).
  final double score;

  /// Creates a palm detection result.
  const PalmDetection({
    required this.sqnRrSize,
    required this.rotation,
    required this.sqnRrCenterX,
    required this.sqnRrCenterY,
    required this.score,
  });
}

/// Axis-aligned bounding box for NMS IoU computation.
///
/// Precomputed from PalmDetection to avoid redundant conversions in O(n²) NMS loop.
class _AABB {
  final double left;
  final double top;
  final double right;
  final double bottom;

  const _AABB(this.left, this.top, this.right, this.bottom);
}

/// SSD-based palm detector for Stage 1 of the hand detection pipeline.
///
/// Detects palm locations in images using a Single Shot Detector (SSD) architecture
/// with anchor-based decoding. Returns rotation rectangles suitable for cropping
/// hand regions for landmark extraction.
///
/// This is a direct port of the Python PalmDetection class.
class PalmDetector {
  IsolateInterpreter? _iso;
  Interpreter? _interpreter;
  bool _isInitialized = false;
  Delegate? _delegate;

  /// Input dimensions (192x192 for palm detection model).
  late int _inH;
  late int _inW;

  /// Pre-generated SSD anchors.
  late List<List<double>> _anchors;

  /// Preprocessing state - matches Python's calculation.
  /// These use original image dimensions like Python does.
  int _imageHeight = 0;
  int _imageWidth = 0;

  /// square_standard_size = max(image_height, image_width) - Python palm_detection.py line 299
  int _squareStandardSize = 0;

  /// square_padding_half_size = abs(image_height - image_width) // 2 - Python palm_detection.py line 300
  int _squarePaddingHalfSize = 0;

  /// Score threshold for detection filtering.
  final double scoreThreshold;

  /// Pre-allocated buffers.
  Float32List? _inputBuffer;
  List<List<List<double>>>? _outputBoxes; // [1, 2016, 18]
  List<List<List<double>>>? _outputScores; // [1, 2016, 1]

  /// Creates a palm detector with the specified score threshold.
  PalmDetector({this.scoreThreshold = 0.45});

  /// Calculates scale for anchor generation.
  static double _calculateScale(
      double minScale, double maxScale, int strideIndex, int numStrides) {
    if (numStrides == 1) {
      return (minScale + maxScale) / 2;
    } else {
      return minScale + (maxScale - minScale) * strideIndex / (numStrides - 1);
    }
  }

  /// Generates SSD anchors based on the given options.
  ///
  /// This is a direct port of the Python generate_anchors function.
  static List<List<double>> generateAnchors(SSDAnchorOptions options) {
    final anchors = <List<double>>[];
    int layerId = 0;
    final nStrides = options.strides.length;

    while (layerId < nStrides) {
      final anchorHeight = <double>[];
      final anchorWidth = <double>[];
      final aspectRatios = <double>[];
      final scales = <double>[];
      int lastSameStrideLayer = layerId;

      while (lastSameStrideLayer < nStrides &&
          options.strides[lastSameStrideLayer] == options.strides[layerId]) {
        final scale = _calculateScale(
            options.minScale, options.maxScale, lastSameStrideLayer, nStrides);

        if (lastSameStrideLayer == 0 && options.reduceBoxesInLowestLayer) {
          aspectRatios.addAll([1.0, 2.0, 0.5]);
          scales.addAll([0.1, scale, scale]);
        } else {
          aspectRatios.addAll(options.aspectRatios);
          for (int i = 0; i < options.aspectRatios.length; i++) {
            scales.add(scale);
          }
          if (options.interpolatedScaleAspectRatio > 0) {
            double scaleNext;
            if (lastSameStrideLayer == nStrides - 1) {
              scaleNext = 1.0;
            } else {
              scaleNext = _calculateScale(options.minScale, options.maxScale,
                  lastSameStrideLayer + 1, nStrides);
            }
            scales.add(math.sqrt(scale * scaleNext));
            aspectRatios.add(options.interpolatedScaleAspectRatio);
          }
        }
        lastSameStrideLayer++;
      }

      for (int i = 0; i < aspectRatios.length; i++) {
        final ratioSqrt = math.sqrt(aspectRatios[i]);
        anchorHeight.add(scales[i] / ratioSqrt);
        anchorWidth.add(scales[i] * ratioSqrt);
      }

      final stride = options.strides[layerId];
      final featureMapHeight = (options.inputSizeHeight / stride).ceil();
      final featureMapWidth = (options.inputSizeWidth / stride).ceil();

      for (int y = 0; y < featureMapHeight; y++) {
        for (int x = 0; x < featureMapWidth; x++) {
          for (int anchorId = 0; anchorId < anchorHeight.length; anchorId++) {
            final xCenter = (x + options.anchorOffsetX) / featureMapWidth;
            final yCenter = (y + options.anchorOffsetY) / featureMapHeight;

            List<double> newAnchor;
            if (options.fixedAnchorSize) {
              newAnchor = [xCenter, yCenter, 1.0, 1.0];
            } else {
              newAnchor = [
                xCenter,
                yCenter,
                anchorWidth[anchorId],
                anchorHeight[anchorId]
              ];
            }
            anchors.add(newAnchor);
          }
        }
      }

      layerId = lastSameStrideLayer;
    }

    return anchors;
  }

  /// Normalizes angle to range [-pi, pi].
  static double normalizeRadians(double angle) {
    return angle - 2 * math.pi * ((angle + math.pi) / (2 * math.pi)).floor();
  }

  /// Initializes the palm detector by loading the TFLite model.
  Future<void> initialize({PerformanceConfig? performanceConfig}) async {
    const String assetPath =
        'packages/hand_detection_tflite/assets/models/hand_detection.tflite';

    if (_isInitialized) await dispose();

    final options = _createInterpreterOptions(performanceConfig);
    final interpreter =
        await Interpreter.fromAsset(assetPath, options: options);
    _interpreter = interpreter;
    interpreter.allocateTensors();

    // Get input shape
    final inTensor = interpreter.getInputTensor(0);
    final inShape = inTensor.shape;
    _inH = inShape[1]; // Should be 192
    _inW = inShape[2]; // Should be 192

    // Generate anchors for palm detection
    final anchorOptions = SSDAnchorOptions(
      numLayers: 4,
      minScale: 0.1484375,
      maxScale: 0.75,
      inputSizeHeight: _inH,
      inputSizeWidth: _inW,
      anchorOffsetX: 0.5,
      anchorOffsetY: 0.5,
      strides: [8, 16, 16, 16],
      aspectRatios: [1.0],
      reduceBoxesInLowestLayer: false,
      interpolatedScaleAspectRatio: 1.0,
      fixedAnchorSize: true,
    );
    _anchors = generateAnchors(anchorOptions);

    // Pre-allocate output buffers
    // Output 0: [1, 2016, 18] - box regressors
    // Output 1: [1, 2016, 1] - classification scores
    final numAnchors = _anchors.length;
    _outputBoxes = List.generate(
      1,
      (_) => List.generate(
        numAnchors,
        (_) => List<double>.filled(18, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );
    _outputScores = List.generate(
      1,
      (_) => List.generate(
        numAnchors,
        (_) => List<double>.filled(1, 0.0, growable: false),
        growable: false,
      ),
      growable: false,
    );

    if (_delegate == null) {
      _iso = await IsolateInterpreter.create(address: interpreter.address);
    }
    _isInitialized = true;
  }

  InterpreterOptions _createInterpreterOptions(PerformanceConfig? config) {
    final options = InterpreterOptions();

    _delegate?.delete();
    _delegate = null;

    if (config == null || config.mode == PerformanceMode.disabled) {
      return options;
    }

    final threadCount = config.numThreads?.clamp(0, 8) ??
        math.min(4, Platform.numberOfProcessors);

    options.threads = threadCount;

    if (config.mode == PerformanceMode.xnnpack ||
        config.mode == PerformanceMode.auto) {
      try {
        final xnnpackDelegate = XNNPackDelegate(
          options: XNNPackDelegateOptions(numThreads: threadCount),
        );
        options.addDelegate(xnnpackDelegate);
        _delegate = xnnpackDelegate;
      } catch (e) {
        // Fallback to CPU
      }
    }

    return options;
  }

  /// Returns true if the detector has been initialized.
  bool get isInitialized => _isInitialized;

  /// Disposes the detector and releases resources.
  Future<void> dispose() async {
    _iso?.close();
    _iso = null;
    _interpreter?.close();
    _interpreter = null;
    _delegate?.delete();
    _delegate = null;
    _inputBuffer = null;
    _outputBoxes = null;
    _outputScores = null;
    _isInitialized = false;
  }

  /// Detects palms in the given image.
  ///
  /// Returns a list of [PalmDetection] objects containing rotation rectangle
  /// parameters for each detected palm.
  Future<List<PalmDetection>> detectOnMat(cv.Mat image) async {
    if (!_isInitialized || _interpreter == null) {
      throw StateError('PalmDetector not initialized.');
    }

    _imageHeight = image.rows;
    _imageWidth = image.cols;

    // Calculate square padding info from original image dimensions (matches Python exactly)
    // Python palm_detection.py lines 299-300:
    // self.square_standard_size = max(image_height, image_width)
    // self.square_padding_half_size = abs(image_height - image_width) // 2
    _squareStandardSize = math.max(_imageHeight, _imageWidth);
    _squarePaddingHalfSize = (_imageHeight - _imageWidth).abs() ~/ 2;

    // Use keep_aspect_resize_and_pad to match Python implementation
    final (paddedImage, resizedImage) = ImageUtils.keepAspectResizeAndPad(
      image,
      _inW,
      _inH,
    );

    // Convert to float32 tensor and normalize (BGR -> RGB)
    final inputSize = _inH * _inW * 3;
    _inputBuffer ??= Float32List(inputSize);
    ImageUtils.matToFloat32Tensor(paddedImage, buffer: _inputBuffer);

    // Clean up OpenCV Mats
    resizedImage.dispose();
    paddedImage.dispose();

    // Run inference
    final inputs = [_inputBuffer!.buffer];
    final outputs = <int, Object>{
      0: _outputBoxes!,
      1: _outputScores!,
    };

    if (_iso != null) {
      await _iso!.runForMultipleInputs(inputs, outputs);
    } else {
      _interpreter!.runForMultipleInputs(inputs, outputs);
    }

    // Decode boxes
    final decodedBoxes = _decodeBoxes(
      _outputBoxes![0],
      _outputScores![0],
    );

    // Postprocess
    return _postprocess(decodedBoxes);
  }

  /// Decodes raw box predictions using anchors.
  ///
  /// Returns decoded boxes as [score, cx, cy, boxSize, kp0X, kp0Y, kp2X, kp2Y].
  ///
  /// Optimization: Only decodes the 4 coordinate pairs actually used (indices 0-5, 8-9)
  /// instead of all 9 pairs (18 values), avoiding intermediate list allocation.
  List<List<double>> _decodeBoxes(
    List<List<double>> rawBoxes,
    List<List<double>> rawScores, {
    double scale = 192.0,
  }) {
    final results = <List<double>>[];
    final invScale = 1.0 / scale;

    for (int i = 0; i < rawBoxes.length; i++) {
      // Apply sigmoid to score
      final rawScore = rawScores[i][0];
      final score = 1.0 / (1.0 + math.exp(-rawScore));

      if (score <= scoreThreshold) continue;

      final rawBox = rawBoxes[i];
      final anchor = _anchors[i];

      // Decode only the coordinates we need (avoiding intermediate list)
      // rawBox layout: [cx, cy, w, h, kp0_x, kp0_y, kp1_x, kp1_y, kp2_x, kp2_y, ...]
      // Each value is offset relative to anchor, scaled by 192
      final anchorW = anchor[2];
      final anchorH = anchor[3];
      final anchorX = anchor[0];
      final anchorY = anchor[1];

      // Center point (rawBox indices 0,1)
      final cx = rawBox[0] * anchorW * invScale + anchorX;
      final cy = rawBox[1] * anchorH * invScale + anchorY;

      // Box size point (rawBox indices 2,3) - used to compute w,h
      final wPoint = rawBox[2] * anchorW * invScale + anchorX;
      final hPoint = rawBox[3] * anchorH * invScale + anchorY;
      final w = wPoint - anchorX;
      final h = hPoint - anchorY;
      final boxSize = math.max(w, h);

      // Keypoint 0 (rawBox indices 4,5) - used for rotation calculation
      final kp0X = rawBox[4] * anchorW * invScale + anchorX;
      final kp0Y = rawBox[5] * anchorH * invScale + anchorY;

      // Keypoint 2 (rawBox indices 8,9) - used for rotation calculation
      final kp2X = rawBox[8] * anchorW * invScale + anchorX;
      final kp2Y = rawBox[9] * anchorH * invScale + anchorY;

      results.add([score, cx, cy, boxSize, kp0X, kp0Y, kp2X, kp2Y]);
    }

    return results;
  }

  /// Post-processes decoded boxes to produce palm detections.
  ///
  /// Transforms coordinates from model space back to original image space,
  /// accounting for the padding applied during preprocessing.
  /// Matches Python's __postprocess implementation.
  List<PalmDetection> _postprocess(List<List<double>> boxes) {
    if (boxes.isEmpty) return [];

    final palms = <PalmDetection>[];

    for (final box in boxes) {
      final pdScore = box[0];
      final boxX = box[1];
      final boxY = box[2];
      final boxSize = box[3];
      final kp0X = box[4];
      final kp0Y = box[5];
      final kp2X = box[6];
      final kp2Y = box[7];

      if (boxSize > 0) {
        final kp02X = kp2X - kp0X;
        final kp02Y = kp2Y - kp0Y;
        var sqnRrSize = 2.9 * boxSize;
        var rotation = 0.5 * math.pi - math.atan2(-kp02Y, kp02X);
        rotation = normalizeRadians(rotation);
        var sqnRrCenterX = boxX + 0.5 * boxSize * math.sin(rotation);
        var sqnRrCenterY = boxY - 0.5 * boxSize * math.cos(rotation);

        // Adjust coordinates based on padding applied during preprocessing
        // Match Python palm_detection.py lines 368-371 exactly:
        // if image_height > image_width:
        //     sqn_rr_center_x = (sqn_rr_center_x * square_standard_size - square_padding_half_size) / image_width
        // else:
        //     sqn_rr_center_y = (sqn_rr_center_y * square_standard_size - square_padding_half_size) / image_height
        if (_imageHeight > _imageWidth) {
          // Portrait: padding was added to width, adjust X
          sqnRrCenterX =
              (sqnRrCenterX * _squareStandardSize - _squarePaddingHalfSize) /
                  _imageWidth;
        } else {
          // Landscape: padding was added to height, adjust Y
          sqnRrCenterY =
              (sqnRrCenterY * _squareStandardSize - _squarePaddingHalfSize) /
                  _imageHeight;
        }

        palms.add(PalmDetection(
          sqnRrSize: sqnRrSize,
          rotation: rotation,
          sqnRrCenterX: sqnRrCenterX,
          sqnRrCenterY: sqnRrCenterY,
          score: pdScore,
        ));
      }
    }

    // Apply NMS to remove overlapping detections
    return _nms(palms);
  }

  /// IoU-based Non-Maximum Suppression for palm detections.
  ///
  /// Uses Intersection-over-Union to properly handle overlapping boxes,
  /// rather than center distance which incorrectly suppresses non-overlapping hands.
  ///
  /// Optimization: Precomputes AABBs once (O(n)) before the O(n²) comparison loop
  /// to avoid redundant coordinate conversions.
  List<PalmDetection> _nms(List<PalmDetection> palms,
      {double iouThreshold = 0.45}) {
    if (palms.isEmpty) return palms;

    // Sort by score descending
    final sorted = List<PalmDetection>.from(palms)
      ..sort((a, b) => b.score.compareTo(a.score));

    // Precompute AABBs once - O(n) instead of O(n²) conversions
    final aabbs = List<_AABB>.generate(
      sorted.length,
      (i) => _palmToAABB(sorted[i]),
      growable: false,
    );

    final keep = <PalmDetection>[];
    final suppressed = List<bool>.filled(sorted.length, false);

    for (int i = 0; i < sorted.length; i++) {
      if (suppressed[i]) continue;
      keep.add(sorted[i]);

      for (int j = i + 1; j < sorted.length; j++) {
        if (suppressed[j]) continue;

        final iou = _calculateIoUFromAABB(aabbs[i], aabbs[j]);
        if (iou >= iouThreshold) {
          suppressed[j] = true;
        }
      }
    }

    return keep;
  }

  /// Converts a PalmDetection to an axis-aligned bounding box.
  _AABB _palmToAABB(PalmDetection p) {
    final maxDim = math.max(_imageWidth, _imageHeight).toDouble();
    final halfSize = (p.sqnRrSize * maxDim) / 2;
    final centerX = p.sqnRrCenterX * _imageWidth;
    final centerY = p.sqnRrCenterY * _imageHeight;
    return _AABB(
      centerX - halfSize,
      centerY - halfSize,
      centerX + halfSize,
      centerY + halfSize,
    );
  }

  /// Calculates Intersection over Union for two precomputed AABBs.
  double _calculateIoUFromAABB(_AABB a, _AABB b) {
    // Calculate intersection rectangle
    final interLeft = math.max(a.left, b.left);
    final interRight = math.min(a.right, b.right);
    final interTop = math.max(a.top, b.top);
    final interBottom = math.min(a.bottom, b.bottom);

    // No intersection if boxes don't overlap
    if (interRight <= interLeft || interBottom <= interTop) {
      return 0.0;
    }

    final interArea = (interRight - interLeft) * (interBottom - interTop);

    // Calculate union area
    final aArea = (a.right - a.left) * (a.bottom - a.top);
    final bArea = (b.right - b.left) * (b.bottom - b.top);
    final unionArea = aArea + bArea - interArea;

    if (unionArea <= 0) return 0.0;
    return interArea / unionArea;
  }

  /// Exposes anchor generation for testing.
  @visibleForTesting
  List<List<double>> get anchorsForTest => _anchors;

  /// Exposes input width for testing.
  @visibleForTesting
  int get inputWidth => _inW;

  /// Exposes input height for testing.
  @visibleForTesting
  int get inputHeight => _inH;
}
