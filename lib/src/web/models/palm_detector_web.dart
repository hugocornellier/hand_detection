// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart'
    show computeLetterboxParams, rgbaToRgbFloat32;
import 'package:flutter_litert/src/web/litertjs_interpreter.dart'
    show LiteRtInterpreter;
import 'package:web/web.dart' as web;

import '../../shared/hand_geometry.dart';

/// Web palm detector (Stage 1). Mirrors the native [PalmDetector] but uses
/// LiteRT.js + an offscreen Canvas for preprocessing instead of OpenCV.
///
/// Reuses the shared pure-Dart [decodePalmBoxes] / [postprocessPalms] so the
/// SSD decode and rotation-rectangle math are identical to native.
class PalmDetectorWeb {
  LiteRtInterpreter? _liteRtItp;
  String? _activeAccelerator;

  /// Detection confidence threshold.
  final double scoreThreshold;

  /// IoU threshold for the palm non-maximum suppression.
  final double nmsIouThreshold;

  /// Expansion factor for the palm ROI fed to the landmark model.
  final double roiScale;

  late int _inH;
  late int _inW;
  late List<List<double>> _anchors;
  int _boxStride = 18;
  int _boxesIdx = 0;
  int _scoresIdx = 1;

  Float32List? _inputBuffer;
  Float32List? _boxesOut;
  Float32List? _scoresOut;
  web.HTMLCanvasElement? _canvas;
  web.CanvasRenderingContext2D? _ctx;
  bool _initialized = false;

  PalmDetectorWeb({
    this.scoreThreshold = 0.45,
    this.nmsIouThreshold = 0.45,
    this.roiScale = 2.6,
  });

  bool get isInitialized => _initialized;
  String? get activeAccelerator =>
      _liteRtItp != null ? _activeAccelerator : null;

  Future<void> initialize({String liteRtAccelerator = 'auto'}) async {
    if (_initialized) await dispose();

    const String assetPath =
        'packages/hand_detection/assets/models/hand_detection.tflite';
    final ByteData raw = await rootBundle.load(assetPath);
    final bytes = raw.buffer.asUint8List();

    final String resolved =
        liteRtAccelerator == 'auto' ? 'webgpu' : liteRtAccelerator;
    _liteRtItp =
        await LiteRtInterpreter.fromBytes(bytes, accelerator: resolved);
    _activeAccelerator = resolved;

    final inT = _liteRtItp!.getInputTensor(0);
    _inH = inT.shape[1];
    _inW = inT.shape[2];
    _anchors = buildPalmAnchors(_inH, _inW);
    final int numAnchors = _anchors.length;

    // Locate the boxes output (last dim is the per-anchor stride, >= 16) and the
    // scores output (one value per anchor). Resolve indices from shapes so we
    // do not assume the model's output order.
    final outs = _liteRtItp!.getOutputTensors();
    int boxesIdx = -1;
    int scoresIdx = -1;
    int boxesElems = 0;
    int scoresElems = 0;
    for (int i = 0; i < outs.length; i++) {
      final shape = List<int>.from(outs[i].shape);
      int n = 1;
      for (final d in shape) {
        n *= d;
      }
      final last = shape.last;
      if (last >= 16 && boxesIdx < 0) {
        boxesIdx = i;
        boxesElems = n;
        _boxStride = last;
      } else if (last == 1 && scoresIdx < 0) {
        scoresIdx = i;
        scoresElems = n;
      }
    }
    if (boxesIdx < 0 || scoresIdx < 0 || scoresElems != numAnchors) {
      throw StateError(
        'Palm model outputs do not match expected shapes. Got '
        '${[for (final t in outs) t.shape]}',
      );
    }
    _boxesIdx = boxesIdx;
    _scoresIdx = scoresIdx;
    _boxesOut = Float32List(boxesElems);
    _scoresOut = Float32List(scoresElems);

    _inputBuffer = Float32List(_inH * _inW * 3);
    _canvas = web.HTMLCanvasElement()
      ..width = _inW
      ..height = _inH;
    _ctx = _canvas!.getContext('2d') as web.CanvasRenderingContext2D;

    _initialized = true;
  }

  Future<void> dispose() async {
    _liteRtItp?.close();
    _liteRtItp = null;
    _activeAccelerator = null;
    _inputBuffer = null;
    _boxesOut = null;
    _scoresOut = null;
    _canvas = null;
    _ctx = null;
    _initialized = false;
  }

  /// Detects palms in a drawable source (ImageBitmap / HTMLVideoElement / ...).
  ///
  /// Letterboxes the source into the model's square input via Canvas (aspect
  /// preserved with centered padding, matching the native
  /// `keepAspectResizeAndPad`), runs inference, then reuses the shared decode +
  /// rotation-rectangle post-processing.
  Future<List<PalmDetection>> detect(
    JSObject canvasSource, {
    required int imageWidth,
    required int imageHeight,
  }) async {
    if (!_initialized) {
      throw StateError('PalmDetectorWeb not initialized.');
    }

    final lb = computeLetterboxParams(
      srcWidth: imageWidth,
      srcHeight: imageHeight,
      targetWidth: _inW,
      targetHeight: _inH,
      roundDimensions: false,
    );

    final ctx = _ctx!;
    ctx.fillStyle = 'rgb(0,0,0)'.toJS;
    ctx.fillRect(0, 0, _inW, _inH);
    ctx.drawImage(
      canvasSource,
      0,
      0,
      imageWidth,
      imageHeight,
      lb.padLeft,
      lb.padTop,
      lb.newWidth,
      lb.newHeight,
    );

    final web.ImageData imageData = ctx.getImageData(0, 0, _inW, _inH);
    final rgba = imageData.data.toDart;
    final Float32List input = _inputBuffer!;
    rgbaToRgbFloat32(Uint8List.view(rgba.buffer), input);

    await _liteRtItp!.runForMultipleInputs(
      <Object>[input],
      <int, Object>{_boxesIdx: _boxesOut!, _scoresIdx: _scoresOut!},
    );

    final decoded = decodePalmBoxes(
      _boxesOut!,
      _scoresOut!,
      _anchors,
      _boxStride,
      scoreThreshold,
      scale: _inH.toDouble(),
    );
    final int squareStandardSize = math.max(imageHeight, imageWidth);
    final int squarePaddingHalfSize = (imageHeight - imageWidth).abs() ~/ 2;
    return postprocessPalms(
      decoded,
      imageWidth: imageWidth,
      imageHeight: imageHeight,
      squareStandardSize: squareStandardSize,
      squarePaddingHalfSize: squarePaddingHalfSize,
      roiScale: roiScale,
      iouThreshold: nmsIouThreshold,
    );
  }
}
