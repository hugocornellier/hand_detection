// ignore_for_file: implementation_imports, public_member_api_docs

import 'dart:js_interop';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:flutter_litert/flutter_litert.dart'
    show rgbaToRgbFloat32, sigmoidClipped;
import 'package:flutter_litert/src/web/litertjs_interpreter.dart'
    show LiteRtInterpreter;
import 'package:web/web.dart' as web;

/// Web hand landmark runner (Stage 2). Mirrors native [HandLandmarkModelRunner]
/// but extracts the rotation-aware hand crop with a Canvas transform (the
/// inverse of which the caller applies to map landmarks back to image space).
class HandLandmarkModelWeb {
  LiteRtInterpreter? _liteRtItp;
  String? _activeAccelerator;

  // MediaPipe hand-landmark output order: landmarks(63), score(1),
  // handedness(1), world landmarks(63).
  static const int _lmIdx = 0;
  static const int _scoreIdx = 1;
  static const int _handIdx = 2;
  static const int _worldIdx = 3;

  late int _inW;
  late int _inH;
  Float32List? _lmOut;
  Float32List? _scoreOut;
  Float32List? _handOut;
  Float32List? _worldOut;
  Float32List? _inputBuffer;
  web.HTMLCanvasElement? _canvas;
  web.CanvasRenderingContext2D? _ctx;
  bool _initialized = false;

  bool get isInitialized => _initialized;
  int get inputWidth => _inW;
  int get inputHeight => _inH;
  String? get activeAccelerator =>
      _liteRtItp != null ? _activeAccelerator : null;

  Future<void> initialize({String liteRtAccelerator = 'auto'}) async {
    if (_initialized) await dispose();

    const String assetPath =
        'packages/hand_detection/assets/models/hand_landmark_full.tflite';
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

    final outs = _liteRtItp!.getOutputTensors();
    int elems(int i) {
      int n = 1;
      for (final d in outs[i].shape) {
        n *= d;
      }
      return n;
    }

    const int lmFloats = 21 * 3;
    if (outs.length < 4 ||
        elems(_lmIdx) != lmFloats ||
        elems(_scoreIdx) != 1 ||
        elems(_handIdx) != 1 ||
        elems(_worldIdx) != lmFloats) {
      throw StateError(
        'Hand landmark model outputs do not match the expected '
        '[landmarks=$lmFloats, score=1, handedness=1, world=$lmFloats] layout. '
        'Got ${[for (final t in outs) t.shape]}',
      );
    }

    _lmOut = Float32List(lmFloats);
    _scoreOut = Float32List(1);
    _handOut = Float32List(1);
    _worldOut = Float32List(lmFloats);
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
    _lmOut = null;
    _scoreOut = null;
    _handOut = null;
    _worldOut = null;
    _inputBuffer = null;
    _canvas = null;
    _ctx = null;
    _initialized = false;
  }

  /// Runs the landmark model on a rotation-aware crop centered at ([cx], [cy])
  /// with side [size] and rotation [theta] (radians). Landmarks come back in
  /// the model's input pixel space; the caller inverts the crop transform.
  Future<
      ({
        Float32List landmarks,
        Float32List world,
        double score,
        double handedness
      })> runOnCrop(
    JSObject canvasSource, {
    required double cx,
    required double cy,
    required double size,
    required double theta,
  }) async {
    if (!_initialized) {
      throw StateError('HandLandmarkModelWeb not initialized.');
    }
    final ctx = _ctx!;

    ctx.save();
    ctx.fillStyle = 'rgb(0,0,0)'.toJS;
    ctx.fillRect(0, 0, _inW, _inH);
    final double scale = _inW / size;
    ctx.translate(_inW / 2.0, _inH / 2.0);
    ctx.rotate(-theta);
    ctx.scale(scale, scale);
    ctx.translate(-cx, -cy);
    ctx.drawImage(canvasSource, 0, 0);
    ctx.restore();

    final web.ImageData imageData = ctx.getImageData(0, 0, _inW, _inH);
    final rgba = imageData.data.toDart;
    final input = _inputBuffer!;
    rgbaToRgbFloat32(Uint8List.view(rgba.buffer), input);

    await _liteRtItp!.runForMultipleInputs(
      <Object>[input],
      <int, Object>{
        _lmIdx: _lmOut!,
        _scoreIdx: _scoreOut!,
        _handIdx: _handOut!,
        _worldIdx: _worldOut!,
      },
    );

    return (
      landmarks: Float32List.fromList(_lmOut!),
      world: Float32List.fromList(_worldOut!),
      score: sigmoidClipped(_scoreOut![0]),
      handedness: _handOut![0],
    );
  }
}
