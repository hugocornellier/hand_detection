// Headless batch video driver: reads every .mp4 in an input directory, runs
// hand detection + landmarks with One-Euro temporal smoothing, and bakes the
// overlay into an H.264 MP4 per clip. This is a headless lift of
// VideoFileScreen._processVideo / _drawHandsOnMat / HandSmoother from
// example/lib/main.dart, useful for regenerating demo videos and for manual
// pipeline A/B runs (e.g. tracking on/off) without the UI.
//
// It is driven entirely by --dart-define values and SKIPS (passing) when no
// input directory is configured, so it is a no-op under CI / runAllTests.sh:
//
//   flutter test integration_test/hand_video_driver_test.dart -d macos \
//     --dart-define=HAND_VIDEO_IN=/abs/path/to/clips \
//     --dart-define=HAND_VIDEO_OUT=/abs/path/to/out \
//     --dart-define=HAND_VIDEO_TRACKING=true \
//     --dart-define=HAND_VIDEO_CONF=0.5 \
//     --dart-define=HAND_VIDEO_MAX_HANDS=2

import 'dart:io';
import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:hand_detection/hand_detection_native.dart';
import 'package:flutter_litert/flutter_litert.dart' show OneEuroFilter;
import 'package:opencv_dart/opencv_dart.dart' as cv;

const String _inDir = String.fromEnvironment('HAND_VIDEO_IN');
const String _outDirDefine = String.fromEnvironment('HAND_VIDEO_OUT');
const bool _enableTracking =
    bool.fromEnvironment('HAND_VIDEO_TRACKING', defaultValue: true);
const String _confStr =
    String.fromEnvironment('HAND_VIDEO_CONF', defaultValue: '0.5');
const String _maxHandsStr =
    String.fromEnvironment('HAND_VIDEO_MAX_HANDS', defaultValue: '2');
// skeleton-and-landmarks-only overlay. Drawing-only: detection is unaffected.
const bool _drawBoxes =
    bool.fromEnvironment('HAND_VIDEO_BOXES', defaultValue: true);
// Hold a hand's last well-formed skeleton for up to this many frames when the
// detector briefly loses it or it becomes occluded. Keeps a skeleton-only
// overlay from flickering off (the bounding box used to hide those gaps).
// 0 keeps the original verbatim behavior.
const int _holdFrames = int.fromEnvironment('HAND_VIDEO_HOLD', defaultValue: 0);

// Overlay style, matching the VideoFileScreen defaults.
const int _boundingBoxColor = 0xFFFF9800; // orange
const int _landmarkColor = 0xFFFF3D00; // red-orange
const int _skeletonColor = 0xFF00E676; // green
const double _boundingBoxThickness = 2.0;
const double _landmarkSize = 3.0;
const double _skeletonThickness = 3.0;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('batch-process hand demo videos', (tester) async {
    if (_inDir.isEmpty) {
      // ignore: avoid_print
      print('SKIP: no --dart-define=HAND_VIDEO_IN=<dir> given.');
      return;
    }
    final inputDir = Directory(_inDir);
    if (!inputDir.existsSync()) {
      // ignore: avoid_print
      print('SKIP: input directory does not exist: $_inDir');
      return;
    }
    final outDir = _outDirDefine.isEmpty ? '$_inDir/annotated' : _outDirDefine;
    Directory(outDir).createSync(recursive: true);

    final conf = double.tryParse(_confStr) ?? 0.5;
    final maxHands = int.tryParse(_maxHandsStr) ?? 2;

    final detector = await HandDetector.create(
      mode: HandMode.boxesAndLandmarks,
      landmarkModel: HandLandmarkModel.full,
      detectorConf: conf,
      maxDetections: maxHands,
      minLandmarkScore: conf,
      performanceConfig: const PerformanceConfig.xnnpack(),
      enableGestures: false,
      enableTracking: _enableTracking,
      useCompiledModel: false,
    );
    // ignore: avoid_print
    print('CONFIG in=$_inDir out=$outDir tracking=$_enableTracking '
        'conf=$conf maxHands=$maxHands');

    final inputs = inputDir
        .listSync()
        .whereType<File>()
        .where((f) => f.path.toLowerCase().endsWith('.mp4'))
        .toList()
      ..sort((a, b) => a.path.compareTo(b.path));
    for (final f in inputs) {
      final name = f.uri.pathSegments.last.replaceAll(RegExp(r'\.mp4$'), '');
      final outPath = '$outDir/${name}_annot.mp4';
      try {
        await _processVideo(detector, f.path, outPath);
      } catch (e, st) {
        // ignore: avoid_print
        print('ERROR $name: $e\n$st');
      }
    }

    await detector.dispose();
  }, timeout: const Timeout(Duration(minutes: 45)));
}

Future<void> _processVideo(
    HandDetector detector, String path, String outPath) async {
  final cap = cv.VideoCapture.fromFile(path);
  if (!cap.isOpened) {
    cap.release();
    throw StateError('Could not open $path');
  }

  final fps = cap.get(cv.CAP_PROP_FPS);
  final width = cap.get(cv.CAP_PROP_FRAME_WIDTH).toInt();
  final height = cap.get(cv.CAP_PROP_FRAME_HEIGHT).toInt();
  final total = cap.get(cv.CAP_PROP_FRAME_COUNT).toInt();

  final writer = cv.VideoWriter.fromFile(outPath, 'avc1', fps, (width, height));
  if (!writer.isOpened) {
    cap.release();
    throw StateError('Could not open writer for $outPath ("avc1" unavailable)');
  }

  // New video: clear cross-frame state in both the detector and the smoother.
  await detector.resetTracking();
  final smoother = HandSmoother(holdFrames: _holdFrames);
  final sw = Stopwatch()..start();
  cv.Mat? frame;
  int idx = 0;
  int framesWithHands = 0;
  int twoHands = 0;
  final List<int> oneHandFrames = [];
  try {
    while (true) {
      final result = cap.read(m: frame);
      final ok = result.$1;
      frame = result.$2;
      if (!ok || frame.isEmpty) break;

      final List<Hand> raw = await detector.detectFromMat(frame);
      final double tSec = fps > 0 ? idx / fps : idx / 30.0;
      final List<Hand> hands = smoother.apply(raw, tSec);
      if (hands.isNotEmpty) framesWithHands++;
      if (hands.length >= 2) {
        twoHands++;
      } else {
        oneHandFrames.add(idx);
      }
      _drawHandsOnMat(frame, hands);
      writer.write(frame);
      idx++;
    }
  } finally {
    sw.stop();
    cap.release();
    writer.release();
    frame?.dispose();
  }

  final base = path.split('/').last;
  // ignore: avoid_print
  print('RESULT name=$base frames=$idx withHands=$framesWithHands '
      'twoHands=$twoHands boxes=$_drawBoxes '
      'total=$total fps=${fps.toStringAsFixed(2)} '
      'secs=${(sw.elapsedMilliseconds / 1000).toStringAsFixed(1)} out=$outPath');
  // ignore: avoid_print
  print('ONEHAND ${oneHandFrames.join(",")}');
}

cv.Scalar _bgr(int argb) {
  final r = (argb >> 16) & 0xFF;
  final g = (argb >> 8) & 0xFF;
  final b = argb & 0xFF;
  return cv.Scalar(b.toDouble(), g.toDouble(), r.toDouble());
}

void _drawHandsOnMat(cv.Mat mat, List<Hand> hands) {
  if (hands.isEmpty) return;
  final black = cv.Scalar(0, 0, 0);
  final w = mat.cols;
  final h = mat.rows;

  for (final hand in hands) {
    if (hand.hasLandmarks) {
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

    if (hand.hasLandmarks) {
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

    if (!_drawBoxes) continue;

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
    if (hand.handedness != null) {
      parts.add(hand.handedness == Handedness.right ? 'R' : 'L');
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

// ─────────────────────────── Hand Smoother ────────────────────────────────
// Verbatim from example/lib/main.dart.

class HandSmoother {
  bool enabled;
  static const int _maxMissed = 5;
  static const double _minIou = 0.2;
  // A hand needs at least this many drawable (visibility > 0.5) landmarks to
  // count as "well formed" and be remembered / re-emitted during a hold.
  static const int _minVisibleLandmarks = 12;
  final List<_HandTrack> _tracks = [];
  final int holdFrames;

  HandSmoother({this.enabled = true, this.holdFrames = 0});

  void reset() => _tracks.clear();

  static bool _wellFormed(Hand h) {
    if (!h.hasLandmarks) return false;
    var n = 0;
    for (final lm in h.landmarks) {
      if (lm.visibility > 0.5) n++;
    }
    return n >= _minVisibleLandmarks;
  }

  List<Hand> apply(List<Hand> hands, double tSec) {
    if (!enabled) {
      _tracks.clear();
      return hands;
    }
    if (holdFrames <= 0 && hands.isEmpty) return hands;

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
      final sh = _smoothHand(hands[p], track, tSec);
      if (holdFrames > 0) {
        if (_wellFormed(sh)) {
          // Good pose: draw it and remember it.
          track.lastDrawn = sh;
          track.blankFrames = 0;
          out.add(sh);
        } else if (track.lastDrawn != null && track.blankFrames < holdFrames) {
          // Occluded/blank this frame: hold the last well-formed skeleton.
          track.blankFrames++;
          out.add(track.lastDrawn!);
        } else {
          out.add(sh);
        }
      } else {
        out.add(sh);
      }
    }

    // Re-emit a briefly-lost hand's last well-formed pose so a skeleton-only
    // overlay does not flicker off during short detector gaps.
    for (final t in unmatched) {
      final tr = _tracks[t];
      if (holdFrames > 0 &&
          tr.lastDrawn != null &&
          tr.missedFrames < holdFrames) {
        out.add(tr.lastDrawn!);
      }
      tr.missedFrames++;
    }
    _tracks
        .removeWhere((t) => t.missedFrames > math.max(_maxMissed, holdFrames));

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
  // Last well-formed pose and how many consecutive frames we have been holding
  // it, used to bridge short detector/occlusion gaps in a skeleton-only overlay.
  Hand? lastDrawn;
  int blankFrames = 0;
}
