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
//
// Optional gesture badges (off by default). Overlays a stabilized gesture
// label + emoji icon above each hand. Icons are <GestureType>.png files (Noto
// emoji PNGs) in the HAND_VIDEO_ICONS directory; detection is unaffected:
//
//     --dart-define=HAND_VIDEO_GESTURES=true \
//     --dart-define=HAND_VIDEO_ICONS=/abs/path/to/icons

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

// ─────────────────────────── Gesture badges ───────────────────────────────
// Opt-in. When enabled, the detector runs gesture recognition and a stabilized
// gesture label + emoji icon is drawn above each hand. Drawing-only.
const bool _enableGestures =
    bool.fromEnvironment('HAND_VIDEO_GESTURES', defaultValue: false);
const String _iconsDir = String.fromEnvironment('HAND_VIDEO_ICONS');
const String _gestureConfStr =
    String.fromEnvironment('HAND_VIDEO_GESTURE_CONF', defaultValue: '0.55');
// A gesture must repeat for this many consecutive frames before it is shown,
// and it is held for this many frames after it stops being detected. Together
// they remove single-frame flicker and brief unknown/low-confidence dips.
const int _gestureConfirm =
    int.fromEnvironment('HAND_VIDEO_GESTURE_CONFIRM', defaultValue: 3);
const int _gestureHold =
    int.fromEnvironment('HAND_VIDEO_GESTURE_HOLD', defaultValue: 6);

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
    final gestureConf = double.tryParse(_gestureConfStr) ?? 0.55;

    final detector = await HandDetector.create(
      mode: HandMode.boxesAndLandmarks,
      landmarkModel: HandLandmarkModel.full,
      detectorConf: conf,
      maxDetections: maxHands,
      minLandmarkScore: conf,
      performanceConfig: const PerformanceConfig.xnnpack(),
      enableGestures: _enableGestures,
      enableTracking: _enableTracking,
      useCompiledModel: false,
    );

    // Load the gesture-badge icons once (512px BGR + binary alpha mask each).
    // They are resized per-video to the badge size inside _processVideo.
    final Map<GestureType, _RawIcon> rawIcons =
        (_enableGestures && _iconsDir.isNotEmpty)
            ? _loadRawIcons(_iconsDir)
            : <GestureType, _RawIcon>{};

    // ignore: avoid_print
    print('CONFIG in=$_inDir out=$outDir tracking=$_enableTracking '
        'conf=$conf maxHands=$maxHands gestures=$_enableGestures '
        'icons=${rawIcons.length} gestureConf=$gestureConf');

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
        await _processVideo(detector, f.path, outPath, rawIcons, gestureConf);
      } catch (e, st) {
        // ignore: avoid_print
        print('ERROR $name: $e\n$st');
      }
    }

    for (final ri in rawIcons.values) {
      ri.dispose();
    }
    await detector.dispose();
  }, timeout: const Timeout(Duration(minutes: 45)));
}

Future<void> _processVideo(HandDetector detector, String path, String outPath,
    Map<GestureType, _RawIcon> rawIcons, double gestureConf) async {
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

  // Per-video badge sizing: scale to the shorter side so portrait and landscape
  // clips get proportionate badges. Resize each icon to the badge size once.
  final int iconD = math.min(width, height) > 0
      ? (math.min(width, height) * 0.06).round().clamp(40, 160)
      : 64;
  final Map<GestureType, _SizedIcon> icons = {};
  rawIcons.forEach((g, ri) {
    icons[g] = _SizedIcon(
      cv.resize(ri.bgr, (iconD, iconD)),
      cv.resize(ri.mask, (iconD, iconD)),
    );
  });

  // New video: clear cross-frame state in both the detector and the smoother.
  await detector.resetTracking();
  final smoother = HandSmoother(
    holdFrames: _holdFrames,
    gestureConf: gestureConf,
    gestureConfirm: _gestureConfirm,
    gestureHold: _gestureHold,
  );
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
      _drawHandsOnMat(frame, hands, icons, iconD);
      writer.write(frame);
      idx++;
    }
  } finally {
    sw.stop();
    cap.release();
    writer.release();
    frame?.dispose();
    for (final si in icons.values) {
      si.dispose();
    }
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

void _drawHandsOnMat(cv.Mat mat, List<Hand> hands,
    Map<GestureType, _SizedIcon> icons, int iconD) {
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

    if (_drawBoxes) {
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

    _drawGestureBadge(mat, hand, icons, iconD);
  }
}

// Draws a stadium ("pill") badge above the hand: [emoji icon] + gesture name,
// on a solid dark background so it reads over any footage. No-op unless the
// hand carries a stabilized gesture.
void _drawGestureBadge(
    cv.Mat mat, Hand hand, Map<GestureType, _SizedIcon> icons, int iconD) {
  final g = hand.gesture;
  if (g == null || g.type == GestureType.unknown) return;
  final label = _gestureLabelText[g.type];
  if (label == null) return;

  final int w = mat.cols;
  final int h = mat.rows;

  final double fontScale = iconD / 50.0;
  final int thickness = math.max(2, (iconD / 26).round());
  final (textSz, _) =
      cv.getTextSize(label, cv.FONT_HERSHEY_DUPLEX, fontScale, thickness);

  final _SizedIcon? icon = icons[g.type];
  final int pad = (iconD * 0.34).round();
  final int gap = icon != null ? (iconD * 0.28).round() : 0;
  final int iconW = icon != null ? iconD : 0;
  final int contentH = math.max(iconD, textSz.height);
  final int pillH = contentH + 2 * pad;
  final int pillW = pad + iconW + gap + textSz.width + pad;
  if (pillW >= w || pillH >= h) return;

  // Center above the box; clamp the whole pill into the frame.
  final bb = hand.boundingBox;
  final double cx = (bb.left + bb.right) / 2;
  final int px0 = (cx - pillW / 2).round().clamp(2, w - pillW - 2);
  int py0 = (bb.top - pillH - iconD * 0.25).round();
  if (py0 < 2) py0 = (bb.top + iconD * 0.25).round();
  py0 = py0.clamp(2, h - pillH - 2);

  // Stadium background: a rectangle capped by two half-circles.
  final pill = cv.Scalar(28, 26, 24); // near-black, faint warm tint (BGR)
  final int r = pillH ~/ 2;
  cv.rectangle(
    mat,
    cv.Rect(px0 + r, py0, pillW - 2 * r, pillH),
    pill,
    thickness: -1,
  );
  cv.circle(mat, cv.Point(px0 + r, py0 + r), r, pill, thickness: -1);
  cv.circle(mat, cv.Point(px0 + pillW - r, py0 + r), r, pill, thickness: -1);

  // Emoji icon, alpha-composited over the (now opaque) pill background.
  if (icon != null) {
    final int ix = px0 + pad;
    final int iy = py0 + (pillH - iconD) ~/ 2;
    final roi = mat.region(cv.Rect(ix, iy, iconD, iconD));
    icon.bgr.copyTo(roi, mask: icon.mask);
    roi.dispose();
  }

  // Gesture name.
  final int tx = px0 + pad + iconW + gap;
  final int ty = py0 + (pillH + textSz.height) ~/ 2;
  cv.putText(
    mat,
    label,
    cv.Point(tx, ty),
    cv.FONT_HERSHEY_DUPLEX,
    fontScale,
    cv.Scalar(255, 255, 255),
    thickness: thickness,
  );
}

// ───────────────────────────── Gesture icons ──────────────────────────────

const Map<GestureType, String> _gestureIconFile = {
  GestureType.thumbUp: 'thumbUp',
  GestureType.thumbDown: 'thumbDown',
  GestureType.victory: 'victory',
  GestureType.openPalm: 'openPalm',
  GestureType.closedFist: 'closedFist',
  GestureType.pointingUp: 'pointingUp',
  GestureType.iLoveYou: 'iLoveYou',
};

const Map<GestureType, String> _gestureLabelText = {
  GestureType.thumbUp: 'Thumbs Up',
  GestureType.thumbDown: 'Thumbs Down',
  GestureType.victory: 'Victory',
  GestureType.openPalm: 'Open Palm',
  GestureType.closedFist: 'Closed Fist',
  GestureType.pointingUp: 'Pointing Up',
  GestureType.iLoveYou: 'I Love You',
};

// A gesture icon at native (512px) resolution: BGR color + binary alpha mask.
class _RawIcon {
  final cv.Mat bgr;
  final cv.Mat mask;
  _RawIcon(this.bgr, this.mask);
  void dispose() {
    bgr.dispose();
    mask.dispose();
  }
}

// A gesture icon resized to the per-video badge size.
class _SizedIcon {
  final cv.Mat bgr;
  final cv.Mat mask;
  _SizedIcon(this.bgr, this.mask);
  void dispose() {
    bgr.dispose();
    mask.dispose();
  }
}

Map<GestureType, _RawIcon> _loadRawIcons(String dir) {
  final out = <GestureType, _RawIcon>{};
  _gestureIconFile.forEach((g, base) {
    final path = '$dir/$base.png';
    if (!File(path).existsSync()) return;
    final raw = cv.imread(path, flags: cv.IMREAD_UNCHANGED);
    if (raw.isEmpty || raw.channels != 4) {
      raw.dispose();
      return;
    }
    final bgr = cv.cvtColor(raw, cv.COLOR_BGRA2BGR);
    final ch = cv.split(raw);
    // Anti-aliased alpha -> a binary mask; drop the near-transparent fringe.
    final (_, mask) = cv.threshold(ch[3], 40, 255, cv.THRESH_BINARY);
    for (final c in ch) {
      c.dispose();
    }
    raw.dispose();
    out[g] = _RawIcon(bgr, mask);
  });
  return out;
}

// ─────────────────────────── Hand Smoother ────────────────────────────────
// Lifted from example/lib/main.dart, extended here with per-track gesture
// stabilization (consecutive-frame confirm + post-detection hold) so gesture
// badges do not flicker frame-to-frame.

class HandSmoother {
  bool enabled;
  static const int _maxMissed = 5;
  static const double _minIou = 0.2;
  // A hand needs at least this many drawable (visibility > 0.5) landmarks to
  // count as "well formed" and be remembered / re-emitted during a hold.
  static const int _minVisibleLandmarks = 12;
  final List<_HandTrack> _tracks = [];
  final int holdFrames;
  final double gestureConf;
  final int gestureConfirm;
  final int gestureHold;

  HandSmoother({
    this.enabled = true,
    this.holdFrames = 0,
    this.gestureConf = 0.55,
    this.gestureConfirm = 3,
    this.gestureHold = 6,
  });

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
      final stable = track.updateGesture(
          hands[p].gesture, gestureConf, gestureConfirm, gestureHold);
      final sh = _smoothHand(hands[p], track, tSec, stable);
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

  Hand _smoothHand(
      Hand hand, _HandTrack track, double tSec, GestureResult? gesture) {
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
      gesture: gesture,
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

  // Gesture stabilization state.
  GestureType? _gCandidate;
  int _gCandidateCount = 0;
  GestureType? _gStable;
  double _gStableConf = 0;
  int _gHold = 0;

  // Feeds this frame's raw gesture in and returns the stabilized gesture (or
  // null). A gesture is shown only after [confirmFrames] consecutive detections
  // and is held for [holdFrames] frames once detection lapses.
  GestureResult? updateGesture(
      GestureResult? raw, double minConf, int confirmFrames, int holdFrames) {
    if (raw != null &&
        raw.type != GestureType.unknown &&
        raw.confidence >= minConf) {
      final t = raw.type;
      if (t == _gCandidate) {
        _gCandidateCount++;
      } else {
        _gCandidate = t;
        _gCandidateCount = 1;
      }
      if (_gCandidateCount >= confirmFrames || _gStable == t) {
        _gStable = t;
        _gStableConf = raw.confidence;
        _gHold = holdFrames;
      }
    } else {
      _gCandidate = null;
      _gCandidateCount = 0;
      if (_gHold > 0) {
        _gHold--;
      } else {
        _gStable = null;
      }
    }
    if (_gStable == null) return null;
    return GestureResult(type: _gStable!, confidence: _gStableConf);
  }
}
