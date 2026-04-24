import 'dart:math' show cos, max, min, sin;
import 'package:flutter/material.dart';
import 'package:flutter_litert/flutter_litert.dart' show drawLandmarkMarker;
import '../types.dart';

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

/// Paints hand detection results over a still image.
///
/// Draws bounding boxes (axis-aligned and rotated), skeleton connections,
/// and landmark markers for all detected hands.
class MultiOverlayPainter extends CustomPainter {
  /// Hands to render.
  final List<Hand> results;

  late final _glowPaint = Paint()..color = Colors.blue.withValues(alpha: 0.3);
  late final _pointPaint = Paint()..color = Colors.red;
  late final _dotPaint = Paint()..color = Colors.white;

  /// Creates a painter for the given [results].
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
        final double cx = l.x * scaleX + offsetX;
        final double cy = l.y * scaleY + offsetY;
        drawLandmarkMarker(canvas, cx, cy,
            glowPaint: _glowPaint,
            pointPaint: _pointPaint,
            centerPaint: _dotPaint);
      }
    }
  }

  void _drawBbox(Canvas canvas, Hand r, double scaleX, double scaleY,
      double offsetX, double offsetY) {
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
  bool shouldRepaint(covariant MultiOverlayPainter old) {
    return old.results != results;
  }
}

String _gestureToEmoji(GestureType gesture) {
  switch (gesture) {
    case GestureType.thumbUp:
      return '\u{1F44D}';
    case GestureType.thumbDown:
      return '\u{1F44E}';
    case GestureType.victory:
      return '\u{270C}';
    case GestureType.openPalm:
      return '\u{1F590}';
    case GestureType.closedFist:
      return '\u{270A}';
    case GestureType.pointingUp:
      return '\u{261D}';
    case GestureType.iLoveYou:
      return '\u{1F91F}';
    case GestureType.unknown:
      return '';
  }
}

/// Paints hand detection results over a live camera preview.
///
/// Handles optional horizontal mirroring for front cameras, draws bounding
/// boxes, skeleton connections, landmark markers, and gesture overlays.
class CameraHandOverlayPainter extends CustomPainter {
  /// Hands to render.
  final List<Hand> hands;

  /// Post-rotation source image size in pixels (for coord-space mapping).
  final Size imageSize;

  /// When true, flips x-coordinates to match a mirrored front-camera preview.
  final bool mirrorHorizontally;

  late final _glowPaint = Paint()..color = Colors.blue.withValues(alpha: 0.3);
  late final _pointPaint = Paint()..color = Colors.red;
  late final _dotPaint = Paint()..color = Colors.white;

  /// Creates a painter for the given [hands], source [imageSize], and mirror
  /// flag.
  CameraHandOverlayPainter({
    required this.hands,
    required this.imageSize,
    required this.mirrorHorizontally,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (hands.isEmpty) return;

    final double sourceWidth = imageSize.width;
    final double sourceHeight = imageSize.height;

    final double sourceAspectRatio = sourceWidth / sourceHeight;
    final double viewportAspectRatio = size.width / size.height;

    double scaleX, scaleY;
    double offsetX = 0, offsetY = 0;

    if (sourceAspectRatio > viewportAspectRatio) {
      scaleY = size.height / sourceHeight;
      scaleX = scaleY;
      offsetX = (size.width - sourceWidth * scaleX) / 2;
    } else {
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
        final double cx = tx(l.x);
        final double cy = l.y * scaleY + offsetY;
        drawLandmarkMarker(canvas, cx, cy,
            glowPaint: _glowPaint,
            pointPaint: _pointPaint,
            centerPaint: _dotPaint);
      }
    }
  }

  void _drawBbox(Canvas canvas, Hand hand, double Function(double) tx,
      double scaleX, double scaleY, double offsetX, double offsetY) {
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

    if (gesture.confidence < 0.6) return;

    final emoji = _gestureToEmoji(gesture.type);
    if (emoji.isEmpty) return;

    final double x = tx((hand.boundingBox.left + hand.boundingBox.right) / 2);
    final double y = hand.boundingBox.top * scaleY + offsetY - 20;

    final Paint bgPaint = Paint()
      ..color = Colors.white.withValues(alpha: 0.9)
      ..style = PaintingStyle.fill;
    canvas.drawCircle(Offset(x, y), 28, bgPaint);

    final Paint borderPaint = Paint()
      ..color = Colors.blue.withValues(alpha: 0.8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3;
    canvas.drawCircle(Offset(x, y), 28, borderPaint);

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

  @override
  bool shouldRepaint(covariant CameraHandOverlayPainter old) {
    return old.hands != hands ||
        old.imageSize != imageSize ||
        old.mirrorHorizontally != mirrorHorizontally;
  }
}
