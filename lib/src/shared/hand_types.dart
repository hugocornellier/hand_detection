import 'package:flutter_litert/flutter_litert.dart'
    show BoundingBox, LandmarkMixin;

/// Hand landmark model variant for landmark extraction.
///
/// Only the full model is available to match the Python implementation.
enum HandLandmarkModel {
  /// Full model with balanced speed and accuracy.
  full,
}

/// Detection mode controlling the two-stage pipeline behavior.
///
/// - [boxes]: Fast detection returning only bounding boxes (Stage 1 only)
/// - [boxesAndLandmarks]: Full pipeline returning boxes + landmarks (both stages)
enum HandMode {
  /// Fast detection mode returning only bounding boxes (Stage 1 only).
  boxes,

  /// Full pipeline mode returning bounding boxes and landmarks per hand.
  boxesAndLandmarks,
}

/// Recognized hand gesture types from MediaPipe gesture classifier.
///
/// The gesture classifier recognizes 7 distinct gestures plus an "unknown" category
/// for unrecognized hand poses.
///
/// Example:
/// ```dart
/// final hands = await detector.detect(imageBytes);
/// for (final hand in hands) {
///   if (hand.gesture != null && hand.gesture!.confidence > 0.7) {
///     switch (hand.gesture!.type) {
///       case GestureType.thumbUp:
///         print('Thumbs up detected!');
///         break;
///       case GestureType.victory:
///         print('Peace sign detected!');
///         break;
///       default:
///         break;
///     }
///   }
/// }
/// ```
enum GestureType {
  /// Unrecognized or ambiguous gesture.
  unknown,

  /// Closed fist gesture. ✊
  closedFist,

  /// Open palm with fingers extended. 🖐️
  openPalm,

  /// Index finger pointing upward. ☝️
  pointingUp,

  /// Thumbs down gesture. 👎
  thumbDown,

  /// Thumbs up gesture. 👍
  thumbUp,

  /// Victory/peace sign (index and middle fingers extended). ✌️
  victory,

  /// "I Love You" sign (thumb, index, and pinky extended). 🤟
  iLoveYou,
}

/// Result of gesture recognition containing the detected gesture and confidence.
///
/// Returned as part of [Hand] when gesture recognition is enabled.
class GestureResult {
  /// The recognized gesture type.
  final GestureType type;

  /// Confidence score for the gesture recognition (0.0 to 1.0).
  ///
  /// Higher values indicate more confident predictions. A threshold of 0.7
  /// is recommended for filtering uncertain predictions.
  final double confidence;

  /// Creates a gesture result with the given type and confidence.
  const GestureResult({
    required this.type,
    required this.confidence,
  });

  /// Serializes this gesture result to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'type': type.name,
        'confidence': confidence,
      };

  /// Deserializes a gesture result from a map.
  static GestureResult fromMap(Map<String, dynamic> map) => GestureResult(
        type: GestureType.values.firstWhere((e) => e.name == map['type']),
        confidence: (map['confidence'] as num).toDouble(),
      );

  @override
  String toString() =>
      'GestureResult(type: ${type.name}, confidence: ${confidence.toStringAsFixed(3)})';
}

/// Collection of hand landmarks with confidence score (internal use).
class HandLandmarks {
  /// List of 21 landmarks extracted from the hand landmark model.
  /// Coordinates are in the original image pixel space.
  final List<HandLandmark> landmarks;

  /// List of 21 world-space landmarks for gesture recognition.
  /// These are in a normalized 3D coordinate system relative to the hand.
  final List<HandLandmark> worldLandmarks;

  /// Confidence score for the landmark extraction (0.0 to 1.0).
  final double score;

  /// Handedness of the detected hand (left or right).
  final Handedness handedness;

  /// Creates a collection of hand landmarks with a confidence score and handedness.
  HandLandmarks({
    required this.landmarks,
    required this.worldLandmarks,
    required this.score,
    required this.handedness,
  });
}

/// A single keypoint with 3D coordinates and visibility score.
///
/// Coordinates are in the original image space (pixels).
/// The [z] coordinate represents depth relative to the center (not absolute depth).
class HandLandmark with LandmarkMixin {
  /// The landmark type this represents
  final HandLandmarkType type;

  /// X coordinate in pixels (original image space)
  @override
  final double x;

  /// Y coordinate in pixels (original image space)
  @override
  final double y;

  /// Z coordinate representing depth (not absolute depth)
  final double z;

  /// Visibility/confidence score (0.0 to 1.0). Higher means more confident the landmark is visible.
  final double visibility;

  /// Creates a hand landmark with 3D coordinates and visibility score.
  HandLandmark({
    required this.type,
    required this.x,
    required this.y,
    required this.z,
    required this.visibility,
  });

  /// Serializes this landmark to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'type': type.name,
        'x': x,
        'y': y,
        'z': z,
        'visibility': visibility,
      };

  /// Deserializes a landmark from a map.
  static HandLandmark fromMap(Map<String, dynamic> map) => HandLandmark(
        type: HandLandmarkType.values.firstWhere((e) => e.name == map['type']),
        x: (map['x'] as num).toDouble(),
        y: (map['y'] as num).toDouble(),
        z: (map['z'] as num).toDouble(),
        visibility: (map['visibility'] as num).toDouble(),
      );
}

/// Handedness type indicating left or right hand.
enum Handedness {
  /// Left hand.
  left,

  /// Right hand.
  right,
}

/// Hand landmark types for the MediaPipe hand landmark model.
///
/// Follows MediaPipe hand landmark topology with 21 landmarks for wrist and fingers.
///
/// Available landmarks (21 total):
/// - **Wrist**: [wrist] (0)
/// - **Thumb**: [thumbCMC] (1), [thumbMCP] (2), [thumbIP] (3), [thumbTip] (4)
/// - **Index**: [indexFingerMCP] (5), [indexFingerPIP] (6), [indexFingerDIP] (7), [indexFingerTip] (8)
/// - **Middle**: [middleFingerMCP] (9), [middleFingerPIP] (10), [middleFingerDIP] (11), [middleFingerTip] (12)
/// - **Ring**: [ringFingerMCP] (13), [ringFingerPIP] (14), [ringFingerDIP] (15), [ringFingerTip] (16)
/// - **Pinky**: [pinkyMCP] (17), [pinkyPIP] (18), [pinkyDIP] (19), [pinkyTip] (20)
///
/// Example:
/// ```dart
/// final hand = hands.first;
/// final wrist = hand.getLandmark(HandLandmarkType.wrist);
/// final indexTip = hand.getLandmark(HandLandmarkType.indexFingerTip);
///
/// if (wrist != null) {
///   print('Wrist at (${wrist.x}, ${wrist.y}) with visibility ${wrist.visibility}');
/// }
/// ```
enum HandLandmarkType {
  /// Wrist landmark (index 0).
  wrist,

  /// Thumb carpometacarpal joint landmark (index 1).
  thumbCMC,

  /// Thumb metacarpophalangeal joint landmark (index 2).
  thumbMCP,

  /// Thumb interphalangeal joint landmark (index 3).
  thumbIP,

  /// Thumb tip landmark (index 4).
  thumbTip,

  /// Index finger metacarpophalangeal joint landmark (index 5).
  indexFingerMCP,

  /// Index finger proximal interphalangeal joint landmark (index 6).
  indexFingerPIP,

  /// Index finger distal interphalangeal joint landmark (index 7).
  indexFingerDIP,

  /// Index finger tip landmark (index 8).
  indexFingerTip,

  /// Middle finger metacarpophalangeal joint landmark (index 9).
  middleFingerMCP,

  /// Middle finger proximal interphalangeal joint landmark (index 10).
  middleFingerPIP,

  /// Middle finger distal interphalangeal joint landmark (index 11).
  middleFingerDIP,

  /// Middle finger tip landmark (index 12).
  middleFingerTip,

  /// Ring finger metacarpophalangeal joint landmark (index 13).
  ringFingerMCP,

  /// Ring finger proximal interphalangeal joint landmark (index 14).
  ringFingerPIP,

  /// Ring finger distal interphalangeal joint landmark (index 15).
  ringFingerDIP,

  /// Ring finger tip landmark (index 16).
  ringFingerTip,

  /// Pinky metacarpophalangeal joint landmark (index 17).
  pinkyMCP,

  /// Pinky proximal interphalangeal joint landmark (index 18).
  pinkyPIP,

  /// Pinky distal interphalangeal joint landmark (index 19).
  pinkyDIP,

  /// Pinky tip landmark (index 20).
  pinkyTip,
}

/// Number of hand landmarks (21 for MediaPipe hand model).
const int numHandLandmarks = 21;

/// Defines the standard skeleton connections between hand landmarks.
///
/// Follows MediaPipe hand topology with 21 connections forming the hand skeleton.
/// Each connection is a pair of [HandLandmarkType] values representing
/// the start and end points of a line segment in the hand skeleton.
///
/// Use this constant to draw skeleton overlays on detected hands:
/// ```dart
/// for (final connection in handLandmarkConnections) {
///   final start = hand.getLandmark(connection[0]);
///   final end = hand.getLandmark(connection[1]);
///   if (start != null && end != null && start.visibility > 0.5 && end.visibility > 0.5) {
///     // Draw line from start to end
///     canvas.drawLine(
///       Offset(start.x, start.y),
///       Offset(end.x, end.y),
///       paint,
///     );
///   }
/// }
/// ```
const List<List<HandLandmarkType>> handLandmarkConnections = [
  [HandLandmarkType.wrist, HandLandmarkType.thumbCMC],
  [HandLandmarkType.thumbCMC, HandLandmarkType.thumbMCP],
  [HandLandmarkType.thumbMCP, HandLandmarkType.thumbIP],
  [HandLandmarkType.thumbIP, HandLandmarkType.thumbTip],
  [HandLandmarkType.wrist, HandLandmarkType.indexFingerMCP],
  [HandLandmarkType.indexFingerMCP, HandLandmarkType.indexFingerPIP],
  [HandLandmarkType.indexFingerPIP, HandLandmarkType.indexFingerDIP],
  [HandLandmarkType.indexFingerDIP, HandLandmarkType.indexFingerTip],
  [HandLandmarkType.indexFingerMCP, HandLandmarkType.middleFingerMCP],
  [HandLandmarkType.middleFingerMCP, HandLandmarkType.middleFingerPIP],
  [HandLandmarkType.middleFingerPIP, HandLandmarkType.middleFingerDIP],
  [HandLandmarkType.middleFingerDIP, HandLandmarkType.middleFingerTip],
  [HandLandmarkType.middleFingerMCP, HandLandmarkType.ringFingerMCP],
  [HandLandmarkType.ringFingerMCP, HandLandmarkType.ringFingerPIP],
  [HandLandmarkType.ringFingerPIP, HandLandmarkType.ringFingerDIP],
  [HandLandmarkType.ringFingerDIP, HandLandmarkType.ringFingerTip],
  [HandLandmarkType.ringFingerMCP, HandLandmarkType.pinkyMCP],
  [HandLandmarkType.pinkyMCP, HandLandmarkType.pinkyPIP],
  [HandLandmarkType.pinkyPIP, HandLandmarkType.pinkyDIP],
  [HandLandmarkType.pinkyDIP, HandLandmarkType.pinkyTip],
  [HandLandmarkType.wrist, HandLandmarkType.pinkyMCP],
];

/// Detected hand with bounding box and optional landmarks.
///
/// This is the main result returned by [HandDetector.detect()].
///
/// Contains:
/// - [boundingBox]: Location of the detected hand in the image
/// - [score]: Confidence score from the hand detector (0.0 to 1.0)
/// - [landmarks]: List of 21 keypoints (empty if [HandMode.boxes])
/// - [handedness]: Whether this is a left or right hand (null if not determined)
/// - [gesture]: Recognized gesture (null if gesture recognition disabled)
/// - [imageWidth] and [imageHeight]: Original image dimensions for coordinate reference
///
/// Example:
/// ```dart
/// final hands = await detector.detect(imageBytes);
/// for (final hand in hands) {
///   print('Hand detected with confidence ${hand.score}');
///   print('Handedness: ${hand.handedness}');
///   if (hand.gesture != null) {
///     print('Gesture: ${hand.gesture!.type} (${hand.gesture!.confidence})');
///   }
///   if (hand.hasLandmarks) {
///     final wrist = hand.getLandmark(HandLandmarkType.wrist);
///     print('Wrist at (${wrist?.x}, ${wrist?.y})');
///   }
/// }
/// ```
class Hand {
  /// Bounding box of the detected hand in pixel coordinates
  final BoundingBox boundingBox;

  /// Confidence score from hand detector (0.0 to 1.0)
  final double score;

  /// List of 21 landmarks. Empty if using [HandMode.boxes].
  final List<HandLandmark> landmarks;

  /// Width of the original image in pixels
  final int imageWidth;

  /// Height of the original image in pixels
  final int imageHeight;

  /// Handedness of the detected hand (left or right).
  /// May be null if handedness detection is not available.
  final Handedness? handedness;

  /// Rotation angle in radians from palm detection.
  /// Used to draw the rotated bounding box that matches the hand orientation.
  /// May be null if rotation data is not preserved.
  final double? rotation;

  /// Center X coordinate of the rotated rectangle in pixels.
  /// May be null if rotation data is not preserved.
  final double? rotatedCenterX;

  /// Center Y coordinate of the rotated rectangle in pixels.
  /// May be null if rotation data is not preserved.
  final double? rotatedCenterY;

  /// Size of the rotated rectangle in pixels.
  /// May be null if rotation data is not preserved.
  final double? rotatedSize;

  /// Recognized gesture for this hand.
  /// Null if gesture recognition is disabled or not available.
  final GestureResult? gesture;

  /// Creates a detected hand with bounding box, landmarks, and image dimensions.
  const Hand({
    required this.boundingBox,
    required this.score,
    required this.landmarks,
    required this.imageWidth,
    required this.imageHeight,
    this.handedness,
    this.rotation,
    this.rotatedCenterX,
    this.rotatedCenterY,
    this.rotatedSize,
    this.gesture,
  });

  /// Serializes this hand to a map for cross-isolate transfer.
  Map<String, dynamic> toMap() => {
        'boundingBox': boundingBox.toMap(),
        'score': score,
        'landmarks': landmarks.map((l) => l.toMap()).toList(),
        'imageWidth': imageWidth,
        'imageHeight': imageHeight,
        'handedness': handedness?.name,
        'rotation': rotation,
        'rotatedCenterX': rotatedCenterX,
        'rotatedCenterY': rotatedCenterY,
        'rotatedSize': rotatedSize,
        'gesture': gesture?.toMap(),
      };

  /// Deserializes a hand from a map.
  static Hand fromMap(Map<String, dynamic> map) => Hand(
        boundingBox:
            BoundingBox.fromMap(map['boundingBox'] as Map<String, dynamic>),
        score: (map['score'] as num).toDouble(),
        landmarks: (map['landmarks'] as List<dynamic>)
            .map((l) => HandLandmark.fromMap(l as Map<String, dynamic>))
            .toList(),
        imageWidth: map['imageWidth'] as int,
        imageHeight: map['imageHeight'] as int,
        handedness: map['handedness'] != null
            ? Handedness.values.firstWhere((e) => e.name == map['handedness'])
            : null,
        rotation: (map['rotation'] as num?)?.toDouble(),
        rotatedCenterX: (map['rotatedCenterX'] as num?)?.toDouble(),
        rotatedCenterY: (map['rotatedCenterY'] as num?)?.toDouble(),
        rotatedSize: (map['rotatedSize'] as num?)?.toDouble(),
        gesture: map['gesture'] != null
            ? GestureResult.fromMap(map['gesture'] as Map<String, dynamic>)
            : null,
      );

  /// Gets a specific landmark by type, or null if not found
  HandLandmark? getLandmark(HandLandmarkType type) {
    for (final l in landmarks) {
      if (l.type == type) return l;
    }
    return null;
  }

  /// Returns true if this hand has landmarks
  bool get hasLandmarks => landmarks.isNotEmpty;

  /// Returns true if this hand has a recognized gesture
  bool get hasGesture => gesture != null;

  @override
  String toString() {
    final String landmarksInfo = landmarks
        .map((l) =>
            '${l.type.name}: (${l.x.toStringAsFixed(2)}, ${l.y.toStringAsFixed(2)}) vis=${l.visibility.toStringAsFixed(2)}')
        .join('\n');
    final String gestureInfo = gesture != null
        ? '  gesture=${gesture!.type.name} (${gesture!.confidence.toStringAsFixed(3)}),\n'
        : '';
    return 'Hand(\n'
        '  score=${score.toStringAsFixed(3)},\n'
        '$gestureInfo'
        '  landmarks=${landmarks.length},\n'
        '  coords:\n$landmarksInfo\n)';
  }
}
