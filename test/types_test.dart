import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/hand_detection.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('HandDetector model version', () {
    test('default modelVersion is non-empty and package-scoped', () {
      expect(HandDetector.modelVersion, isNotEmpty);
      expect(HandDetector.modelVersion, contains('hand_detection:2.2.0'));
      expect(HandDetector.modelVersion, HandDetector.modelVersionFor());
    });

    test('modelVersionFor changes with detector configuration', () {
      final base = HandDetector.modelVersionFor();

      expect(
        HandDetector.modelVersionFor(mode: HandMode.boxes),
        isNot(base),
      );
      expect(
        HandDetector.modelVersionFor(enableGestures: true),
        isNot(base),
      );
    });
  });

  group('PerformanceConfig', () {
    test('disabled mode has disabled PerformanceMode', () {
      const config = PerformanceConfig(mode: PerformanceMode.disabled);
      expect(config.mode, PerformanceMode.disabled);
    });

    test('disabled static constant has disabled mode', () {
      expect(PerformanceConfig.disabled.mode, PerformanceMode.disabled);
    });

    test('xnnpack with null threads', () {
      const config = PerformanceConfig.xnnpack();
      expect(config.mode, PerformanceMode.xnnpack);
      expect(config.numThreads, isNull);
    });

    test('auto with null threads', () {
      const config = PerformanceConfig.auto();
      expect(config.mode, PerformanceMode.auto);
      expect(config.numThreads, isNull);
    });

    test('xnnpack with explicit threads stores value', () {
      const config = PerformanceConfig.xnnpack(numThreads: 4);
      expect(config.numThreads, 4);
    });

    test('default constructor uses auto mode', () {
      const config = PerformanceConfig();
      expect(config.mode, PerformanceMode.auto);
      expect(config.numThreads, isNull);
    });
  });

  group('PerformanceMode enum', () {
    test('has 5 values', () {
      expect(PerformanceMode.values.length, 5);
    });

    test('values in expected order', () {
      expect(PerformanceMode.disabled.index, 0);
      expect(PerformanceMode.xnnpack.index, 1);
      expect(PerformanceMode.gpu.index, 2);
      expect(PerformanceMode.coreml.index, 3);
      expect(PerformanceMode.auto.index, 4);
    });
  });

  group('HandMode enum', () {
    test('has 2 values', () {
      expect(HandMode.values.length, 2);
    });

    test('values in expected order', () {
      expect(HandMode.boxes.index, 0);
      expect(HandMode.boxesAndLandmarks.index, 1);
    });
  });

  group('HandLandmarkModel enum', () {
    test('has 1 value', () {
      expect(HandLandmarkModel.values.length, 1);
    });

    test('full is at index 0', () {
      expect(HandLandmarkModel.full.index, 0);
    });
  });

  group('GestureType enum', () {
    test('has 8 values', () {
      expect(GestureType.values.length, 8);
    });

    test('values in expected order matching model output', () {
      expect(GestureType.unknown.index, 0);
      expect(GestureType.closedFist.index, 1);
      expect(GestureType.openPalm.index, 2);
      expect(GestureType.pointingUp.index, 3);
      expect(GestureType.thumbDown.index, 4);
      expect(GestureType.thumbUp.index, 5);
      expect(GestureType.victory.index, 6);
      expect(GestureType.iLoveYou.index, 7);
    });
  });

  group('HandLandmarkType enum', () {
    test('has 21 values', () {
      expect(HandLandmarkType.values.length, 21);
    });

    test('indices match MediaPipe topology', () {
      expect(HandLandmarkType.wrist.index, 0);
      expect(HandLandmarkType.thumbCMC.index, 1);
      expect(HandLandmarkType.thumbMCP.index, 2);
      expect(HandLandmarkType.thumbIP.index, 3);
      expect(HandLandmarkType.thumbTip.index, 4);
      expect(HandLandmarkType.indexFingerMCP.index, 5);
      expect(HandLandmarkType.indexFingerPIP.index, 6);
      expect(HandLandmarkType.indexFingerDIP.index, 7);
      expect(HandLandmarkType.indexFingerTip.index, 8);
      expect(HandLandmarkType.middleFingerMCP.index, 9);
      expect(HandLandmarkType.middleFingerPIP.index, 10);
      expect(HandLandmarkType.middleFingerDIP.index, 11);
      expect(HandLandmarkType.middleFingerTip.index, 12);
      expect(HandLandmarkType.ringFingerMCP.index, 13);
      expect(HandLandmarkType.ringFingerPIP.index, 14);
      expect(HandLandmarkType.ringFingerDIP.index, 15);
      expect(HandLandmarkType.ringFingerTip.index, 16);
      expect(HandLandmarkType.pinkyMCP.index, 17);
      expect(HandLandmarkType.pinkyPIP.index, 18);
      expect(HandLandmarkType.pinkyDIP.index, 19);
      expect(HandLandmarkType.pinkyTip.index, 20);
    });
  });

  group('GestureResult', () {
    test('toMap/fromMap round-trip', () {
      const original =
          GestureResult(type: GestureType.thumbUp, confidence: 0.95);
      final map = original.toMap();
      final restored = GestureResult.fromMap(map);

      expect(restored.type, GestureType.thumbUp);
      expect(restored.confidence, 0.95);
    });

    test('toMap produces expected keys', () {
      const gesture = GestureResult(type: GestureType.victory, confidence: 0.8);
      final map = gesture.toMap();

      expect(map['type'], 'victory');
      expect(map['confidence'], 0.8);
    });

    test('fromMap handles integer confidence', () {
      final map = {'type': 'closedFist', 'confidence': 1};
      final gesture = GestureResult.fromMap(map);

      expect(gesture.type, GestureType.closedFist);
      expect(gesture.confidence, 1.0);
    });

    test('toString formats correctly', () {
      const gesture =
          GestureResult(type: GestureType.openPalm, confidence: 0.875);
      final str = gesture.toString();

      expect(str, contains('openPalm'));
      expect(str, contains('0.875'));
    });

    test('round-trip all gesture types', () {
      for (final type in GestureType.values) {
        final original = GestureResult(type: type, confidence: 0.5);
        final restored = GestureResult.fromMap(original.toMap());
        expect(restored.type, type);
      }
    });
  });

  group('HandLandmark', () {
    test('toMap/fromMap round-trip', () {
      final original = HandLandmark(
        type: HandLandmarkType.wrist,
        x: 100.5,
        y: 200.3,
        z: -0.05,
        visibility: 0.95,
      );
      final map = original.toMap();
      final restored = HandLandmark.fromMap(map);

      expect(restored.type, HandLandmarkType.wrist);
      expect(restored.x, 100.5);
      expect(restored.y, 200.3);
      expect(restored.z, -0.05);
      expect(restored.visibility, 0.95);
    });

    test('toMap produces expected keys', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.thumbTip,
        x: 10.0,
        y: 20.0,
        z: 0.1,
        visibility: 0.8,
      );
      final map = landmark.toMap();

      expect(map['type'], 'thumbTip');
      expect(map['x'], 10.0);
      expect(map['y'], 20.0);
      expect(map['z'], 0.1);
      expect(map['visibility'], 0.8);
    });

    test('fromMap handles integer coordinates', () {
      final map = {
        'type': 'indexFingerTip',
        'x': 100,
        'y': 200,
        'z': 0,
        'visibility': 1,
      };
      final landmark = HandLandmark.fromMap(map);

      expect(landmark.type, HandLandmarkType.indexFingerTip);
      expect(landmark.x, 100.0);
      expect(landmark.y, 200.0);
      expect(landmark.z, 0.0);
      expect(landmark.visibility, 1.0);
    });

    test('xNorm clamps to [0, 1]', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.wrist,
        x: -10.0,
        y: 500.0,
        z: 0.0,
        visibility: 1.0,
      );

      expect(landmark.xNorm(100), 0.0); // negative clamped to 0
      expect(landmark.yNorm(100), 1.0); // >1 clamped to 1
    });

    test('xNorm and yNorm compute correctly for in-range values', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.wrist,
        x: 320.0,
        y: 240.0,
        z: 0.0,
        visibility: 1.0,
      );

      expect(landmark.xNorm(640), closeTo(0.5, 0.0001));
      expect(landmark.yNorm(480), closeTo(0.5, 0.0001));
    });

    test('toPixel returns double coordinates', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.thumbTip,
        x: 123.7,
        y: 456.9,
        z: 0.0,
        visibility: 1.0,
      );

      final point = landmark.toPixel(640, 480);
      expect(point.x, 123.7);
      expect(point.y, 456.9);
    });

    test('round-trip all landmark types', () {
      for (final type in HandLandmarkType.values) {
        final original = HandLandmark(
          type: type,
          x: 50.0,
          y: 50.0,
          z: 0.1,
          visibility: 0.9,
        );
        final restored = HandLandmark.fromMap(original.toMap());
        expect(restored.type, type);
      }
    });

    test('negative z value is valid and preserved', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.wrist,
        x: 100.0,
        y: 100.0,
        z: -0.75,
        visibility: 0.9,
      );
      expect(landmark.z, -0.75);
      final restored = HandLandmark.fromMap(landmark.toMap());
      expect(restored.z, -0.75);
    });

    test('z value greater than 1.0 is valid and preserved', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.thumbTip,
        x: 100.0,
        y: 100.0,
        z: 2.5,
        visibility: 0.8,
      );
      expect(landmark.z, 2.5);
      final restored = HandLandmark.fromMap(landmark.toMap());
      expect(restored.z, 2.5);
    });

    test('visibility exactly 0.0 is preserved', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.pinkyTip,
        x: 10.0,
        y: 10.0,
        z: 0.0,
        visibility: 0.0,
      );
      expect(landmark.visibility, 0.0);
      final restored = HandLandmark.fromMap(landmark.toMap());
      expect(restored.visibility, 0.0);
    });

    test('visibility exactly 1.0 is preserved', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.indexFingerTip,
        x: 10.0,
        y: 10.0,
        z: 0.0,
        visibility: 1.0,
      );
      expect(landmark.visibility, 1.0);
      final restored = HandLandmark.fromMap(landmark.toMap());
      expect(restored.visibility, 1.0);
    });

    test('very large coordinate values are preserved', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.wrist,
        x: 10000.0,
        y: 10000.0,
        z: 0.0,
        visibility: 0.9,
      );
      expect(landmark.x, 10000.0);
      expect(landmark.y, 10000.0);
      final restored = HandLandmark.fromMap(landmark.toMap());
      expect(restored.x, 10000.0);
      expect(restored.y, 10000.0);
    });

    test('xNorm with imageWidth = 1 clamps out-of-range x', () {
      final inRange = HandLandmark(
        type: HandLandmarkType.wrist,
        x: 0.5,
        y: 0.5,
        z: 0.0,
        visibility: 1.0,
      );
      expect(inRange.xNorm(1), closeTo(0.5, 0.0001));

      final over = HandLandmark(
        type: HandLandmarkType.wrist,
        x: 2.0,
        y: 0.0,
        z: 0.0,
        visibility: 1.0,
      );
      expect(over.xNorm(1), 1.0);

      final under = HandLandmark(
        type: HandLandmarkType.wrist,
        x: -1.0,
        y: 0.0,
        z: 0.0,
        visibility: 1.0,
      );
      expect(under.xNorm(1), 0.0);
    });

    test('yNorm with imageHeight = 1 clamps out-of-range y', () {
      final inRange = HandLandmark(
        type: HandLandmarkType.wrist,
        x: 0.0,
        y: 0.5,
        z: 0.0,
        visibility: 1.0,
      );
      expect(inRange.yNorm(1), closeTo(0.5, 0.0001));

      final over = HandLandmark(
        type: HandLandmarkType.wrist,
        x: 0.0,
        y: 2.0,
        z: 0.0,
        visibility: 1.0,
      );
      expect(over.yNorm(1), 1.0);

      final under = HandLandmark(
        type: HandLandmarkType.wrist,
        x: 0.0,
        y: -1.0,
        z: 0.0,
        visibility: 1.0,
      );
      expect(under.yNorm(1), 0.0);
    });

    test('toPixel with negative x and y preserves doubles', () {
      final landmark = HandLandmark(
        type: HandLandmarkType.wrist,
        x: -10.9,
        y: -5.3,
        z: 0.0,
        visibility: 1.0,
      );
      final point = landmark.toPixel(640, 480);
      expect(point.x, -10.9);
      expect(point.y, -5.3);
    });

    test('fromMap ignores extra fields in map', () {
      final map = {
        'type': 'wrist',
        'x': 50.0,
        'y': 60.0,
        'z': 0.1,
        'visibility': 0.95,
        'extraField': 'should be ignored',
        'anotherExtra': 42,
      };
      final landmark = HandLandmark.fromMap(map);
      expect(landmark.type, HandLandmarkType.wrist);
      expect(landmark.x, 50.0);
      expect(landmark.y, 60.0);
      expect(landmark.z, 0.1);
      expect(landmark.visibility, 0.95);
    });
  });

  group('BoundingBox', () {
    test('toMap/fromMap round-trip', () {
      final original = BoundingBox.ltrb(10.5, 20.3, 100.7, 200.1);
      final map = original.toMap();
      final restored = BoundingBox.fromMap(map);

      expect(restored.left, 10.5);
      expect(restored.top, 20.3);
      expect(restored.right, 100.7);
      expect(restored.bottom, 200.1);
    });

    test('toMap produces corner-based keys', () {
      final bbox = BoundingBox.ltrb(0.0, 0.0, 100.0, 100.0);
      final map = bbox.toMap();

      expect(map.containsKey('topLeft'), true);
      expect(map.containsKey('topRight'), true);
      expect(map.containsKey('bottomRight'), true);
      expect(map.containsKey('bottomLeft'), true);
    });

    test('fromMap restores correct LTRB values', () {
      final original = BoundingBox.ltrb(0.0, 0.0, 100.0, 100.0);
      final restored = BoundingBox.fromMap(original.toMap());

      expect(restored.left, 0.0);
      expect(restored.right, 100.0);
      expect(restored.top, 0.0);
      expect(restored.bottom, 100.0);
    });

    test('negative coordinates are stored as-is', () {
      final bbox = BoundingBox.ltrb(-50.0, -30.0, -10.0, -5.0);
      expect(bbox.left, -50.0);
      expect(bbox.top, -30.0);
      expect(bbox.right, -10.0);
      expect(bbox.bottom, -5.0);
    });

    test('negative coordinates round-trip via toMap/fromMap', () {
      final original = BoundingBox.ltrb(-100.0, -80.0, -20.0, -10.0);
      final restored = BoundingBox.fromMap(original.toMap());
      expect(restored.left, -100.0);
      expect(restored.top, -80.0);
      expect(restored.right, -20.0);
      expect(restored.bottom, -10.0);
    });

    test('zero-size box (left == right and top == bottom) is preserved', () {
      final bbox = BoundingBox.ltrb(50.0, 50.0, 50.0, 50.0);
      expect(bbox.left, 50.0);
      expect(bbox.top, 50.0);
      expect(bbox.right, 50.0);
      expect(bbox.bottom, 50.0);
      final restored = BoundingBox.fromMap(bbox.toMap());
      expect(restored.left, restored.right);
      expect(restored.top, restored.bottom);
    });
  });

  group('Point', () {
    test('stores x and y coordinates', () {
      final point = Point(42, 99);
      expect(point.x, 42);
      expect(point.y, 99);
    });

    test('handles negative coordinates', () {
      final point = Point(-1, -1);
      expect(point.x, -1);
      expect(point.y, -1);
    });

    test('handles zero coordinates', () {
      final point = Point(0, 0);
      expect(point.x, 0);
      expect(point.y, 0);
    });
  });

  group('HandLandmarks', () {
    test('stores all fields', () {
      final landmarks = [
        HandLandmark(
          type: HandLandmarkType.wrist,
          x: 10.0,
          y: 20.0,
          z: 0.0,
          visibility: 0.9,
        ),
      ];
      final worldLandmarks = [
        HandLandmark(
          type: HandLandmarkType.wrist,
          x: 0.1,
          y: 0.2,
          z: 0.3,
          visibility: 0.9,
        ),
      ];

      final result = HandLandmarks(
        landmarks: landmarks,
        worldLandmarks: worldLandmarks,
        score: 0.85,
        handedness: Handedness.right,
      );

      expect(result.landmarks.length, 1);
      expect(result.worldLandmarks.length, 1);
      expect(result.score, 0.85);
      expect(result.handedness, Handedness.right);
    });
  });

  group('Hand serialization', () {
    Hand createFullHand() {
      return Hand(
        boundingBox: BoundingBox.ltrb(10, 20, 200, 300),
        score: 0.95,
        landmarks: [
          HandLandmark(
            type: HandLandmarkType.wrist,
            x: 100.0,
            y: 200.0,
            z: -0.05,
            visibility: 0.9,
          ),
          HandLandmark(
            type: HandLandmarkType.thumbTip,
            x: 150.0,
            y: 180.0,
            z: 0.1,
            visibility: 0.85,
          ),
        ],
        imageWidth: 640,
        imageHeight: 480,
        handedness: Handedness.right,
        rotation: 0.25,
        rotatedCenterX: 105.0,
        rotatedCenterY: 250.0,
        rotatedSize: 190.0,
        gesture:
            const GestureResult(type: GestureType.thumbUp, confidence: 0.9),
      );
    }

    test('toMap/fromMap round-trip with all fields', () {
      final original = createFullHand();
      final map = original.toMap();
      final restored = Hand.fromMap(map);

      expect(restored.boundingBox.left, original.boundingBox.left);
      expect(restored.boundingBox.top, original.boundingBox.top);
      expect(restored.boundingBox.right, original.boundingBox.right);
      expect(restored.boundingBox.bottom, original.boundingBox.bottom);
      expect(restored.score, original.score);
      expect(restored.landmarks.length, original.landmarks.length);
      expect(restored.imageWidth, original.imageWidth);
      expect(restored.imageHeight, original.imageHeight);
      expect(restored.handedness, Handedness.right);
      expect(restored.rotation, 0.25);
      expect(restored.rotatedCenterX, 105.0);
      expect(restored.rotatedCenterY, 250.0);
      expect(restored.rotatedSize, 190.0);
      expect(restored.gesture!.type, GestureType.thumbUp);
      expect(restored.gesture!.confidence, 0.9);
    });

    test('toMap/fromMap with null handedness', () {
      final hand = Hand(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.8,
        landmarks: [],
        imageWidth: 640,
        imageHeight: 480,
      );

      final restored = Hand.fromMap(hand.toMap());
      expect(restored.handedness, isNull);
    });

    test('toMap/fromMap with null gesture', () {
      final hand = Hand(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.8,
        landmarks: [],
        imageWidth: 640,
        imageHeight: 480,
        handedness: Handedness.left,
      );

      final restored = Hand.fromMap(hand.toMap());
      expect(restored.gesture, isNull);
      expect(restored.hasGesture, false);
    });

    test('toMap/fromMap with null rotation fields', () {
      final hand = Hand(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.8,
        landmarks: [],
        imageWidth: 640,
        imageHeight: 480,
      );

      final restored = Hand.fromMap(hand.toMap());
      expect(restored.rotation, isNull);
      expect(restored.rotatedCenterX, isNull);
      expect(restored.rotatedCenterY, isNull);
      expect(restored.rotatedSize, isNull);
    });

    test('toMap serializes landmarks correctly', () {
      final hand = createFullHand();
      final map = hand.toMap();

      final landmarksList = map['landmarks'] as List;
      expect(landmarksList.length, 2);

      final first = landmarksList[0] as Map<String, dynamic>;
      expect(first['type'], 'wrist');
      expect(first['x'], 100.0);
    });

    test('hasGesture returns true when gesture present', () {
      final hand = createFullHand();
      expect(hand.hasGesture, true);
    });

    test('hasLandmarks returns true when landmarks present', () {
      final hand = createFullHand();
      expect(hand.hasLandmarks, true);
    });

    test('hasLandmarks returns false for empty landmarks', () {
      final hand = Hand(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.8,
        landmarks: [],
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(hand.hasLandmarks, false);
    });

    test('getLandmark returns null for missing type', () {
      final hand = Hand(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.8,
        landmarks: [],
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(hand.getLandmark(HandLandmarkType.wrist), isNull);
    });

    test('getLandmark finds landmark by type', () {
      final hand = createFullHand();
      final wrist = hand.getLandmark(HandLandmarkType.wrist);
      expect(wrist, isNotNull);
      expect(wrist!.x, 100.0);
    });

    test('getLandmark returns null for type not in landmarks', () {
      final hand = createFullHand();
      // Only wrist and thumbTip are in this hand
      final pinky = hand.getLandmark(HandLandmarkType.pinkyTip);
      expect(pinky, isNull);
    });

    test('toString includes gesture info when present', () {
      final hand = createFullHand();
      final str = hand.toString();
      expect(str, contains('gesture=thumbUp'));
    });

    test('toString omits gesture info when null', () {
      final hand = Hand(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.8,
        landmarks: [],
        imageWidth: 640,
        imageHeight: 480,
      );
      final str = hand.toString();
      expect(str, isNot(contains('gesture=')));
    });

    test('toString includes score and landmark count', () {
      final hand = createFullHand();
      final str = hand.toString();
      expect(str, contains('Hand('));
      expect(str, contains('score='));
      expect(str, contains('landmarks=2'));
    });

    test('toString with rotation data present includes score and landmarks',
        () {
      // createFullHand already has rotation, rotatedCenterX/Y, rotatedSize set.
      // Verify toString still produces well-formed output with those fields present.
      final hand = createFullHand();
      expect(hand.rotation, isNotNull);
      expect(hand.rotatedCenterX, isNotNull);
      expect(hand.rotatedCenterY, isNotNull);
      expect(hand.rotatedSize, isNotNull);
      final str = hand.toString();
      expect(str, contains('Hand('));
      expect(str, contains('score=0.950'));
      expect(str, contains('landmarks=2'));
      expect(str, contains('gesture=thumbUp'));
    });

    test('toString with null rotation does not throw', () {
      final hand = Hand(
        boundingBox: BoundingBox.ltrb(0, 0, 100, 100),
        score: 0.5,
        landmarks: [],
        imageWidth: 640,
        imageHeight: 480,
      );
      expect(hand.rotation, isNull);
      expect(hand.rotatedCenterX, isNull);
      expect(hand.rotatedCenterY, isNull);
      expect(hand.rotatedSize, isNull);
      final str = hand.toString();
      expect(str, contains('Hand('));
      expect(str, contains('score=0.500'));
      expect(str, contains('landmarks=0'));
    });
  });

  group('handLandmarkConnections', () {
    test('has 21 connections', () {
      expect(handLandmarkConnections.length, 21);
    });

    test('each connection has exactly 2 endpoints', () {
      for (final connection in handLandmarkConnections) {
        expect(connection.length, 2);
      }
    });

    test('all endpoints are valid HandLandmarkType values', () {
      final allTypes = HandLandmarkType.values.toSet();
      for (final connection in handLandmarkConnections) {
        expect(allTypes.contains(connection[0]), true,
            reason: 'Invalid start: ${connection[0]}');
        expect(allTypes.contains(connection[1]), true,
            reason: 'Invalid end: ${connection[1]}');
      }
    });

    test('wrist connects to thumb, index, and pinky', () {
      final wristConnections = handLandmarkConnections
          .where((c) =>
              c[0] == HandLandmarkType.wrist || c[1] == HandLandmarkType.wrist)
          .toList();
      // Wrist connects to: thumbCMC, indexFingerMCP, pinkyMCP
      expect(wristConnections.length, 3);
    });

    test('forms a connected skeleton (every landmark is reachable)', () {
      final connected = <HandLandmarkType>{HandLandmarkType.wrist};
      bool changed = true;
      while (changed) {
        changed = false;
        for (final connection in handLandmarkConnections) {
          if (connected.contains(connection[0]) &&
              !connected.contains(connection[1])) {
            connected.add(connection[1]);
            changed = true;
          }
          if (connected.contains(connection[1]) &&
              !connected.contains(connection[0])) {
            connected.add(connection[0]);
            changed = true;
          }
        }
      }
      expect(connected.length, 21,
          reason: 'Not all landmarks are connected in the skeleton');
    });
  });

  group('numHandLandmarks constant', () {
    test('equals 21', () {
      expect(numHandLandmarks, 21);
    });

    test('matches HandLandmarkType.values.length', () {
      expect(numHandLandmarks, HandLandmarkType.values.length);
    });
  });

  group('Handedness enum', () {
    test('has left and right', () {
      expect(Handedness.values.length, 2);
      expect(Handedness.left.index, 0);
      expect(Handedness.right.index, 1);
    });

    test('name property works', () {
      expect(Handedness.left.name, 'left');
      expect(Handedness.right.name, 'right');
    });
  });
}
