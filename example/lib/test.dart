import 'dart:io';
import 'dart:typed_data';
import 'package:hand_detection/hand_detection.dart';

Future main() async {
  // 1. initialize
  final HandDetector detector = HandDetector(
    mode: HandMode.boxesAndLandmarks,
    landmarkModel: HandLandmarkModel.full,
  );
  await detector.initialize();

  // 2. detect
  final Uint8List imageBytes = await File('path/to/image.jpg').readAsBytes();
  final List<Hand> results = await detector.detect(imageBytes);

  // 3. access results
  for (final Hand hand in results) {
    final BoundingBox bbox = hand.boundingBox;
    stdout.writeln(
        'Bounding box: (${bbox.left}, ${bbox.top}) → (${bbox.right}, ${bbox.bottom})');

    if (hand.hasLandmarks) {
      // iterate through landmarks
      for (final HandLandmark lm in hand.landmarks) {
        stdout.writeln(
          '${lm.type}: (${lm.x.toStringAsFixed(1)}, ${lm.y.toStringAsFixed(1)}) '
          'vis=${lm.visibility.toStringAsFixed(2)}',
        );
      }

      // access individual landmarks
      final HandLandmark? wrist = hand.getLandmark(HandLandmarkType.wrist);
      if (wrist != null) {
        stdout.writeln(
            'Wrist visibility: ${wrist.visibility.toStringAsFixed(2)}');
      }
    }
  }

  // 4. clean-up
  await detector.dispose();
}
