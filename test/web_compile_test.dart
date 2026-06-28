@TestOn('browser')
library;

import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/src/native/hand_native_lib.dart';

/// Web-target compile + smoke test.
///
/// Running this under `flutter test --platform chrome` forces the package's
/// conditional exports to resolve to the web (`dart.library.js_interop`)
/// implementation and compiles every web file (LiteRT.js models + Canvas
/// pipeline) for the JS target. It does not run inference (LiteRT.js is not
/// loaded in the headless harness), only that the web build is sound and the
/// public surface instantiates.
void main() {
  test('web HandDetector compiles and instantiates', () {
    final detector = HandDetector();
    expect(detector.isReady, isFalse);
    expect(detector.isInitialized, isFalse);
  });

  test('shared public types are available on web', () {
    expect(numHandLandmarks, 21);
    expect(HandLandmarkType.values.length, 21);
    expect(handLandmarkConnections, isNotEmpty);
    final box = BoundingBox.ltrb(0, 0, 1, 1);
    expect(box.right, 1);
  });

  test('native-only methods throw UnsupportedError on web', () {
    final detector = HandDetector();
    expect(() => detector.detectFromFilepath('x.png'), throwsUnsupportedError);
  });
}
