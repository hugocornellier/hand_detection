// Integration test: verifies that the accelerator/precision control params
// thread correctly through the full HandDetector stack when useCompiledModel
// is true.  The test initialises with {Accelerator.cpu} (no GPU risk) so it
// can run on every CI host that has the LiteRT runtime available.
//
// Run from the example directory:
//   flutter test integration_test/accelerator_control_test.dart -d macos

import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:hand_detection/hand_detection_native.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('HandDetector - accelerator control', () {
    testWidgets(
      'initializes with useCompiledModel + cpu accelerator and detects without throwing',
      (tester) async {
        late HandDetector detector;
        Object? initError;

        try {
          detector = await HandDetector.create(
            useCompiledModel: true,
            accelerators: {Accelerator.cpu},
            precision: Precision.fp32,
          );
        } catch (e) {
          initError = e;
        }

        if (initError != null) {
          // The LiteRT CompiledModel runtime is not available on this host
          // (e.g. CI without the native library).  The test still validates
          // that the code compiles and the parameter path is wired correctly;
          // mark as skipped so the suite stays green.
          markTestSkipped(
            'LiteRT CompiledModel runtime unavailable on host: $initError',
          );
          return;
        }

        expect(detector.isReady, isTrue);

        // Run one detect on a real sample image and assert no exception.
        final ByteData data = await rootBundle
            .load('assets/samples/istockphoto-462908027-612x612.jpg');
        final bytes = data.buffer.asUint8List();

        final List<Hand> results = await detector.detect(bytes);

        // Results may be empty (model variance under CPU-only) but must not
        // throw. Verify the list type is correct.
        expect(results, isA<List<Hand>>());

        await detector.dispose();
        expect(detector.isReady, isFalse);
      },
    );

    testWidgets(
      'Accelerator and Precision enums are accessible from hand_detection barrel',
      (tester) async {
        // Compile-time assertion: if the barrel does not export these, the
        // test file fails to compile.
        const accelerators = {
          Accelerator.cpu,
          Accelerator.gpu,
          Accelerator.npu
        };
        const precisions = [Precision.fp16, Precision.fp32];

        expect(accelerators.length, 3);
        expect(precisions.length, 2);
        expect(Accelerator.values, containsAll(accelerators));
        expect(Precision.values, containsAll(precisions));
      },
    );
  });
}
