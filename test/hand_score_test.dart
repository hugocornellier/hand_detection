import 'package:flutter_test/flutter_test.dart';
import 'package:hand_detection/src/shared/hand_score.dart';

void main() {
  group('decodeHandPresenceProbability', () {
    test('preserves the model logistic probability without a second sigmoid',
        () {
      expect(decodeHandPresenceProbability(0), 0);
      expect(decodeHandPresenceProbability(0.25), 0.25);
      expect(decodeHandPresenceProbability(0.5), 0.5);
      expect(decodeHandPresenceProbability(0.99), 0.99);
      expect(decodeHandPresenceProbability(1), 1);
    });

    test('clamps delegate drift and rejects non-finite values', () {
      expect(decodeHandPresenceProbability(-0.001), 0);
      expect(decodeHandPresenceProbability(1.001), 1);
      expect(decodeHandPresenceProbability(double.nan), 0);
      expect(decodeHandPresenceProbability(double.infinity), 0);
      expect(decodeHandPresenceProbability(double.negativeInfinity), 0);
    });
  });
}
