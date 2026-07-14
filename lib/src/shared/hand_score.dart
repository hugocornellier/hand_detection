/// Decodes the hand-presence output of the bundled landmark model.
///
/// The model graph already ends this output with a LOGISTIC operator, so the
/// value is a probability in `[0, 1]`. Applying sigmoid again would compress
/// the useful range to `[0.5, 0.731...]` and make the default `0.5` presence
/// threshold accept non-hand crops. Clamp only to absorb minor delegate
/// floating-point drift, and fail closed for a non-finite result.
double decodeHandPresenceProbability(double modelOutput) {
  if (!modelOutput.isFinite) return 0;
  return modelOutput.clamp(0.0, 1.0).toDouble();
}
