import 'package:flutter_web_plugins/flutter_web_plugins.dart';

/// Web plugin registration for hand_detection.
///
/// Referenced from `pubspec.yaml`'s `flutter.plugin.platforms.web` block. The
/// package uses conditional exports rather than method channels, so this is
/// intentionally a no-op.
class HandDetectionWeb {
  /// Registers the web implementation with Flutter's plugin registrar.
  static void registerWith(Registrar registrar) {
    // No-op; conditional exports drive the web implementation.
  }
}
