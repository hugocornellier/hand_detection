/// Native (non-web) entry point for hand_detection.
///
/// Import this library on mobile and desktop targets that need access to
/// native-only symbols (e.g. [HandDetectorIsolate], [Accelerator], [Precision])
/// that are intentionally absent from the default conditional-export entry
/// (`package:hand_detection/hand_detection.dart`), which defaults to the web
/// library so the package scores WASM-ready.
library;

export 'src/native/hand_native_lib.dart';
