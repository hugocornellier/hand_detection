// Conditional re-export of the small slice of `opencv_dart` that the public
// API surfaces (so `detectFromMat` callers can build a `Mat`).
//
// Native platforms get the real `opencv_dart` symbols; web gets nothing
// (opencv_dart has no web support); other/unsupported platforms also get an
// empty stub.
export 'opencv_exports_unsupported.dart'
    if (dart.library.io) 'opencv_exports_native.dart'
    if (dart.library.js_interop) 'opencv_exports_web.dart';
