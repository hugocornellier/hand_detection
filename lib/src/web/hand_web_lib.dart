/// Web implementation of hand_detection.
///
/// This is the target of the conditional export in
/// `package:hand_detection/hand_detection.dart` when `dart.library.js_interop`
/// is available (Flutter web). It re-exports the same shared pure-Dart types,
/// overlay helpers, and flutter_litert utilities as the native aggregator so
/// user code sees one API on every platform.
library;

// Single source of truth for all public types and constants.
export '../shared/hand_types.dart';
export '../shared/hand_geometry.dart' show PalmDetection;

// Drawing/overlay helpers (pure dart:ui; identical to native).
export '../ui/hand_overlay.dart';

// LiteRT.js + Canvas backed detector (web HandDetector).
export 'hand_detector_web.dart' show HandDetector;

export '../dart_registration.dart';

// opencv_dart has no web build; this resolves to an empty stub on web.
export '../exports/opencv_exports.dart';

export 'package:flutter_litert/flutter_litert.dart'
    show
        PerformanceMode,
        PerformanceConfig,
        createNHWCTensor4D,
        fillNHWC4D,
        allocTensorShape,
        flattenDynamicTensor,
        sigmoid,
        sigmoidClipped,
        clamp01,
        clip,
        computeLetterboxParams,
        LetterboxParams,
        bgrBytesToRgbFloat32,
        bgrBytesToSignedFloat32,
        Point,
        BoundingBox,
        packYuv420,
        YuvPlane,
        YuvLayout,
        PackedYuv,
        CameraPlane,
        CameraFrame,
        CameraFrameConversion,
        CameraFrameRotation,
        prepareCameraFrame,
        prepareCameraFrameFromImage,
        rotationForFrame,
        detectionSize,
        coverFitScaleOffset,
        barQuarterTurns,
        FpsCounter,
        drawLandmarkMarker,
        drawSkeletonConnections,
        drawBoundingBoxOutline;
