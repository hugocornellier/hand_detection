/// Native (non-web) implementation of hand_detection.
///
/// Aggregates the native implementation parts (the isolate-backed
/// [HandDetector], the model runners, UI overlays) and re-exports the shared
/// pure-Dart types so user code sees the same `Hand`, `HandLandmark`,
/// `GestureResult`, etc. on every platform. This is the target of the
/// conditional export in `package:hand_detection/hand_detection.dart` on
/// everything except web.
library;

// Single source of truth for all public types and constants.
export '../shared/hand_types.dart';

// Drawing/overlay helpers (pure dart:ui; safe on every platform but kept here
// to mirror the web aggregator's export set).
export '../ui/hand_overlay.dart';

// Native, isolate-backed detector and raw model result type.
export '../hand_detector.dart' show HandDetector;
export '../isolate/hand_detector_isolate.dart' show HandDetectorIsolate;
export '../models/palm_detector.dart' show PalmDetection;
export '../dart_registration.dart';

// Re-export the small opencv_dart slice for users who call detectFromMat.
export '../exports/opencv_exports.dart';

export 'package:flutter_litert/flutter_litert.dart'
    show
        Accelerator,
        Precision,
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
