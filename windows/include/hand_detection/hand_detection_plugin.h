#ifndef HAND_DETECTION_PUBLIC_PLUGIN_H_
#define HAND_DETECTION_PUBLIC_PLUGIN_H_

#include <flutter_windows.h>

#ifdef FLUTTER_PLUGIN_IMPL
#define FLUTTER_PLUGIN_EXPORT __declspec(dllexport)
#else
#define FLUTTER_PLUGIN_EXPORT __declspec(dllimport)
#endif

FLUTTER_PLUGIN_EXPORT void HandDetectionPluginRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar);

#endif  // HAND_DETECTION_PUBLIC_PLUGIN_H_
