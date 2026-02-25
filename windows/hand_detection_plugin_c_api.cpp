#include "include/hand_detection/hand_detection_plugin_c_api.h"

#include <flutter/plugin_registrar_windows.h>

#include "hand_detection_plugin.h"

void HandDetectionPluginCApiRegisterWithRegistrar(
    FlutterDesktopPluginRegistrarRef registrar) {
  hand_detection::HandDetectionPlugin::RegisterWithRegistrar(
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar));
}
