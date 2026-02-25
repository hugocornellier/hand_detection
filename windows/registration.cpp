#include "include/hand_detection/hand_detection_plugin.h"
#include "hand_detection_plugin.h"
#include <flutter/plugin_registrar_windows.h>

void HandDetectionPluginRegisterWithRegistrar(FlutterDesktopPluginRegistrarRef registrar) {
  auto cpp_registrar =
      flutter::PluginRegistrarManager::GetInstance()
          ->GetRegistrar<flutter::PluginRegistrarWindows>(registrar);
  hand_detection::HandDetectionPlugin::RegisterWithRegistrar(cpp_registrar);
}
