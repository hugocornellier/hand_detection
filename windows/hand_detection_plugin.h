#ifndef FLUTTER_PLUGIN_HAND_DETECTION_PLUGIN_H_
#define FLUTTER_PLUGIN_HAND_DETECTION_PLUGIN_H_

#include <flutter/method_channel.h>
#include <flutter/plugin_registrar_windows.h>

#include <memory>

namespace hand_detection {

class HandDetectionPlugin : public flutter::Plugin {
 public:
  static void RegisterWithRegistrar(flutter::PluginRegistrarWindows *registrar);

  HandDetectionPlugin();

  virtual ~HandDetectionPlugin();

  HandDetectionPlugin(const HandDetectionPlugin&) = delete;
  HandDetectionPlugin& operator=(const HandDetectionPlugin&) = delete;

  void HandleMethodCall(
      const flutter::MethodCall<flutter::EncodableValue> &method_call,
      std::unique_ptr<flutter::MethodResult<flutter::EncodableValue>> result);
};

}  // namespace hand_detection

#endif  // FLUTTER_PLUGIN_HAND_DETECTION_PLUGIN_H_
