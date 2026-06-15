# Hand Detection TFLite - Example App

This example app demonstrates the full capabilities of the `hand_detection`
package across three detection modes.

## Features

### 🖼️ Still Image Detection
- Pick images from gallery
- 21-landmark hand skeleton + rotated palm boxes
- Handedness (left/right) and gesture labels
- Full feature customization (bounding boxes, skeleton, landmarks, labels)
- Color and size customization
- Adjustable max hands and optional gesture recognition
- Detailed performance metrics

### 📹 Live Camera Detection
- Real-time hand detection from camera
- 21 landmarks, skeleton, and gesture overlays
- Adjustable max hands (1–10)
- Optional gesture recognition
- One-tap CompiledModel (GPU) / XNNPACK backend toggle
- FPS + per-frame inference time
- Cross-platform (iOS, Android, macOS, Windows, Linux)

### 🎬 Video File Detection
- Pick an MP4/MOV and annotate hands frame-by-frame (OpenCV)
- One-Euro temporal smoothing to remove landmark jitter
- Style customization (colors, sizes, toggles)
- In-app playback of the annotated output

## Quick Start

1. **Run the app:**
   ```bash
   flutter run
   ```

2. **Choose a demo:**
   - Tap **Live Camera** for real-time detection
   - Tap **Still Image** for image analysis
   - Tap **Video File** to annotate an MP4 frame-by-frame

## Detection Modes

| Mode | Output | Use Case |
|------|--------|----------|
| **Boxes** | Palm bounding boxes only | Fast presence/tracking |
| **Boxes + Landmarks** | Boxes + 21 keypoints per hand | Skeleton, gestures, analysis |

The demos run the full `boxesAndLandmarks` pipeline; the display options just
control what is drawn.

## Gestures

When gesture recognition is enabled, each hand is classified into one of:
`closedFist`, `openPalm`, `pointingUp`, `thumbDown`, `thumbUp`, `victory`,
`iLoveYou` (or `unknown`).

## 21 Hand Landmarks

- Wrist (0)
- Thumb: CMC (1), MCP (2), IP (3), Tip (4)
- Index: MCP (5), PIP (6), DIP (7), Tip (8)
- Middle: MCP (9), PIP (10), DIP (11), Tip (12)
- Ring: MCP (13), PIP (14), DIP (15), Tip (16)
- Pinky: MCP (17), PIP (18), DIP (19), Tip (20)

## Color Customization (Still Image / Video)

Default visualization colors:
- 🟧 **Orange** - Bounding boxes
- 🔴 **Red** - Landmarks (21 points)
- 🟢 **Green** - Skeleton connections

All colors are customizable via the color picker.

## Performance Tips

### Live Camera Detection
- Lower **Max Hands** for higher FPS
- Disable **Gestures** when not needed (skips the gesture models)
- Use the **CM/XNN** toggle to A/B the CompiledModel (GPU) and XNNPACK
  backends on your device

### Still Image / Video
- Higher **Max Hands** finds more hands but is slower
- Display toggles (boxes/skeleton/landmarks) only affect rendering, not
  detection cost

## Platform Support

- ✅ **iOS** - Full support with camera package
- ✅ **Android** - Full support with camera package
- ✅ **macOS** - Full support with camera_desktop package
- ✅ **Windows/Linux** - Full support with camera_desktop package
- ✅ **Web** - Still image detection (camera/video file demos are desktop/mobile)

## Troubleshooting

### Camera Not Working
- Check camera permissions in app settings
- Restart the app
- Try a different camera (front/back)

### Low FPS
- Reduce Max Hands
- Turn off gesture recognition
- Try the other inference backend (CM/XNN)

### Hands Not Detected
- Ensure good lighting
- Make sure hands are visible and unobstructed
- Hands that are very small in frame may be missed

### Video Could Not Open / Write
- Linux requires GStreamer plugins (see the in-app hint)
- The output uses the `avc1` (H.264) codec, which must be available on the OS
  video backend

## Documentation

- [📚 Main Package README](../README.md) - Package documentation

## License

This example is part of the hand_detection package and shares the same license.
