# Testing Guide

This directory contains tests for the `hand_detection` package.

## Test Structure

### `/test` - Unit Tests (Limited)
Contains basic unit tests that can run in a pure Dart VM environment:
- ✅ Error handling (StateError when not initialized)
- ✅ API structure validation

**Note:** Most functionality requires TensorFlow Lite native libraries, so comprehensive testing must be done via integration tests.

### `/integration_test` - Integration Tests (Comprehensive)
Contains full integration tests that run in an actual app environment with TFLite support:
- ✅ Initialization and disposal
- ✅ Error handling
- ✅ Detection with real sample images
- ✅ `detect()` and `detectOnImage()` methods
- ✅ Different model variants (lite, full, heavy)
- ✅ Different modes (boxes, boxesAndLandmarks)
- ✅ Landmark and bounding box access (all 21 hand landmarks)
- ✅ Configuration parameters
- ✅ Edge cases

**Total Test Cases: 30**

## Running Tests

### ⚠️ Important: TensorFlow Lite Requirement

The standard `flutter test` command runs tests in a Dart VM which **does not** have access to native libraries. Since this package uses TensorFlow Lite (a native library), most tests will fail with:

```
Failed to load dynamic library 'libtensorflowlite_c-mac.dylib'
```

**This is expected!** You must run integration tests instead.

### Integration Tests (Recommended)

Integration tests run in an actual app environment where native libraries are available.

#### Using the Example App (Easiest)

```bash
cd example
flutter test integration_test/
```

This will run the integration tests within the example app on a connected device or simulator.

#### On iOS Simulator

```bash
# List available simulators
flutter devices

# Run tests on a specific simulator
cd example
flutter test integration_test/ --device-id=<simulator-id>
```

#### On Android Emulator

```bash
# Start an emulator first
flutter emulators --launch <emulator-name>

# Run tests
cd example
flutter test integration_test/ --device-id=<emulator-id>
```

#### On Physical Device

```bash
# Connect device via USB/WiFi
cd example
flutter test integration_test/
```

### Quick Validation (Limited)

To run the limited unit tests (only error handling):

```bash
flutter test test/hand_detector_test.dart
```

**Expected result:** Limited host-safe checks pass; most model-backed tests require platform/native TFLite support.

## Test Coverage

The test suite covers:

1. **Initialization**
   - Default configuration
   - Custom parameters
   - Re-initialization
   - Multiple dispose calls

2. **Error Handling**
   - StateError when not initialized
   - Invalid image bytes
   - Empty images

3. **Real Image Detection**
   - Bundled hand sample images
   - Multiple hands per image
   - Different image sizes

4. **API Methods**
   - `detect(Uint8List)` with byte arrays
   - `detectOnImage(img.Image)` with pre-decoded images
   - Results consistency between both methods

5. **Model Variants**
   - HandLandmarkModel.full

6. **Detection Modes**
   - HandMode.boxesAndLandmarks (full pipeline)
   - HandMode.boxes (fast, no landmarks)

7. **Data Access**
   - 21 hand landmarks
   - Bounding box coordinates
   - Normalized coordinates
   - Visibility scores
   - Depth (z) coordinates

8. **Configuration**
   - detectorConf threshold
   - detectorIou threshold
   - maxDetections limit
   - minLandmarkScore threshold

## Sample Images

The tests use real hand images from `assets/samples/`:
- `2-hands.png`
- `two-palms.png`
- `img-standing.png`
- Additional bundled sample images
- Images contain hands in varied positions
- Different lighting conditions and backgrounds
- Multiple people in some images

## Expected Test Results

When running in a proper environment (device or platform-specific tests):
- ✅ All tests should pass
- Detection should find hands in sample images
- Landmarks should have valid coordinates within image bounds
- Visibility scores should be between 0.0 and 1.0

## Troubleshooting

### "Failed to load dynamic library" error
This error occurs when running `flutter test` without a proper platform environment. The TensorFlow Lite native library is not available to pure Dart tests.

**Solutions:**
- Preferred: Run tests on a device or use platform-specific test commands.
- Use platform or integration tests when native TensorFlow Lite libraries are required.

### Tests timing out
Some tests process multiple images and may take longer on slower devices.

**Solution:** Increase the test timeout or use the lite model for faster processing.

### No people detected in images
If tests fail because no hands are detected, verify:
1. Sample images are properly bundled (check pubspec.yaml)
2. Model files are accessible
3. Configuration thresholds aren't too strict

## Adding New Tests

When adding new tests:

1. Use real sample images when possible
2. Test both `detect()` and `detectOnImage()` paths
3. Verify results are within expected bounds
4. Test edge cases and error conditions
5. Clean up resources with `await detector.dispose()`

## CI/CD Integration

For CI/CD pipelines, consider:
- Running tests on emulators/simulators
- Using integration_test package for full E2E tests
- Splitting unit tests (pure Dart) from integration tests (require TFLite)
