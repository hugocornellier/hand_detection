# Local patches

This directory vendors `video_player_avfoundation` 2.9.4 from pub.dev, used via a
`path:` dependency override in `example/pubspec.yaml`. The vendored `pubspec.yaml`
keeps `version: 2.9.4`, so this remains an exact 2.9.4 pin.

Why 2.9.4 is pinned: later versions (2.9.7) ship a macOS Swift Package that imports
the iOS-only `<Flutter/Flutter.h>` module and fail to build on macOS. 2.9.5 and 2.9.6
still use the deprecated `AVKeyValueStatus` API, so they would not remove the warning
below either.

Why it is vendored (not the hosted 2.9.4): on a clean macOS build the plugin emits a
deprecation warning:

```
FVPAVFactory.h: warning: 'AVKeyValueStatus' is deprecated: first deprecated in
macOS 13.0 - Use AVAsyncProperty.Status instead
```

`AVKeyValueStatus` is `AVF_DEPRECATED_FOR_SWIFT_ONLY` (deprecated only when the
Objective-C header is imported into Swift). The Objective-C API is still supported
and is needed for the plugin's older deployment targets.

## Change

- `darwin/video_player_avfoundation/Sources/video_player_avfoundation/include/video_player_avfoundation/FVPAVFactory.h`
  wraps the single `-statusOfValueForKey:error:` declaration in
  `#pragma clang diagnostic push` / `ignored "-Wdeprecated-declarations"` / `pop`.

The pragma has no runtime or ABI effect and suppresses only this one deprecation at
its declaration site (verified: a clean module rebuild goes from 6 warnings to 0).

To re-create from a fresh pub cache: re-vendor
`video_player_avfoundation-2.9.4` (excluding `example/` and `test/`) and re-apply the
pragma above.

The vendored copy excludes the upstream package's own `example/` and `test/`
directories, which a `path:` dependency does not need.
