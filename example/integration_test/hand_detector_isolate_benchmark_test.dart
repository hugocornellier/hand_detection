// ignore_for_file: avoid_print, deprecated_member_use

// Benchmark tests for HandDetectorIsolate (deprecated — intentional coverage).
//
// Unlike hand_detector_benchmark_test.dart (which exercises HandDetector
// directly), this benchmark targets HandDetectorIsolate.spawn() to measure the
// full background-isolate path. It runs three configurations to expose the
// nested-IsolateInterpreter overhead:
//
//   - disabled  : PerformanceConfig.disabled, no gestures
//   - auto      : PerformanceConfig() (auto: XNNPACK / Metal), no gestures
//   - disabledG : PerformanceConfig.disabled with enableGestures: true
//                 (4 InterpreterPool instances - max bug surface)
//
// The "disabled" runs are the only ones that touch the affected code path
// (no delegate -> createIsolateIfNeeded() will create a nested
// IsolateInterpreter on iOS/Android/Linux/Windows). The "auto" run is a
// regression guard.
//
// To run on a non-macOS target (the macOS short-circuit at
// flutter_litert/interpreter_factory.dart:59 hides the bug):
//   flutter test integration_test/hand_detector_isolate_benchmark_test.dart -d <ios-sim-id>

import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

const int iterations = 100;
const int warmupIterations = 10;

const List<String> sampleImages = [
  'packages/hand_detection/assets/samples/2-hands.png',
  'packages/hand_detection/assets/samples/360_F_554788951_fLAy5C8e9bha4caBTWVJN6rvTD0pEVfE.jpg',
  'packages/hand_detection/assets/samples/img-standing.png',
  'packages/hand_detection/assets/samples/istockphoto-462908027-612x612.jpg',
  'packages/hand_detection/assets/samples/two-palms.png',
];

class BenchmarkStats {
  final String imagePath;
  // Microsecond timings — IsolateInterpreter overhead on iOS Simulator is
  // sub-millisecond, so elapsedMilliseconds is too coarse to detect a delta.
  final List<int> timings;
  final int imageSize;
  final int detectionCount;

  BenchmarkStats({
    required this.imagePath,
    required this.timings,
    required this.imageSize,
    required this.detectionCount,
  });

  double get mean => timings.reduce((a, b) => a + b) / timings.length;

  double get median {
    final sorted = List<int>.from(timings)..sort();
    final middle = sorted.length ~/ 2;
    if (sorted.length % 2 == 1) return sorted[middle].toDouble();
    return (sorted[middle - 1] + sorted[middle]) / 2.0;
  }

  double get p95 {
    final sorted = List<int>.from(timings)..sort();
    final idx = ((sorted.length - 1) * 0.95).round();
    return sorted[idx].toDouble();
  }

  int get min => timings.reduce((a, b) => a < b ? a : b);
  int get max => timings.reduce((a, b) => a > b ? a : b);

  double get stdDev {
    final m = mean;
    final variance =
        timings.map((x) => (x - m) * (x - m)).reduce((a, b) => a + b) /
            timings.length;
    return variance > 0 ? variance : 0.0;
  }

  void printResults(String label) {
    print('\n$label:');
    print('  Image size: ${(imageSize / 1024).toStringAsFixed(1)} KB');
    print('  Detections: $detectionCount hand(s)');
    print('  Mean:   ${(mean / 1000).toStringAsFixed(3)} ms');
    print('  Median: ${(median / 1000).toStringAsFixed(3)} ms');
    print('  P95:    ${(p95 / 1000).toStringAsFixed(3)} ms');
    print('  Min:    ${(min / 1000).toStringAsFixed(3)} ms');
    print('  Max:    ${(max / 1000).toStringAsFixed(3)} ms');
    print('  StdDev: ${(stdDev / 1000).toStringAsFixed(3)} ms');
  }

  Map<String, dynamic> toJson() => {
        'image_path': imagePath,
        'image_size_kb': (imageSize / 1024),
        'detection_count': detectionCount,
        'iterations': timings.length,
        'timings_us': timings,
        'mean_us': mean,
        'median_us': median,
        'p95_us': p95,
        'min_us': min,
        'max_us': max,
        'stddev_us': stdDev,
      };
}

class BenchmarkResults {
  final String timestamp;
  final String testName;
  final Map<String, dynamic> configuration;
  final List<BenchmarkStats> results;

  BenchmarkResults({
    required this.timestamp,
    required this.testName,
    required this.configuration,
    required this.results,
  });

  double get overallMean {
    final allTimings = results.expand((r) => r.timings).toList();
    return allTimings.reduce((a, b) => a + b) / allTimings.length;
  }

  double get overallMedian {
    final all = results.expand((r) => r.timings).toList()..sort();
    final mid = all.length ~/ 2;
    if (all.length.isOdd) return all[mid].toDouble();
    return (all[mid - 1] + all[mid]) / 2.0;
  }

  void printSummary() {
    print('\n${'=' * 60}');
    print('BENCHMARK SUMMARY');
    print('=' * 60);
    print('Test: $testName');
    print('Timestamp: $timestamp');
    print('Configuration:');
    configuration.forEach((key, value) {
      print('  $key: $value');
    });
    print('\nOverall mean:   ${(overallMean / 1000).toStringAsFixed(3)} ms');
    print('Overall median: ${(overallMedian / 1000).toStringAsFixed(3)} ms');
    print('Total iterations: ${results.length * iterations}');
    print('=' * 60);
  }

  Map<String, dynamic> toJson() => {
        'timestamp': timestamp,
        'test_name': testName,
        'configuration': configuration,
        'overall_mean_us': overallMean,
        'overall_median_us': overallMedian,
        'results': results.map((r) => r.toJson()).toList(),
      };

  void printJson(String filename) {
    print('\n📊 BENCHMARK_JSON_START:$filename');
    print(const JsonEncoder.withIndent('  ').convert(toJson()));
    print('📊 BENCHMARK_JSON_END:$filename');
  }
}

Future<void> runIsolateBenchmark({
  required String label,
  required String shortName,
  required Future<HandDetectorIsolate> Function() spawn,
  required Map<String, dynamic> configuration,
}) async {
  final detector = await spawn();

  print('\n${'=' * 60}');
  print('BENCHMARK: $label');
  print('Iterations per image: $iterations (warmup: $warmupIterations)');
  print('=' * 60);

  final allStats = <BenchmarkStats>[];

  for (final imagePath in sampleImages) {
    final ByteData data = await rootBundle.load(imagePath);
    final Uint8List bytes = data.buffer.asUint8List();

    // Warm-up: load TFLite caches, JIT, etc. Not measured.
    for (int i = 0; i < warmupIterations; i++) {
      final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
      await detector.detectHandsFromMat(mat);
      mat.dispose();
    }

    final List<int> timings = [];
    int detectionCount = 0;

    for (int i = 0; i < iterations; i++) {
      final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

      final stopwatch = Stopwatch()..start();
      final results = await detector.detectHandsFromMat(mat);
      stopwatch.stop();

      mat.dispose();

      timings.add(stopwatch.elapsedMicroseconds);
      if (i == 0) detectionCount = results.length;
    }

    final stats = BenchmarkStats(
      imagePath: imagePath,
      timings: timings,
      imageSize: bytes.length,
      detectionCount: detectionCount,
    );
    stats.printResults(imagePath);
    allStats.add(stats);
  }

  await detector.dispose();

  final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
  final benchmarkResults = BenchmarkResults(
    timestamp: timestamp,
    testName: label,
    configuration: {
      ...configuration,
      'iterations': iterations,
      'warmup_iterations': warmupIterations,
      'sample_images': sampleImages.length,
    },
    results: allStats,
  );
  benchmarkResults.printSummary();
  benchmarkResults.printJson('isolate_${shortName}_$timestamp.json');
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('HandDetectorIsolate - Performance Benchmarks', () {
    test(
      'Isolate + disabled (no delegate)',
      () => runIsolateBenchmark(
        label: 'HandDetectorIsolate + PerformanceConfig.disabled',
        shortName: 'disabled',
        spawn: () => HandDetectorIsolate.spawn(
          performanceConfig: PerformanceConfig.disabled,
        ),
        configuration: {
          'performance_mode': 'disabled',
          'enable_gestures': false,
          'pool_size': 1,
        },
      ),
      timeout: const Timeout(Duration(minutes: 30)),
    );

    test(
      'Isolate + auto (XNNPACK/Metal)',
      () => runIsolateBenchmark(
        label: 'HandDetectorIsolate + PerformanceConfig() auto',
        shortName: 'auto',
        spawn: () => HandDetectorIsolate.spawn(
          performanceConfig: const PerformanceConfig(),
        ),
        configuration: {
          'performance_mode': 'auto',
          'enable_gestures': false,
          'pool_size': 1,
        },
      ),
      timeout: const Timeout(Duration(minutes: 30)),
    );

    test(
      'Isolate + disabled + gestures (4 pools)',
      () => runIsolateBenchmark(
        label: 'HandDetectorIsolate + disabled + gestures (4 pools)',
        shortName: 'disabled_gestures',
        spawn: () => HandDetectorIsolate.spawn(
          performanceConfig: PerformanceConfig.disabled,
          enableGestures: true,
        ),
        configuration: {
          'performance_mode': 'disabled',
          'enable_gestures': true,
          'pool_size': 1,
        },
      ),
      timeout: const Timeout(Duration(minutes: 30)),
    );

    // Captures Fix D (HandDetectorIsolate.spawn default change).
    // Calls spawn() with NO args to measure whatever the package default
    // currently produces.
    test(
      'Isolate + default (spawn with no args)',
      () => runIsolateBenchmark(
        label: 'HandDetectorIsolate.spawn() with no args (defaults)',
        shortName: 'default',
        spawn: () => HandDetectorIsolate.spawn(),
        configuration: {
          'performance_mode': 'package-default',
          'enable_gestures': false,
          'pool_size': 'package-default',
        },
      ),
      timeout: const Timeout(Duration(minutes: 30)),
    );

    // Captures Fix F (lift interpreterPoolSize=1 force under delegate).
    // Asks for poolSize=3 with auto delegate. Without the fix, the constructor
    // silently clamps it to 1; with the fix, it actually parallelizes
    // landmark inference across 3 interpreters for multi-hand frames.
    test(
      'Isolate + auto + poolSize 3',
      () => runIsolateBenchmark(
        label: 'HandDetectorIsolate + auto + interpreterPoolSize=3',
        shortName: 'auto_pool3',
        spawn: () => HandDetectorIsolate.spawn(
          performanceConfig: const PerformanceConfig(),
          interpreterPoolSize: 3,
        ),
        configuration: {
          'performance_mode': 'auto',
          'enable_gestures': false,
          'pool_size': 3,
        },
      ),
      timeout: const Timeout(Duration(minutes: 30)),
    );
  });
}
