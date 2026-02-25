// ignore_for_file: avoid_print

// Benchmark tests for HandDetector.
//
// This test measures the performance of hand detection across multiple iterations
// and sample images. Results are printed to the console with special markers that
// the runBenchmark.sh script extracts and saves to benchmark_results/*.json files.
//
// To run:
// - Use the runBenchmark.sh script in the project root (recommended)
// - Or run directly: flutter test integration_test/hand_detector_benchmark_test.dart -d macos

import 'dart:convert';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:hand_detection/hand_detection.dart';
import 'package:opencv_dart/opencv_dart.dart' as cv;

// Benchmark configuration
const int iterations = 100;

// Sample images in assets/samples/ directory
// Note: Flutter assets cannot be dynamically discovered, so this list must be maintained manually
const List<String> sampleImages = [
  '../assets/samples/2-hands.png',
  '../assets/samples/360_F_554788951_fLAy5C8e9bha4caBTWVJN6rvTD0pEVfE.jpg',
  '../assets/samples/img-standing.png',
  '../assets/samples/istockphoto-462908027-612x612.jpg',
  '../assets/samples/two-palms.png',
];

/// Statistics for a single image benchmark
class BenchmarkStats {
  final String imagePath;
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
    if (sorted.length % 2 == 1) {
      return sorted[middle].toDouble();
    } else {
      return (sorted[middle - 1] + sorted[middle]) / 2.0;
    }
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
    print('  Mean:   ${mean.toStringAsFixed(2)} ms');
    print('  Median: ${median.toStringAsFixed(2)} ms');
    print('  Min:    $min ms');
    print('  Max:    $max ms');
    print('  StdDev: ${stdDev.toStringAsFixed(2)} ms');
  }

  Map<String, dynamic> toJson() => {
        'image_path': imagePath,
        'image_size_kb': (imageSize / 1024),
        'detection_count': detectionCount,
        'iterations': timings.length,
        'timings_ms': timings,
        'mean_ms': mean,
        'median_ms': median,
        'min_ms': min,
        'max_ms': max,
        'stddev_ms': stdDev,
      };
}

/// Aggregated benchmark results
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
    print('\nOverall mean: ${overallMean.toStringAsFixed(2)} ms');
    print('Total iterations: ${results.length * iterations}');
    print('=' * 60);
  }

  Map<String, dynamic> toJson() => {
        'timestamp': timestamp,
        'test_name': testName,
        'configuration': configuration,
        'overall_mean_ms': overallMean,
        'results': results.map((r) => r.toJson()).toList(),
      };

  void printJson(String filename) {
    print('\n📊 BENCHMARK_JSON_START:$filename');
    print(const JsonEncoder.withIndent('  ').convert(toJson()));
    print('📊 BENCHMARK_JSON_END:$filename');
  }
}

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  group('HandDetector - Performance Benchmarks', () {
    test(
      'Benchmark with cv.Mat input + XNNPACK',
      () async {
        print('\nUsing ${sampleImages.length} sample images:');
        for (final img in sampleImages) {
          print('  - $img');
        }

        final detector = HandDetector(
          mode: HandMode.boxesAndLandmarks,
          landmarkModel: HandLandmarkModel.full,
          performanceConfig: const PerformanceConfig.xnnpack(),
        );
        await detector.initialize();

        print('\n${'=' * 60}');
        print('BENCHMARK: Hand Detection with cv.Mat + XNNPACK');
        print('Iterations per image: $iterations');
        print('=' * 60);

        final allStats = <BenchmarkStats>[];

        for (final imagePath in sampleImages) {
          final ByteData data = await rootBundle.load(imagePath);
          final Uint8List bytes = data.buffer.asUint8List();

          final List<int> timings = [];
          int detectionCount = 0;

          // Run iterations - decode fresh Mat each time
          for (int i = 0; i < iterations; i++) {
            final cv.Mat mat = cv.imdecode(bytes, cv.IMREAD_COLOR);

            final stopwatch = Stopwatch()..start();
            final results = await detector.detectOnMat(mat);
            stopwatch.stop();

            mat.dispose();

            timings.add(stopwatch.elapsedMilliseconds);
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

        // Write results to file
        final timestamp = DateTime.now().toIso8601String().replaceAll(':', '-');
        final benchmarkResults = BenchmarkResults(
          timestamp: timestamp,
          testName: 'Hand Detection with cv.Mat + XNNPACK',
          configuration: {
            'mode': 'boxesAndLandmarks',
            'landmark_model': 'full',
            'performance_config': 'xnnpack',
            'api': 'detectOnMat',
            'iterations': iterations,
            'sample_images': sampleImages.length,
          },
          results: allStats,
        );
        benchmarkResults.printSummary();
        benchmarkResults.printJson('benchmark_$timestamp.json');
      },
      timeout: const Timeout(Duration(minutes: 30)),
    );

    test(
      'Benchmark: Live camera simulation (fresh Mat each frame)',
      () async {
        final detector = HandDetector(
          mode: HandMode.boxesAndLandmarks,
          landmarkModel: HandLandmarkModel.full,
          performanceConfig: const PerformanceConfig.xnnpack(),
        );
        await detector.initialize();

        print('\n${'=' * 60}');
        print('BENCHMARK: Live Camera Simulation');
        print('Fresh cv.Mat each frame (simulates camera frame processing)');
        print('=' * 60);

        // Use first image for sustained throughput test
        final ByteData data = await rootBundle.load(sampleImages[0]);
        final Uint8List bytes = data.buffer.asUint8List();

        // Warm-up with fresh Mats
        for (int i = 0; i < 5; i++) {
          final warmupMat = cv.imdecode(bytes, cv.IMREAD_COLOR);
          await detector.detectOnMat(warmupMat);
          warmupMat.dispose();
        }

        // Measure sustained throughput with fresh Mat each frame
        const int frames = 100;
        final stopwatch = Stopwatch()..start();
        for (int i = 0; i < frames; i++) {
          final mat = cv.imdecode(bytes, cv.IMREAD_COLOR);
          await detector.detectOnMat(mat);
          mat.dispose();
        }
        stopwatch.stop();

        final totalMs = stopwatch.elapsedMilliseconds;
        final avgMs = totalMs / frames;
        final fps = 1000 / avgMs;

        print('\nSustained throughput test ($frames frames):');
        print('  Total time:    $totalMs ms');
        print(
            '  Avg per frame: ${avgMs.toStringAsFixed(1)} ms (includes decode)');
        print('  Throughput:    ${fps.toStringAsFixed(1)} FPS');
        print('=' * 60);

        await detector.dispose();
      },
      timeout: const Timeout(Duration(minutes: 10)),
    );
  });
}
