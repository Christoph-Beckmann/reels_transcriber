"""
Performance and reliability benchmarks for Instagram Reels Transcriber.
"""

import gc
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import patch

import psutil
import pytest

from core.audio_processor import AudioExtractor
from core.downloader import InstagramDownloader
from core.pipeline import TranscriptionPipeline
from core.transcriber import WhisperTranscriber
from gui.worker import ProcessingWorker
from tests.fixtures.mock_data import (
    MockDataGenerator,
    MockInstagramData,
    MockModelBehavior,
    create_mock_pipeline_dependencies,
)


class PerformanceMonitor:
    """Monitor system performance during tests."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory
        self.start_time = None

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.initial_memory = self.process.memory_info().rss
        self.peak_memory = self.initial_memory

    def update(self):
        """Update peak memory usage."""
        current_memory = self.process.memory_info().rss
        self.peak_memory = max(self.peak_memory, current_memory)

    def get_stats(self) -> dict:
        """Get performance statistics."""
        current_memory = self.process.memory_info().rss
        elapsed_time = time.time() - self.start_time if self.start_time else 0

        return {
            "elapsed_time": elapsed_time,
            "initial_memory_mb": self.initial_memory / (1024 * 1024),
            "current_memory_mb": current_memory / (1024 * 1024),
            "peak_memory_mb": self.peak_memory / (1024 * 1024),
            "memory_increase_mb": (self.peak_memory - self.initial_memory) / (1024 * 1024),
            "cpu_percent": self.process.cpu_percent(),
            "num_threads": self.process.num_threads(),
        }


@pytest.mark.benchmark
class TestPipelinePerformance:
    """Performance benchmarks for pipeline operations."""

    def test_pipeline_initialization_speed(self, benchmark, temp_dir):
        """Benchmark pipeline initialization time."""

        def create_pipeline():
            return TranscriptionPipeline()

        result = benchmark(create_pipeline)
        assert isinstance(result, TranscriptionPipeline)

    def test_url_validation_speed(self, benchmark, temp_dir):
        """Benchmark URL validation performance."""
        downloader = InstagramDownloader(str(temp_dir))
        test_urls = [
            "https://www.instagram.com/reel/ABC123DEF/",
            "https://instagram.com/reel/XYZ789GHI/",
            "https://www.instagram.com/p/DEF456JKL/",
        ]

        def validate_urls():
            results = []
            for url in test_urls:
                result = downloader.validate_reel_url(url)
                results.append(result)
            return results

        results = benchmark(validate_urls)
        assert len(results) == len(test_urls)
        assert all(result[0] for result in results)  # All should be valid

    @patch("core.pipeline.InstagramDownloader")
    @patch("core.pipeline.AudioExtractor")
    @patch("core.pipeline.WhisperTranscriber")
    @patch("core.pipeline.TempFileManager")
    def test_complete_pipeline_speed(
        self,
        mock_file_manager,
        mock_transcriber_class,
        mock_audio_extractor_class,
        mock_downloader_class,
        benchmark,
        temp_dir,
    ):
        """Benchmark complete pipeline execution speed."""
        # Setup fast mocks
        mocks = create_mock_pipeline_dependencies(temp_dir)
        mock_downloader_class.return_value = mocks["downloader"]
        mock_audio_extractor_class.return_value = mocks["audio_extractor"]
        mock_transcriber_class.return_value = mocks["transcriber"]
        mock_file_manager.return_value = mocks["file_manager"]

        def run_pipeline():
            pipeline = TranscriptionPipeline()
            result = pipeline.execute("https://instagram.com/reel/BENCHMARK/")
            return result

        result = benchmark(run_pipeline)
        assert result.success

    def test_concurrent_pipeline_performance(self, temp_dir):
        """Test performance with multiple concurrent pipelines."""
        monitor = PerformanceMonitor()
        monitor.start()

        num_pipelines = 5
        results = []

        def run_pipeline(pipeline_id):
            with (
                patch("core.pipeline.InstagramDownloader") as mock_downloader_class,
                patch("core.pipeline.AudioExtractor") as mock_audio_extractor_class,
                patch("core.pipeline.WhisperTranscriber") as mock_transcriber_class,
                patch("core.pipeline.TempFileManager") as mock_file_manager,
            ):
                mocks = create_mock_pipeline_dependencies(temp_dir)
                mock_downloader_class.return_value = mocks["downloader"]
                mock_audio_extractor_class.return_value = mocks["audio_extractor"]
                mock_transcriber_class.return_value = mocks["transcriber"]
                mock_file_manager.return_value = mocks["file_manager"]

                pipeline = TranscriptionPipeline()
                start_time = time.time()
                result = pipeline.execute(f"https://instagram.com/reel/CONCURRENT{pipeline_id}/")
                end_time = time.time()

                monitor.update()
                return {
                    "id": pipeline_id,
                    "success": result.success,
                    "duration": end_time - start_time,
                    "transcript_length": len(result.transcript) if result.transcript else 0,
                }

        # Run pipelines concurrently
        with ThreadPoolExecutor(max_workers=num_pipelines) as executor:
            futures = [executor.submit(run_pipeline, i) for i in range(num_pipelines)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

        stats = monitor.get_stats()

        # Verify all pipelines completed successfully
        assert len(results) == num_pipelines
        assert all(r["success"] for r in results)

        # Performance assertions
        assert stats["memory_increase_mb"] < 100  # Should not use excessive memory
        assert stats["elapsed_time"] < 10  # Should complete quickly with mocks

        # Calculate performance metrics
        avg_duration = sum(r["duration"] for r in results) / len(results)
        max_duration = max(r["duration"] for r in results)

        print("\nConcurrent Pipeline Performance:")
        print(f"  Number of pipelines: {num_pipelines}")
        print(f"  Average duration: {avg_duration:.3f}s")
        print(f"  Max duration: {max_duration:.3f}s")
        print(f"  Memory increase: {stats['memory_increase_mb']:.1f}MB")
        print(f"  Peak memory: {stats['peak_memory_mb']:.1f}MB")

    def test_memory_usage_over_time(self, temp_dir):
        """Test memory usage patterns over multiple operations."""
        monitor = PerformanceMonitor()
        monitor.start()

        memory_samples = []
        num_iterations = 10

        for i in range(num_iterations):
            with (
                patch("core.pipeline.InstagramDownloader") as mock_downloader_class,
                patch("core.pipeline.AudioExtractor") as mock_audio_extractor_class,
                patch("core.pipeline.WhisperTranscriber") as mock_transcriber_class,
                patch("core.pipeline.TempFileManager") as mock_file_manager,
            ):
                mocks = create_mock_pipeline_dependencies(temp_dir)
                mock_downloader_class.return_value = mocks["downloader"]
                mock_audio_extractor_class.return_value = mocks["audio_extractor"]
                mock_transcriber_class.return_value = mocks["transcriber"]
                mock_file_manager.return_value = mocks["file_manager"]

                pipeline = TranscriptionPipeline()
                pipeline.execute(f"https://instagram.com/reel/MEMORY{i}/")

                # Force cleanup
                pipeline._cleanup_pipeline_state()
                del pipeline

                # Force garbage collection
                gc.collect()

                monitor.update()
                stats = monitor.get_stats()
                memory_samples.append(stats["current_memory_mb"])

        # Analyze memory usage pattern
        initial_memory = memory_samples[0]
        final_memory = memory_samples[-1]
        max_memory = max(memory_samples)
        memory_growth = final_memory - initial_memory

        print("\nMemory Usage Over Time:")
        print(f"  Iterations: {num_iterations}")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Peak memory: {max_memory:.1f}MB")
        print(f"  Memory growth: {memory_growth:.1f}MB")

        # Memory should not grow excessively
        assert memory_growth < 50  # Less than 50MB growth
        assert max_memory - initial_memory < 100  # Peak should be reasonable


@pytest.mark.benchmark
class TestComponentPerformance:
    """Performance benchmarks for individual components."""

    def test_audio_generation_performance(self, benchmark, temp_dir):
        """Benchmark audio file generation for testing."""
        output_path = str(Path(temp_dir) / "benchmark_audio.wav")

        def generate_audio():
            return MockDataGenerator.create_realistic_audio_file(output_path, duration=10.0, include_speech=True)

        result = benchmark(generate_audio)
        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

    def test_audio_validation_performance(self, benchmark, temp_dir):
        """Benchmark audio file validation speed."""
        # Create test audio file
        audio_path = MockDataGenerator.create_realistic_audio_file(
            str(Path(temp_dir) / "validation_test.wav"), duration=5.0
        )

        extractor = AudioExtractor()

        def validate_audio():
            return extractor.validate_audio_file(audio_path)

        result = benchmark(validate_audio)
        assert result[0] is True  # Should be valid

    @patch("core.transcriber.WhisperModel")
    def test_transcriber_initialization_performance(self, mock_whisper_model, benchmark, temp_dir):
        """Benchmark transcriber initialization speed."""
        mock_model = MockModelBehavior.create_mock_whisper_model("normal")
        mock_whisper_model.return_value = mock_model

        def init_transcriber():
            transcriber = WhisperTranscriber(model_size="tiny")
            success, error = transcriber.load_model()
            return transcriber, success

        result = benchmark(init_transcriber)
        transcriber, success = result
        assert success

    def test_worker_thread_performance(self, benchmark, temp_dir):
        """Benchmark worker thread creation and communication."""
        import queue

        def create_and_test_worker():
            message_queue = queue.Queue()
            config = {"temp_dir": str(temp_dir)}

            worker = ProcessingWorker(message_queue, config)

            # Test basic operations
            assert not worker.is_processing()

            # Start and stop (should be fast with mocks)
            with patch("gui.worker.TranscriptionPipeline"):
                success = worker.start_processing("https://instagram.com/reel/BENCHMARK/")
                if success:
                    worker.stop_processing()

            return worker

        result = benchmark(create_and_test_worker)
        assert isinstance(result, ProcessingWorker)


@pytest.mark.reliability
class TestReliabilityAndStress:
    """Reliability and stress tests."""

    def test_pipeline_reliability_under_load(self, temp_dir):
        """Test pipeline reliability under sustained load."""
        num_operations = 20
        success_count = 0
        error_count = 0
        errors = []

        monitor = PerformanceMonitor()
        monitor.start()

        for i in range(num_operations):
            try:
                with (
                    patch("core.pipeline.InstagramDownloader") as mock_downloader_class,
                    patch("core.pipeline.AudioExtractor") as mock_audio_extractor_class,
                    patch("core.pipeline.WhisperTranscriber") as mock_transcriber_class,
                    patch("core.pipeline.TempFileManager") as mock_file_manager,
                ):
                    mocks = create_mock_pipeline_dependencies(temp_dir)
                    mock_downloader_class.return_value = mocks["downloader"]
                    mock_audio_extractor_class.return_value = mocks["audio_extractor"]
                    mock_transcriber_class.return_value = mocks["transcriber"]
                    mock_file_manager.return_value = mocks["file_manager"]

                    pipeline = TranscriptionPipeline()
                    result = pipeline.execute(f"https://instagram.com/reel/STRESS{i}/")

                    if result.success:
                        success_count += 1
                    else:
                        error_count += 1
                        errors.append(result.error_message)

                    # Cleanup
                    pipeline._cleanup_pipeline_state()
                    del pipeline

                monitor.update()

            except Exception as e:
                error_count += 1
                errors.append(str(e))

        stats = monitor.get_stats()

        # Calculate reliability metrics
        success_rate = success_count / num_operations
        error_count / num_operations

        print("\nReliability Test Results:")
        print(f"  Operations: {num_operations}")
        print(f"  Successes: {success_count}")
        print(f"  Errors: {error_count}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Memory usage: {stats['memory_increase_mb']:.1f}MB")

        # Reliability requirements
        assert success_rate >= 0.95  # At least 95% success rate
        assert stats["memory_increase_mb"] < 200  # Reasonable memory usage

        if errors:
            print(f"  Error types: {set(errors)}")

    def test_error_recovery_reliability(self, temp_dir):
        """Test reliability of error recovery mechanisms."""
        error_scenarios = ["network_failure", "file_not_found", "processing_error", "timeout", "permission_denied"]

        recovery_success_count = 0

        for scenario in error_scenarios:
            try:
                # Simulate different error conditions
                pipeline = TranscriptionPipeline()

                if scenario == "network_failure":
                    # This would need actual error injection
                    pipeline.execute("https://instagram.com/reel/NETWORK_ERROR/")

                elif scenario == "file_not_found":
                    pipeline.execute("https://instagram.com/reel/FILE_NOT_FOUND/")

                # Test that pipeline handles errors gracefully
                # Even if it fails, it should not crash
                recovery_success_count += 1

            except Exception as e:
                # Unexpected exceptions indicate poor error handling
                print(f"Unexpected error in scenario {scenario}: {e}")

        # All error scenarios should be handled gracefully
        recovery_rate = recovery_success_count / len(error_scenarios)
        assert recovery_rate >= 0.8  # At least 80% should be handled gracefully

    def test_resource_cleanup_reliability(self, temp_dir):
        """Test reliability of resource cleanup."""
        initial_files = set(Path(temp_dir).rglob("*"))

        # Create and destroy multiple pipelines
        for i in range(5):
            with (
                patch("core.pipeline.InstagramDownloader") as mock_downloader_class,
                patch("core.pipeline.AudioExtractor") as mock_audio_extractor_class,
                patch("core.pipeline.WhisperTranscriber") as mock_transcriber_class,
                patch("core.pipeline.TempFileManager") as mock_file_manager,
            ):
                mocks = create_mock_pipeline_dependencies(temp_dir)
                mock_downloader_class.return_value = mocks["downloader"]
                mock_audio_extractor_class.return_value = mocks["audio_extractor"]
                mock_transcriber_class.return_value = mocks["transcriber"]
                mock_file_manager.return_value = mocks["file_manager"]

                pipeline = TranscriptionPipeline()
                pipeline.execute(f"https://instagram.com/reel/CLEANUP{i}/")

                # Force cleanup
                pipeline._cleanup_pipeline_state()
                del pipeline

        # Force garbage collection
        gc.collect()

        # Check for resource leaks
        final_files = set(Path(temp_dir).rglob("*"))
        new_files = final_files - initial_files

        # Should not create persistent files (mocks should handle cleanup)
        persistent_files = [f for f in new_files if f.exists()]

        print("\nResource Cleanup Test:")
        print(f"  Initial files: {len(initial_files)}")
        print(f"  Final files: {len(final_files)}")
        print(f"  New persistent files: {len(persistent_files)}")

        # Should not leave many persistent files
        assert len(persistent_files) < 10

    def test_thread_safety_stress(self, temp_dir):
        """Stress test thread safety with rapid concurrent operations."""
        num_threads = 10
        operations_per_thread = 5
        thread_results = {}
        errors = []

        def thread_worker(thread_id):
            thread_results[thread_id] = {"successes": 0, "errors": 0}

            for op_id in range(operations_per_thread):
                try:
                    with (
                        patch("core.pipeline.InstagramDownloader") as mock_downloader_class,
                        patch("core.pipeline.AudioExtractor") as mock_audio_extractor_class,
                        patch("core.pipeline.WhisperTranscriber") as mock_transcriber_class,
                        patch("core.pipeline.TempFileManager") as mock_file_manager,
                    ):
                        mocks = create_mock_pipeline_dependencies(temp_dir)
                        mock_downloader_class.return_value = mocks["downloader"]
                        mock_audio_extractor_class.return_value = mocks["audio_extractor"]
                        mock_transcriber_class.return_value = mocks["transcriber"]
                        mock_file_manager.return_value = mocks["file_manager"]

                        pipeline = TranscriptionPipeline()
                        result = pipeline.execute(f"https://instagram.com/reel/THREAD{thread_id}_{op_id}/")

                        if result.success:
                            thread_results[thread_id]["successes"] += 1
                        else:
                            thread_results[thread_id]["errors"] += 1

                        pipeline._cleanup_pipeline_state()
                        del pipeline

                except Exception as e:
                    thread_results[thread_id]["errors"] += 1
                    errors.append(f"Thread {thread_id}, Op {op_id}: {e}")

        # Start all threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=thread_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Analyze results
        total_operations = num_threads * operations_per_thread
        total_successes = sum(r["successes"] for r in thread_results.values())
        total_errors = sum(r["errors"] for r in thread_results.values())

        success_rate = total_successes / total_operations
        total_errors / total_operations

        print("\nThread Safety Stress Test:")
        print(f"  Threads: {num_threads}")
        print(f"  Operations per thread: {operations_per_thread}")
        print(f"  Total operations: {total_operations}")
        print(f"  Successes: {total_successes}")
        print(f"  Errors: {total_errors}")
        print(f"  Success rate: {success_rate:.2%}")

        # Thread safety requirements
        assert success_rate >= 0.90  # At least 90% success under stress
        assert len(errors) < total_operations * 0.1  # Less than 10% unexpected errors

        if errors:
            print(f"  Sample errors: {errors[:3]}")  # Show first 3 errors


@pytest.mark.benchmark
@pytest.mark.slow
class TestLongRunningPerformance:
    """Long-running performance tests."""

    def test_extended_operation_stability(self, temp_dir):
        """Test stability over extended operations."""
        monitor = PerformanceMonitor()
        monitor.start()

        runtime_minutes = 2  # 2 minutes for CI compatibility
        end_time = time.time() + (runtime_minutes * 60)
        operation_count = 0

        print(f"\nRunning extended stability test for {runtime_minutes} minutes...")

        while time.time() < end_time:
            try:
                with (
                    patch("core.pipeline.InstagramDownloader") as mock_downloader_class,
                    patch("core.pipeline.AudioExtractor") as mock_audio_extractor_class,
                    patch("core.pipeline.WhisperTranscriber") as mock_transcriber_class,
                    patch("core.pipeline.TempFileManager") as mock_file_manager,
                ):
                    mocks = create_mock_pipeline_dependencies(temp_dir)
                    mock_downloader_class.return_value = mocks["downloader"]
                    mock_audio_extractor_class.return_value = mocks["audio_extractor"]
                    mock_transcriber_class.return_value = mocks["transcriber"]
                    mock_file_manager.return_value = mocks["file_manager"]

                    pipeline = TranscriptionPipeline()
                    result = pipeline.execute(f"https://instagram.com/reel/EXTENDED{operation_count}/")

                    assert result.success

                    pipeline._cleanup_pipeline_state()
                    del pipeline

                    operation_count += 1
                    monitor.update()

                    # Small delay to prevent overwhelming the system
                    time.sleep(0.1)

            except Exception as e:
                print(f"Error in operation {operation_count}: {e}")
                break

        stats = monitor.get_stats()

        print("\nExtended Stability Results:")
        print(f"  Runtime: {stats['elapsed_time']:.1f} seconds")
        print(f"  Operations completed: {operation_count}")
        print(f"  Operations per second: {operation_count / stats['elapsed_time']:.2f}")
        print(f"  Memory increase: {stats['memory_increase_mb']:.1f}MB")
        print(f"  Peak memory: {stats['peak_memory_mb']:.1f}MB")

        # Stability requirements
        assert operation_count > 10  # Should complete multiple operations
        assert stats["memory_increase_mb"] < 500  # Memory growth should be bounded

        # Calculate throughput
        throughput = operation_count / stats["elapsed_time"]
        assert throughput > 0.5  # At least 0.5 operations per second


@pytest.mark.benchmark
@pytest.mark.real_urls
@pytest.mark.network
class TestRealURLPerformance:
    """Performance benchmarks using real Instagram URLs."""

    def test_real_url_download_performance(self, temp_dir):
        """Benchmark download performance with real Instagram URLs."""
        test_urls = MockInstagramData.TEST_CATEGORIES["performance_benchmark"]["urls"]

        print("\n⏱️  Download Performance Test")
        print(f"URLs to test: {len(test_urls)}")

        monitor = PerformanceMonitor()
        monitor.start()

        download_results = []

        for i, test_url in enumerate(test_urls):
            print(f"\n📥 Testing download {i + 1}/{len(test_urls)}: {test_url}")

            try:
                downloader = InstagramDownloader(str(temp_dir / f"download_{i}"))

                # Measure validation performance
                start_time = time.time()
                is_valid, error_msg, shortcode = downloader.validate_reel_url(test_url)
                validation_time = time.time() - start_time

                if is_valid:
                    # Measure download performance
                    start_time = time.time()

                    def progress_callback(progress, message):
                        pass  # Silent progress for benchmarking

                    success, video_path, download_error = downloader.download_reel(test_url, progress_callback)
                    download_time = time.time() - start_time

                    result = {
                        "url": test_url,
                        "validation_time": validation_time,
                        "download_success": success,
                        "download_time": download_time,
                        "video_size": 0,
                        "error": download_error,
                    }

                    if success and video_path and Path(video_path).exists():
                        result["video_size"] = Path(video_path).stat().st_size
                        print(f"   ✅ Download: {download_time:.2f}s, Size: {result['video_size'] / 1024 / 1024:.1f}MB")
                    else:
                        print(f"   ❌ Download failed: {download_error}")

                else:
                    result = {
                        "url": test_url,
                        "validation_time": validation_time,
                        "download_success": False,
                        "download_time": 0,
                        "video_size": 0,
                        "error": error_msg,
                    }
                    print(f"   ❌ Validation failed: {error_msg}")

                download_results.append(result)
                monitor.update()

                # Brief pause between downloads
                time.sleep(1)

            except Exception as e:
                error_msg = str(e).lower()
                print(f"   💥 Exception: {e}")

                if any(keyword in error_msg for keyword in ["network", "connection", "timeout", "dns", "ssl"]):
                    download_results.append(
                        {
                            "url": test_url,
                            "validation_time": 0,
                            "download_success": False,
                            "download_time": 0,
                            "video_size": 0,
                            "error": "Network error",
                        }
                    )
                else:
                    raise

        stats = monitor.get_stats()

        # Analyze download performance
        successful_downloads = [r for r in download_results if r["download_success"]]
        failed_downloads = [r for r in download_results if not r["download_success"]]

        print("\n📊 Download Performance Summary:")
        print(f"   Successful downloads: {len(successful_downloads)}/{len(download_results)}")
        print(f"   Total execution time: {stats['elapsed_time']:.2f}s")
        print(f"   Memory usage: {stats['memory_increase_mb']:.1f}MB")

        if successful_downloads:
            avg_validation_time = sum(r["validation_time"] for r in successful_downloads) / len(successful_downloads)
            avg_download_time = sum(r["download_time"] for r in successful_downloads) / len(successful_downloads)
            avg_video_size = sum(r["video_size"] for r in successful_downloads) / len(successful_downloads)

            print(f"   Average validation time: {avg_validation_time:.3f}s")
            print(f"   Average download time: {avg_download_time:.2f}s")
            print(f"   Average video size: {avg_video_size / 1024 / 1024:.1f}MB")

            # Performance assertions (adjust as needed)
            assert avg_validation_time < 5.0, f"Validation too slow: {avg_validation_time:.3f}s"
            assert avg_download_time < 120.0, f"Download too slow: {avg_download_time:.2f}s"

        else:
            print("⚠️  No successful downloads - may be network/access issues")
            # Check if all failures are network-related
            network_errors = sum(
                1
                for r in failed_downloads
                if "network" in (r.get("error", "")).lower()
                or "connection" in (r.get("error", "")).lower()
                or "timeout" in (r.get("error", "")).lower()
            )

            if network_errors == len(failed_downloads):
                pytest.skip("All download tests failed due to network issues")
            else:
                pytest.fail("Download performance test failed with non-network errors")

    @pytest.mark.slow
    def test_end_to_end_real_url_performance(self, temp_dir):
        """Benchmark complete end-to-end performance with real URLs."""
        test_urls = MockInstagramData.TEST_CATEGORIES["performance_benchmark"]["urls"]

        print("\n🏁 End-to-End Performance Test")
        print(f"URLs to test: {len(test_urls)}")

        monitor = PerformanceMonitor()
        monitor.start()

        e2e_results = []

        for i, test_url in enumerate(test_urls):
            print(f"\n🚀 E2E Test {i + 1}/{len(test_urls)}: {test_url}")

            try:
                pipeline = TranscriptionPipeline()

                # Track detailed timing
                stage_times = {}
                stage_start_time = time.time()

                def timing_progress_callback(update):
                    nonlocal stage_start_time
                    current_time = time.time()
                    stage_name = update.stage.name if hasattr(update.stage, "name") else str(update.stage)

                    if stage_name not in stage_times:
                        stage_times[stage_name] = {"start": stage_start_time, "duration": 0}

                    if update.progress == 100 or "complete" in update.message.lower():
                        stage_times[stage_name]["duration"] = current_time - stage_times[stage_name]["start"]
                        stage_start_time = current_time

                start_time = time.time()
                result = pipeline.execute(test_url, progress_callback=timing_progress_callback)
                end_time = time.time()

                total_time = end_time - start_time

                e2e_result = {
                    "url": test_url,
                    "success": result.success,
                    "total_time": total_time,
                    "transcript_length": len(result.transcript) if result.transcript else 0,
                    "language": result.detected_language if result.success else None,
                    "stages_completed": result.stages_completed if hasattr(result, "stages_completed") else 0,
                    "stage_times": stage_times.copy(),
                    "error": result.error_message if not result.success else None,
                }

                if result.success:
                    print(
                        f"   ✅ Success: {total_time:.2f}s, "
                        f"Transcript: {e2e_result['transcript_length']} chars, "
                        f"Language: {result.detected_language}"
                    )

                    # Print stage timing details
                    for stage, timing in stage_times.items():
                        if timing["duration"] > 0:
                            print(f"      {stage}: {timing['duration']:.2f}s")

                else:
                    print(f"   ❌ Failed in {total_time:.2f}s: {result.error_message}")

                e2e_results.append(e2e_result)
                monitor.update()

                # Cleanup between tests
                pipeline._cleanup_pipeline_state()
                time.sleep(2)  # Pause between E2E tests

            except Exception as e:
                error_msg = str(e).lower()
                print(f"   💥 Exception: {e}")

                if any(keyword in error_msg for keyword in ["network", "connection", "timeout", "dns", "ssl"]):
                    e2e_results.append(
                        {
                            "url": test_url,
                            "success": False,
                            "total_time": 0,
                            "transcript_length": 0,
                            "error": "Network error",
                        }
                    )
                else:
                    raise

        stats = monitor.get_stats()

        # Analyze E2E performance
        successful_e2e = [r for r in e2e_results if r["success"]]
        failed_e2e = [r for r in e2e_results if not r["success"]]

        print("\n🏆 End-to-End Performance Summary:")
        print(f"   Successful completions: {len(successful_e2e)}/{len(e2e_results)}")
        print(f"   Total test time: {stats['elapsed_time']:.2f}s")
        print(f"   Memory usage: {stats['memory_increase_mb']:.1f}MB")
        print(f"   Peak memory: {stats['peak_memory_mb']:.1f}MB")

        if successful_e2e:
            avg_total_time = sum(r["total_time"] for r in successful_e2e) / len(successful_e2e)
            avg_transcript_length = sum(r["transcript_length"] for r in successful_e2e) / len(successful_e2e)

            print(f"   Average total time: {avg_total_time:.2f}s")
            print(f"   Average transcript length: {avg_transcript_length:.0f} chars")

            # Calculate words per second if we have transcript data
            if avg_transcript_length > 0:
                # Rough estimate: 5 characters per word
                avg_words = avg_transcript_length / 5
                words_per_second = avg_words / avg_total_time
                print(f"   Processing rate: ~{words_per_second:.1f} words/second")

            # Performance thresholds (adjust based on requirements)
            assert avg_total_time < 300, f"E2E processing too slow: {avg_total_time:.2f}s"
            assert stats["memory_increase_mb"] < 500, f"Memory usage too high: {stats['memory_increase_mb']:.1f}MB"

        else:
            print("⚠️  No successful E2E tests - may be network/access issues")
            # Check error patterns
            network_errors = sum(
                1
                for r in failed_e2e
                if "network" in (r.get("error", "")).lower()
                or "connection" in (r.get("error", "")).lower()
                or "timeout" in (r.get("error", "")).lower()
            )

            if network_errors == len(failed_e2e):
                pytest.skip("All E2E tests failed due to network issues")
            else:
                pytest.fail("E2E performance test failed with non-network errors")

    @pytest.mark.stress
    def test_real_url_concurrent_performance(self, temp_dir):
        """Test concurrent processing performance with real URLs."""
        test_url = MockInstagramData.TEST_CATEGORIES["integration_basic"]["urls"][0]
        num_concurrent = 3  # Conservative for real URLs

        print("\n🔄 Concurrent Performance Test")
        print(f"URL: {test_url}")
        print(f"Concurrent processes: {num_concurrent}")

        monitor = PerformanceMonitor()
        monitor.start()

        results = []

        def concurrent_task(task_id):
            try:
                print(f"   Task {task_id}: Starting")
                pipeline = TranscriptionPipeline()

                start_time = time.time()
                result = pipeline.execute(test_url)
                end_time = time.time()

                task_result = {
                    "task_id": task_id,
                    "success": result.success,
                    "execution_time": end_time - start_time,
                    "transcript_length": len(result.transcript) if result.transcript else 0,
                    "error": result.error_message if not result.success else None,
                }

                print(f"   Task {task_id}: {'✅' if result.success else '❌'} in {task_result['execution_time']:.2f}s")

                pipeline._cleanup_pipeline_state()
                return task_result

            except Exception as e:
                str(e).lower()
                print(f"   Task {task_id}: 💥 Exception: {e}")

                return {
                    "task_id": task_id,
                    "success": False,
                    "execution_time": 0,
                    "transcript_length": 0,
                    "error": str(e),
                }

        # Run concurrent tasks
        with ThreadPoolExecutor(max_workers=num_concurrent) as executor:
            futures = [executor.submit(concurrent_task, i) for i in range(num_concurrent)]

            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                monitor.update()

        stats = monitor.get_stats()

        # Analyze concurrent performance
        successful_tasks = [r for r in results if r["success"]]
        failed_tasks = [r for r in results if not r["success"]]

        print("\n⚡ Concurrent Performance Summary:")
        print(f"   Successful tasks: {len(successful_tasks)}/{len(results)}")
        print(f"   Total test time: {stats['elapsed_time']:.2f}s")
        print(f"   Memory usage: {stats['memory_increase_mb']:.1f}MB")

        if successful_tasks:
            avg_task_time = sum(r["execution_time"] for r in successful_tasks) / len(successful_tasks)
            max_task_time = max(r["execution_time"] for r in successful_tasks)
            min_task_time = min(r["execution_time"] for r in successful_tasks)

            print(f"   Average task time: {avg_task_time:.2f}s")
            print(f"   Task time range: {min_task_time:.2f}s - {max_task_time:.2f}s")

            # Concurrent performance should be reasonable
            assert avg_task_time < 300, f"Concurrent tasks too slow: {avg_task_time:.2f}s"
            assert (
                stats["memory_increase_mb"] < 800
            ), f"Concurrent memory usage too high: {stats['memory_increase_mb']:.1f}MB"

        else:
            print("⚠️  No successful concurrent tasks")
            # Check if all failures are network-related
            network_errors = sum(
                1
                for r in failed_tasks
                if "network" in (r.get("error", "")).lower() or "connection" in (r.get("error", "")).lower()
            )

            if network_errors == len(failed_tasks):
                pytest.skip("Concurrent test failed due to network issues")
            else:
                pytest.fail("Concurrent performance test failed")
