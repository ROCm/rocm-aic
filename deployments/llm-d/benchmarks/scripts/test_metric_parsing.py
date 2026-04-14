"""
Unit tests for metric parsing in load generators.
"""

import json
import pytest
from pathlib import Path
from load_generators.vllm_bench_serve import VllmBenchServe
from load_generators.multi_turn_benchmark import MultiTurnBenchmark


class TestVllmBenchServeParser:
    """Test vLLM bench serve metric parsing."""

    def test_parse_valid_json(self, tmp_path):
        """Test parsing valid vLLM bench serve JSON output."""
        # Create mock output file
        output_file = tmp_path / "benchmark_output.json"
        data = {
            "completed_requests": 100,
            "total_requests": 100,
            "request_throughput": 45.2,
            "output_throughput": 5824.3,
            "mean_ttft_ms": 125.4,
            "median_ttft_ms": 118.2,
            "p99_ttft_ms": 245.8,
            "mean_tpot_ms": 15.2,
            "median_tpot_ms": 14.8,
            "p99_tpot_ms": 28.5,
            "mean_itl_ms": 18.3,
            "median_itl_ms": 17.1,
            "p99_itl_ms": 35.2
        }
        with open(output_file, 'w') as f:
            json.dump(data, f)

        # Parse
        generator = VllmBenchServe(None)
        result = generator.parse_metrics(output_file)

        # Verify parsing status
        assert result["parsing_status"] == "success"
        assert len(result["parsing_errors"]) == 0

        # Verify core metrics
        metrics = result["metrics"]
        assert metrics["completed_requests"] == 100
        assert metrics["total_requests"] == 100
        assert metrics["request_throughput"] == 45.2
        assert metrics["output_throughput"] == 5824.3

        # Verify TTFT metrics
        assert "ttft" in metrics
        assert metrics["ttft"]["mean_ms"] == 125.4
        assert metrics["ttft"]["median_ms"] == 118.2
        assert metrics["ttft"]["p99_ms"] == 245.8

        # Verify TPOT metrics
        assert "tpot" in metrics
        assert metrics["tpot"]["mean_ms"] == 15.2
        assert metrics["tpot"]["median_ms"] == 14.8
        assert metrics["tpot"]["p99_ms"] == 28.5

        # Verify ITL metrics
        assert "itl" in metrics
        assert metrics["itl"]["mean_ms"] == 18.3

    def test_parse_missing_file(self, tmp_path):
        """Test graceful handling of missing output file."""
        output_file = tmp_path / "nonexistent.json"

        generator = VllmBenchServe(None)
        result = generator.parse_metrics(output_file)

        assert result["parsing_status"] == "failed"
        assert len(result["parsing_errors"]) > 0
        assert "not found" in result["parsing_errors"][0].lower()
        assert result["metrics"] == {}

    def test_parse_invalid_json(self, tmp_path):
        """Test handling of malformed JSON."""
        output_file = tmp_path / "invalid.json"
        with open(output_file, 'w') as f:
            f.write("{ invalid json content }")

        generator = VllmBenchServe(None)
        result = generator.parse_metrics(output_file)

        assert result["parsing_status"] == "failed"
        assert len(result["parsing_errors"]) > 0
        assert "json" in result["parsing_errors"][0].lower()

    def test_parse_partial_metrics(self, tmp_path):
        """Test parsing with only some metrics present."""
        output_file = tmp_path / "partial.json"
        data = {
            "completed_requests": 50,
            "total_requests": 100,
            # Missing throughput metrics
            "mean_ttft_ms": 100.0,
            # Missing other TTFT percentiles
        }
        with open(output_file, 'w') as f:
            json.dump(data, f)

        generator = VllmBenchServe(None)
        result = generator.parse_metrics(output_file)

        assert result["parsing_status"] == "success"
        metrics = result["metrics"]
        assert metrics["completed_requests"] == 50
        assert "ttft" in metrics
        assert metrics["ttft"]["mean_ms"] == 100.0

    def test_parse_all_percentiles(self, tmp_path):
        """Test parsing all percentile metrics."""
        output_file = tmp_path / "percentiles.json"
        data = {
            "mean_ttft_ms": 100.0,
            "median_ttft_ms": 95.0,
            "p25_ttft_ms": 80.0,
            "p50_ttft_ms": 95.0,
            "p75_ttft_ms": 110.0,
            "p90_ttft_ms": 130.0,
            "p95_ttft_ms": 150.0,
            "p99_ttft_ms": 200.0
        }
        with open(output_file, 'w') as f:
            json.dump(data, f)

        generator = VllmBenchServe(None)
        result = generator.parse_metrics(output_file)

        assert result["parsing_status"] == "success"
        ttft = result["metrics"]["ttft"]
        assert len(ttft) == 8  # All percentiles
        assert ttft["mean_ms"] == 100.0
        assert ttft["p99_ms"] == 200.0


class TestMultiTurnBenchmarkParser:
    """Test multi-turn benchmark metric parsing."""

    def test_parse_valid_output(self, tmp_path):
        """Test parsing valid multi-turn benchmark text output."""
        output_file = tmp_path / "benchmark_output.txt"
        content = """
        Processing conversations from workload.json...
        Running benchmark...
        Completed: 95/100 successful
        Mean TTFT: 125ms
        Median TTFT: 118ms
        Mean TPOT: 15ms
        Median TPOT: 14ms
        Throughput: 25 conversations/sec
        Total tokens: 12450
        """
        with open(output_file, 'w') as f:
            f.write(content)

        generator = MultiTurnBenchmark(None)
        result = generator.parse_metrics(output_file)

        assert result["parsing_status"] == "success"
        metrics = result["metrics"]
        assert metrics["successful_conversations"] == 95
        assert metrics["total_conversations"] == 100
        assert metrics["failed_conversations"] == 5
        assert metrics["success_rate"] == 0.95
        assert metrics["mean_ttft_ms"] == 125.0
        assert metrics["median_ttft_ms"] == 118.0
        assert metrics["mean_tpot_ms"] == 15.0
        assert metrics["median_tpot_ms"] == 14.0
        assert metrics["throughput_conversations_per_sec"] == 25.0
        assert metrics["total_tokens"] == 12450

    def test_parse_missing_file(self, tmp_path):
        """Test graceful handling of missing output file."""
        output_file = tmp_path / "nonexistent.txt"

        generator = MultiTurnBenchmark(None)
        result = generator.parse_metrics(output_file)

        assert result["parsing_status"] == "failed"
        assert len(result["parsing_errors"]) > 0
        assert "not found" in result["parsing_errors"][0].lower()

    def test_parse_no_metrics(self, tmp_path):
        """Test handling of output with no recognizable metrics."""
        output_file = tmp_path / "empty.txt"
        content = "Some random log output without metrics\n"
        with open(output_file, 'w') as f:
            f.write(content)

        generator = MultiTurnBenchmark(None)
        result = generator.parse_metrics(output_file)

        assert result["parsing_status"] == "failed"
        assert "no recognized metrics" in result["parsing_errors"][0].lower()
        assert result["metrics"] == {}

    def test_parse_partial_metrics(self, tmp_path):
        """Test parsing with only some metrics present."""
        output_file = tmp_path / "partial.txt"
        content = """
        Running...
        Completed: 80/90 successful
        Mean TTFT: 150.5ms
        """
        with open(output_file, 'w') as f:
            f.write(content)

        generator = MultiTurnBenchmark(None)
        result = generator.parse_metrics(output_file)

        # Should be partial since we're missing TPOT
        assert result["parsing_status"] == "partial"
        assert len(result["parsing_warnings"]) > 0
        metrics = result["metrics"]
        assert metrics["successful_conversations"] == 80
        assert metrics["success_rate"] == pytest.approx(0.888, 0.01)
        assert metrics["mean_ttft_ms"] == 150.5

    def test_parse_variations_in_format(self, tmp_path):
        """Test parsing with variations in spacing and formatting."""
        output_file = tmp_path / "variations.txt"
        content = """
        Completed:   90  /  100   successful
        Mean   TTFT:  120.3   ms
        Throughput:  30.5  conversations/sec
        Mean TPOT: 12ms
        """
        with open(output_file, 'w') as f:
            f.write(content)

        generator = MultiTurnBenchmark(None)
        result = generator.parse_metrics(output_file)

        assert result["parsing_status"] == "success"
        metrics = result["metrics"]
        assert metrics["successful_conversations"] == 90
        assert metrics["mean_ttft_ms"] == 120.3
        assert metrics["throughput_conversations_per_sec"] == 30.5
        assert metrics["mean_tpot_ms"] == 12.0

    def test_parse_with_additional_metrics(self, tmp_path):
        """Test parsing with additional optional metrics."""
        output_file = tmp_path / "extra_metrics.txt"
        content = """
        Completed: 100/100 successful
        Mean TTFT: 110ms
        Mean TPOT: 13ms
        Average tokens per conversation: 245.5
        Request rate: 10.5 req/s
        """
        with open(output_file, 'w') as f:
            f.write(content)

        generator = MultiTurnBenchmark(None)
        result = generator.parse_metrics(output_file)

        assert result["parsing_status"] == "success"
        metrics = result["metrics"]
        assert metrics["avg_tokens_per_conversation"] == 245.5
        assert metrics["request_rate"] == 10.5

    def test_parse_zero_division_safety(self, tmp_path):
        """Test that zero total conversations doesn't cause division error."""
        output_file = tmp_path / "zero.txt"
        content = """
        Completed: 0/0 successful
        """
        with open(output_file, 'w') as f:
            f.write(content)

        generator = MultiTurnBenchmark(None)
        result = generator.parse_metrics(output_file)

        # Should handle gracefully
        assert result["parsing_status"] == "partial"
        metrics = result["metrics"]
        assert metrics["success_rate"] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
