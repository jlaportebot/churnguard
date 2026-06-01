"""Tests for the CLI module."""

from __future__ import annotations

from click.testing import CliRunner
from pathlib import Path

import pandas as pd
import pytest

from churnguard.cli import main
from churnguard.data import generate_sample_data


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    """Create a sample CSV for CLI testing."""
    df = generate_sample_data(n_rows=200, churn_rate=0.25, random_state=42)
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return path


class TestCLI:
    """Tests for the CLI interface."""

    def test_version(self, runner: CliRunner):
        """Test --version flag."""
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "ChurnGuard" in result.output

    def test_help(self, runner: CliRunner):
        """Test --help flag."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "churn prediction" in result.output.lower()

    def test_sample_command(self, runner: CliRunner, tmp_path: Path):
        """Test the sample command."""
        output = str(tmp_path / "sample.csv")
        result = runner.invoke(main, ["sample", "--output", output, "--rows", "100"])
        assert result.exit_code == 0
        assert Path(output).exists()
        df = pd.read_csv(output)
        assert len(df) == 100

    def test_sample_with_churn_rate(self, runner: CliRunner, tmp_path: Path):
        """Test sample command with custom churn rate."""
        output = str(tmp_path / "sample.csv")
        result = runner.invoke(main, [
            "sample", "--output", output, "--rows", "500", "--churn-rate", "0.3"
        ])
        assert result.exit_code == 0
        df = pd.read_csv(output)
        assert 0.1 < df["churn"].mean() < 0.6

    def test_analyze_command(self, runner: CliRunner, sample_csv: Path, tmp_path: Path):
        """Test the analyze command."""
        output = str(tmp_path / "analysis_output")
        result = runner.invoke(main, [
            "analyze", str(sample_csv),
            "--target", "churn",
            "--output", output,
            "--models", "logistic",
            "--no-plots",
        ])
        assert result.exit_code == 0

    def test_analyze_with_multiple_models(self, runner: CliRunner, sample_csv: Path, tmp_path: Path):
        """Test analyze with multiple models."""
        output = str(tmp_path / "multi_output")
        result = runner.invoke(main, [
            "analyze", str(sample_csv),
            "--target", "churn",
            "--output", output,
            "--models", "logistic",
            "--models", "random_forest",
            "--no-plots",
        ])
        assert result.exit_code == 0

    def test_analyze_nonexistent_file(self, runner: CliRunner):
        """Test analyze with nonexistent file."""
        result = runner.invoke(main, [
            "analyze", "/nonexistent/file.csv", "--target", "churn"
        ])
        assert result.exit_code != 0

    def test_compare_command(self, runner: CliRunner, sample_csv: Path, tmp_path: Path):
        """Test the compare command."""
        output = str(tmp_path / "compare_output")
        result = runner.invoke(main, [
            "compare", str(sample_csv),
            "--target", "churn",
            "--output", output,
        ])
        assert result.exit_code == 0

    def test_sample_help(self, runner: CliRunner):
        """Test sample command help."""
        result = runner.invoke(main, ["sample", "--help"])
        assert result.exit_code == 0
        assert "rows" in result.output.lower()

    def test_analyze_help(self, runner: CliRunner):
        """Test analyze command help."""
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
