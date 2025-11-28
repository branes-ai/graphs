"""
Calibration Logging System

Provides structured logging for hardware calibration with automatic file output.
Logs are saved alongside calibration JSON files with matching filenames.

Usage:
    from graphs.hardware.calibration.logging import CalibrationLogger, get_logger

    # Initialize logging for a calibration run
    logger = CalibrationLogger(output_dir=Path("calibrations/"), prefix="performance_1037MHz_numpy")

    # Get the logger instance for use in any module
    log = get_logger()
    log.info("Starting calibration...")
    log.section("BLAS Benchmarks")
    log.result("fp32", "1024x1024", gflops=547.0, efficiency=0.427)

    # Or use the context manager to capture all stdout (including print() calls)
    with CalibrationLogger(output_dir, prefix, capture_stdout=True) as log:
        print("This gets captured!")  # Captured to log file
        log.info("So does this")
"""

import io
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, TextIO
from contextlib import contextmanager
from dataclasses import dataclass


# Module-level logger instance
_calibration_logger: Optional['CalibrationLogger'] = None


def get_logger() -> 'CalibrationLogger':
    """
    Get the current calibration logger instance.

    Returns:
        The active CalibrationLogger, or a default console-only logger if none initialized.
    """
    global _calibration_logger
    if _calibration_logger is None:
        # Create a default console-only logger
        _calibration_logger = CalibrationLogger()
    return _calibration_logger


def set_logger(logger: 'CalibrationLogger'):
    """Set the module-level calibration logger."""
    global _calibration_logger
    _calibration_logger = logger


@dataclass
class LogConfig:
    """Configuration for calibration logging."""

    # Output directory for log files
    output_dir: Optional[Path] = None

    # Prefix for log filename (e.g., "performance_1037MHz_numpy")
    filename_prefix: Optional[str] = None

    # Log level for console output
    console_level: int = logging.INFO

    # Log level for file output
    file_level: int = logging.DEBUG

    # Whether to include timestamps in console output
    console_timestamps: bool = False

    # Whether to include timestamps in file output
    file_timestamps: bool = True

    # Width for section separators
    separator_width: int = 80


class TeeStream:
    """
    A stream that writes to both the original stream and captures to a buffer.

    This allows capturing stdout while still displaying output in real-time.
    """

    def __init__(self, original_stream: TextIO, buffer: list):
        self.original = original_stream
        self.buffer = buffer

    def write(self, text: str):
        self.original.write(text)
        # Split by newlines and add to buffer (excluding empty trailing newline)
        if text:
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if i < len(lines) - 1:
                    # This line was followed by a newline
                    self.buffer.append(line)
                elif line:
                    # Last segment without trailing newline - append to last buffer line
                    if self.buffer:
                        self.buffer[-1] += line
                    else:
                        self.buffer.append(line)

    def flush(self):
        self.original.flush()

    def isatty(self):
        return self.original.isatty()


class CalibrationLogger:
    """
    Structured logger for hardware calibration.

    Provides:
    - Dual output to console and file
    - Structured formatting for benchmark results
    - Section headers and separators
    - Automatic log file creation alongside calibration JSON
    - Optional stdout capture to log all print() calls

    The logger is designed to produce human-readable output that serves
    as both real-time feedback and a permanent record of the calibration run.
    """

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        filename_prefix: Optional[str] = None,
        config: Optional[LogConfig] = None,
        capture_stdout: bool = False,
    ):
        """
        Initialize the calibration logger.

        Args:
            output_dir: Directory to save log file. If None, logs to console only.
            filename_prefix: Prefix for log filename (e.g., "performance_1037MHz_numpy").
                           The log file will be named "{prefix}.log".
            config: Optional LogConfig for advanced configuration.
            capture_stdout: If True, capture all stdout (print() calls) to the log.
        """
        self.config = config or LogConfig()
        self.output_dir = output_dir or self.config.output_dir
        self.filename_prefix = filename_prefix or self.config.filename_prefix
        self.capture_stdout = capture_stdout

        self._log_file: Optional[TextIO] = None
        self._log_path: Optional[Path] = None
        self._lines: list[str] = []  # Buffer for log content
        self._original_stdout: Optional[TextIO] = None
        self._tee_stream: Optional[TeeStream] = None

        # Set up file logging if output directory is specified
        if self.output_dir and self.filename_prefix:
            self._setup_file_logging()

        # Set up stdout capture if requested
        if capture_stdout:
            self._start_stdout_capture()

        # Register as the global logger
        set_logger(self)

    def _setup_file_logging(self):
        """Set up file logging."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self.output_dir / f"{self.filename_prefix}.log"
        self._log_file = open(self._log_path, 'w')

    def _start_stdout_capture(self):
        """Start capturing stdout."""
        self._original_stdout = sys.stdout
        self._tee_stream = TeeStream(sys.stdout, self._lines)
        sys.stdout = self._tee_stream

    def _stop_stdout_capture(self):
        """Stop capturing stdout and restore original."""
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
            self._tee_stream = None

    @property
    def log_path(self) -> Optional[Path]:
        """Get the path to the log file, if any."""
        return self._log_path

    def _write(self, message: str, to_console: bool = True, to_file: bool = True):
        """Write a message to console and/or file."""
        if to_console:
            print(message)
            # If TeeStream is capturing stdout, it will add to _lines automatically
            # so we don't need to append again here
            if self.capture_stdout and self._tee_stream is not None:
                # TeeStream already captured this, don't double-add
                pass
            else:
                self._lines.append(message)
        else:
            # Not going to console, so TeeStream won't capture it
            self._lines.append(message)

        if to_file and self._log_file:
            timestamp = ""
            if self.config.file_timestamps:
                timestamp = f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] "
            self._log_file.write(f"{timestamp}{message}\n")
            self._log_file.flush()  # Ensure immediate write

    def info(self, message: str):
        """Log an informational message."""
        self._write(message)

    def debug(self, message: str):
        """Log a debug message (file only by default)."""
        self._write(message, to_console=False, to_file=True)

    def warning(self, message: str):
        """Log a warning message."""
        self._write(f"⚠ WARNING: {message}")

    def error(self, message: str):
        """Log an error message."""
        self._write(f"✗ ERROR: {message}")

    def success(self, message: str):
        """Log a success message."""
        self._write(f"✓ {message}")

    def section(self, title: str, level: int = 1):
        """
        Print a section header.

        Args:
            title: Section title
            level: Header level (1=major, 2=minor)
        """
        width = self.config.separator_width
        if level == 1:
            self._write("")
            self._write("=" * width)
            self._write(title)
            self._write("=" * width)
        else:
            self._write("")
            self._write(title)
            self._write("-" * width)

    def separator(self, char: str = "-"):
        """Print a separator line."""
        self._write(char * self.config.separator_width)

    def blank(self):
        """Print a blank line."""
        self._write("")

    def table_header(self, *columns: str, widths: Optional[list[int]] = None):
        """
        Print a table header row.

        Args:
            columns: Column headers
            widths: Optional column widths (default: auto)
        """
        if widths is None:
            widths = [max(12, len(col) + 2) for col in columns]

        header = "  ".join(f"{col:<{w}}" for col, w in zip(columns, widths))
        self._write(header)
        self._write("-" * len(header))

    def table_row(self, *values, widths: Optional[list[int]] = None):
        """
        Print a table row.

        Args:
            values: Row values
            widths: Optional column widths (must match header)
        """
        if widths is None:
            widths = [max(12, len(str(v)) + 2) for v in values]

        row = "  ".join(f"{str(v):<{w}}" for v, w in zip(values, widths))
        self._write(row)

    def result(
        self,
        precision: str,
        size: str,
        gflops: Optional[float] = None,
        latency_ms: Optional[float] = None,
        efficiency: Optional[float] = None,
        bandwidth_gbps: Optional[float] = None,
        status: str = "OK",
    ):
        """
        Log a benchmark result in a structured format.

        Args:
            precision: Precision tested (e.g., "fp32", "int8")
            size: Problem size (e.g., "1024x1024", "2K")
            gflops: Measured GFLOPS/GIOPS
            latency_ms: Measured latency in milliseconds
            efficiency: Efficiency as fraction (0.0-1.0)
            bandwidth_gbps: Measured bandwidth in GB/s
            status: Status string (e.g., "OK", "SLOW", "SKIP")
        """
        parts = [f"  {precision:8s}"]

        if size:
            parts.append(f"{size:>8s}")

        if gflops is not None:
            unit = "GIOPS" if precision.startswith("int") else "GFLOPS"
            parts.append(f"{gflops:>8.1f} {unit}")

        if latency_ms is not None:
            if latency_ms >= 1000:
                parts.append(f"{latency_ms/1000:>7.2f}s")
            else:
                parts.append(f"{latency_ms:>7.2f}ms")

        if efficiency is not None:
            parts.append(f"{efficiency*100:>6.1f}%")

        if bandwidth_gbps is not None:
            parts.append(f"{bandwidth_gbps:>7.1f} GB/s")

        if status != "OK":
            parts.append(f"[{status}]")

        self._write("  ".join(parts))

    def summary(self, title: str, **metrics):
        """
        Log a summary with key-value metrics.

        Args:
            title: Summary title
            **metrics: Key-value pairs to display
        """
        self._write("")
        self._write(f"{title}:")
        for key, value in metrics.items():
            # Format key: replace underscores with spaces, title case
            formatted_key = key.replace("_", " ").title()
            self._write(f"  {formatted_key}: {value}")

    def get_content(self) -> str:
        """Get all logged content as a string."""
        return "\n".join(self._lines)

    def close(self):
        """Close the log file and stop stdout capture."""
        # Stop stdout capture first
        self._stop_stdout_capture()

        # Then close the file
        if self._log_file:
            self._log_file.close()
            self._log_file = None

    def __enter__(self) -> 'CalibrationLogger':
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop capture and close log file."""
        self.close()
        return False


class LogAdapter:
    """
    Adapter that provides print()-compatible interface for gradual migration.

    This allows existing code using print() to be migrated incrementally
    by replacing `print` with `log` (an instance of LogAdapter).

    Usage:
        log = LogAdapter(get_logger())
        log("Starting calibration...")  # Works like print()
        log.section("BLAS")              # Also supports structured methods
    """

    def __init__(self, logger: CalibrationLogger):
        self._logger = logger

    def __call__(self, *args, **kwargs):
        """Make the adapter callable like print()."""
        message = " ".join(str(arg) for arg in args)
        self._logger.info(message)

    def __getattr__(self, name):
        """Delegate attribute access to the underlying logger."""
        return getattr(self._logger, name)


def create_calibration_logger(
    output_dir: Optional[Path] = None,
    power_mode: str = "unknown",
    freq_mhz: int = 0,
    framework: str = "numpy",
) -> CalibrationLogger:
    """
    Create a calibration logger with standard naming convention.

    The log file will be named: {power_mode}_{freq_mhz}MHz_{framework}.log

    Args:
        output_dir: Directory to save log file (typically calibrations/ subdirectory)
        power_mode: Power mode name (e.g., "performance", "MAXN", "7W")
        freq_mhz: Frequency in MHz
        framework: Framework name ("numpy" or "pytorch")

    Returns:
        Configured CalibrationLogger instance
    """
    import re

    # Sanitize power mode (remove special characters)
    power_mode_clean = re.sub(r'[^a-zA-Z0-9]', '', power_mode)

    filename_prefix = f"{power_mode_clean}_{freq_mhz}MHz_{framework}"

    return CalibrationLogger(
        output_dir=output_dir,
        filename_prefix=filename_prefix,
    )
