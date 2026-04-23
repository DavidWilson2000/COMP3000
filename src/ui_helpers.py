from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import Callable


def format_metric(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    if sys.platform.startswith("win"):
        os.startfile(str(path))  # type: ignore[attr-defined]
        return

    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def choose_existing_python(venv_python: Path) -> str:
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def stream_subprocess(
    cmd: list[str],
    cwd: Path,
    on_line: Callable[[str], None],
    on_done: Callable[[int], None],
) -> None:
    """Run a subprocess in a background thread and stream output line by line."""

    def worker() -> None:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            assert proc.stdout is not None
            for line in proc.stdout:
                on_line(line.rstrip("\n"))

            proc.wait()
            on_done(proc.returncode)
        except Exception as exc:
            on_line(f"[ERROR] {exc}")
            on_done(-1)

    threading.Thread(target=worker, daemon=True).start()
