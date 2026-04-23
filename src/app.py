from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

from PIL import Image, ImageTk, UnidentifiedImageError

import config
from metrics_reader import RunSummary, load_latest_run_summary
from ui_helpers import choose_existing_python, format_metric, open_path, stream_subprocess


class FishAIDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(config.WINDOW_TITLE)
        self.minsize(*config.APP_MIN_SIZE)
        self.geometry("1250x820")

        self.summary: RunSummary = RunSummary()
        self.preview_refs: list[ImageTk.PhotoImage] = []
        self.is_running = False

        self._build_ui()
        self.after(1000, lambda: self.refresh_summary(initial=True))

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=12)
        left.grid(row=0, column=0, sticky="ns")

        right = ttk.Frame(self, padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1)

        title = ttk.Label(left, text="Controls", font=("Segoe UI", 14, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        btns: list[tuple[str, Callable[[], None]]] = [
            ("Refresh Summary", self.refresh_summary),
            ("Train Detector", self.run_train_detector),
            ("Run Video Pipeline", self.run_video_pipeline),
            ("Open Raw Video Folder", lambda: self.safe_open(config.RAW_VIDEO_DIR)),
            ("Open Processed Video Folder", lambda: self.safe_open(config.PROCESSED_VIDEO_DIR)),
            ("Open Video Runs Folder", lambda: self.safe_open(config.VIDEO_RUNS_DIR)),
            ("Open Latest Video Run Folder", self.open_latest_video_run_folder),
            ("Open Dataset Folder", lambda: self.safe_open(config.DATASETS_DIR)),
            ("Open Runs Folder", lambda: self.safe_open(config.RUNS_DIR)),
            ("Open Latest Run Folder", self.open_latest_run_folder),
            ("Open Best Weights", self.open_best_weights),
            ("Open Results Graph", self.open_results_graph),
            ("Open Confusion Matrix", self.open_confusion_matrix),
        ]

        for text, command in btns:
            ttk.Button(left, text=text, command=command).pack(fill="x", pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=10)

        ttk.Label(left, text="Project Root", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.root_var = tk.StringVar(value=str(config.PROJECT_ROOT))
        ttk.Label(left, textvariable=self.root_var, wraplength=280).pack(anchor="w", pady=(2, 8))

        ttk.Label(left, text="Dataset YAML", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.dataset_var = tk.StringVar(value=str(config.DEFAULT_DATASET_YAML))
        ttk.Label(left, textvariable=self.dataset_var, wraplength=280).pack(anchor="w", pady=(2, 8))

        ttk.Label(left, text="Latest Detector Run", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.latest_run_var = tk.StringVar(value="-")
        ttk.Label(left, textvariable=self.latest_run_var, wraplength=280).pack(anchor="w", pady=(2, 8))

        ttk.Label(left, text="Latest Video Run", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.latest_video_run_var = tk.StringVar(value="-")
        ttk.Label(left, textvariable=self.latest_video_run_var, wraplength=280).pack(anchor="w", pady=(2, 8))

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(left, textvariable=self.status_var, foreground="#1e5aa8").pack(anchor="w", pady=(10, 0))

        summary_frame = ttk.LabelFrame(right, text="Latest Metrics", padding=12)
        summary_frame.grid(row=0, column=0, sticky="ew")
        for i in range(4):
            summary_frame.columnconfigure(i, weight=1)

        self.metric_vars = {
            "precision": tk.StringVar(value="-"),
            "recall": tk.StringVar(value="-"),
            "map50": tk.StringVar(value="-"),
            "map50_95": tk.StringVar(value="-"),
            "epochs": tk.StringVar(value="-"),
        }

        items = [
            ("Precision", "precision"),
            ("Recall", "recall"),
            ("mAP50", "map50"),
            ("mAP50-95", "map50_95"),
        ]
        for idx, (label, key) in enumerate(items):
            block = ttk.Frame(summary_frame)
            block.grid(row=0, column=idx, sticky="nsew", padx=8)
            ttk.Label(block, text=label, font=("Segoe UI", 10, "bold")).pack(anchor="center")
            ttk.Label(block, textvariable=self.metric_vars[key], font=("Segoe UI", 16)).pack(anchor="center", pady=(6, 0))

        meta_frame = ttk.Frame(right)
        meta_frame.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        ttk.Label(meta_frame, text="Completed Epochs:", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(meta_frame, textvariable=self.metric_vars["epochs"]).pack(side="left", padx=(6, 16))

        preview_frame = ttk.LabelFrame(right, text="Preview", padding=10)
        preview_frame.grid(row=2, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)

        self.results_preview = ttk.Label(preview_frame, text="No results preview")
        self.results_preview.grid(row=0, column=0, padx=8, pady=8, sticky="n")

        self.conf_preview = ttk.Label(preview_frame, text="No confusion matrix preview")
        self.conf_preview.grid(row=0, column=1, padx=8, pady=8, sticky="n")

        log_frame = ttk.LabelFrame(right, text="Run Log", padding=8)
        log_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        right.rowconfigure(3, weight=1)

        self.log = tk.Text(log_frame, height=14, wrap="word")
        self.log.pack(fill="both", expand=True)
        self.log.configure(state="disabled")

    def append_log(self, line: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def safe_open(self, path: Path) -> None:
        try:
            open_path(path)
        except Exception as exc:
            messagebox.showerror("Open Path Failed", str(exc))

    def get_latest_video_run_dir(self) -> Path | None:
        if not config.VIDEO_RUNS_DIR.exists():
            return None

        candidates = [p for p in config.VIDEO_RUNS_DIR.iterdir() if p.is_dir()]
        if not candidates:
            return None

        return max(candidates, key=lambda p: p.stat().st_mtime)

    def refresh_summary(self, initial: bool = False) -> None:
        self.summary = load_latest_run_summary(config.DETECT_RUNS_DIR)
        self.latest_run_var.set(str(self.summary.run_dir) if self.summary.run_dir else "-")
        latest_video = self.get_latest_video_run_dir()
        self.latest_video_run_var.set(str(latest_video) if latest_video else "-")
        self.metric_vars["precision"].set(format_metric(self.summary.precision))
        self.metric_vars["recall"].set(format_metric(self.summary.recall))
        self.metric_vars["map50"].set(format_metric(self.summary.map50))
        self.metric_vars["map50_95"].set(format_metric(self.summary.map50_95))
        self.metric_vars["epochs"].set(str(self.summary.epochs) if self.summary.epochs is not None else "-")
        self._load_preview(self.results_preview, self.summary.results_png, "No results preview")
        self._load_preview(self.conf_preview, self.summary.confusion_matrix_png, "No confusion matrix preview")
        if not initial:
            self.status_var.set("Summary refreshed")

    def _load_preview(self, label, image_path, empty_text) -> None:
        label.configure(text=empty_text, image="")
        label.image = None

        if not image_path:
            return

        path = Path(image_path)
        if not path.exists():
            return

        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                img.thumbnail(config.PREVIEW_SIZE)
                preview = img.copy()

            photo = ImageTk.PhotoImage(preview)
            label.configure(image=photo, text="")
            label.image = photo

        except (UnidentifiedImageError, OSError, PermissionError, ValueError):
            label.configure(text=f"{empty_text}\nPreview unavailable while file is updating.")
            label.image = None

    def _start_process(self, name: str, script: Path) -> None:
        if self.is_running:
            messagebox.showinfo("Busy", "A process is already running. Please wait for it to finish.")
            return

        if not script.exists():
            messagebox.showerror("Missing Script", f"Could not find script:\n{script}")
            return

        self.is_running = True
        self.status_var.set(f"Running: {name}")
        self.append_log(f"\n=== {name} ===")

        python_exe = choose_existing_python(config.DEFAULT_VENV_PYTHON)
        cmd = [python_exe, str(script)]

        def on_line(line: str) -> None:
            self.after(0, lambda: self.append_log(line))

        def on_done(code: int) -> None:
            def finish() -> None:
                self.is_running = False
                if code == 0:
                    self.status_var.set(f"Finished: {name}")
                    self.append_log(f"[DONE] {name} completed successfully.")
                else:
                    self.status_var.set(f"Failed: {name}")
                    self.append_log(f"[FAILED] {name} exited with code {code}.")
                self.refresh_summary()

            self.after(0, finish)

        stream_subprocess(cmd, config.PROJECT_ROOT, on_line, on_done)

    def run_train_detector(self) -> None:
        self._start_process("Train Detector", config.TRAIN_SCRIPT)

    def run_video_pipeline(self) -> None:
        self._start_process("Run Video Pipeline", config.PIPELINE_SCRIPT)

    def open_latest_run_folder(self) -> None:
        if not self.summary.run_dir:
            messagebox.showinfo("No Run", "No detector run could be found yet.")
            return
        self.safe_open(self.summary.run_dir)

    def open_latest_video_run_folder(self) -> None:
        latest_video = self.get_latest_video_run_dir()
        if not latest_video:
            messagebox.showinfo("No Video Run", "No video run folder could be found yet.")
            return
        self.safe_open(latest_video)

    def open_best_weights(self) -> None:
        path = self.summary.best_weights
        if path is None or not path.exists():
            messagebox.showinfo("No Weights", "No best.pt file was found for the latest run.")
            return
        self.safe_open(path.parent)

    def open_results_graph(self) -> None:
        path = self.summary.results_png
        if path is None or not path.exists():
            messagebox.showinfo("No Results Graph", "No results.png file was found for the latest run.")
            return
        self.safe_open(path)

    def open_confusion_matrix(self) -> None:
        path = self.summary.confusion_matrix_png
        if path is None or not path.exists():
            messagebox.showinfo("No Confusion Matrix", "No confusion_matrix.png file was found for the latest run.")
            return
        self.safe_open(path)


if __name__ == "__main__":
    app = FishAIDashboard()
    app.mainloop()
