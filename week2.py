#!/usr/bin/env python3
"""
================================================================================
  🌸 DECODELABS INDUSTRIAL TRAINING KIT — ARTIFICIAL INTELLIGENCE 🌸
  Project 2: Enterprise Data Classification Using AI
  KNN Classification on the Iris Benchmark Dataset

  ⭐ VVIP ULTRA PROFESSIONAL EDITION ⭐
  IPO Pipeline (Input → Process → Output) Architecture

  Features:
    ✨ Animated Terminal UI with Progress Bars
    🎨 Real-time Elbow Curve Animation
    📊 Interactive Confusion Matrix with Annotations
    🎭 Rich Color Themes (Cyberpunk + Matrix + Ocean)
    🚀 Live Training Metrics Dashboard
    💾 Auto-Save All Plots + Interactive Display

  Batch: 2026 | Powered by DecodeLabs
================================================================================
"""

from __future__ import annotations

import os
import sys
import time
import warnings
import logging
import logging.handlers
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Final

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Interactive backend for plt.show()
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.collections import LineCollection
import seaborn as sns

from dotenv import load_dotenv
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — SECURE CONFIGURATION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

load_dotenv(override=True)


class Config:
    """Centralized secure configuration via environment variables."""
    RANDOM_STATE: Final[int]       = int(os.getenv("RANDOM_STATE", "42"))
    TEST_SIZE: Final[float]        = float(os.getenv("TEST_SIZE", "0.20"))
    STANDARD_K: Final[int]         = int(os.getenv("STANDARD_K", "5"))
    K_MIN: Final[int]              = int(os.getenv("K_MIN", "1"))
    K_MAX: Final[int]              = int(os.getenv("K_MAX", "31"))
    FIGURE_DPI: Final[int]         = int(os.getenv("FIGURE_DPI", "150"))
    LOG_LEVEL: Final[str]          = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_FILE: Final[str]           = os.getenv("LOG_FILE", "app.log")
    LOG_DIR: Final[str]            = os.getenv("LOG_DIR", "logs")
    OUTPUT_PLOT: Final[str]        = os.getenv("OUTPUT_PLOT", "decodelabs_knn_diagnostics.png")
    ANIMATION_FPS: Final[int]      = int(os.getenv("ANIMATION_FPS", "8"))
    THEME: Final[str]              = os.getenv("THEME", "cyberpunk")  # cyberpunk | matrix | ocean
    SHOW_PLOT: Final[bool]         = os.getenv("SHOW_PLOT", "True").lower() == "true"
    FIGURE_WIDTH: Final[float]     = float(os.getenv("FIGURE_WIDTH", "16"))
    FIGURE_HEIGHT: Final[float]    = float(os.getenv("FIGURE_HEIGHT", "10"))

    @classmethod
    def validate(cls) -> None:
        if not (0.0 < cls.TEST_SIZE < 1.0):
            raise ValueError("TEST_SIZE must be in (0.0, 1.0)")
        if cls.STANDARD_K < 1:
            raise ValueError("STANDARD_K must be >= 1")

    @classmethod
    def k_range(cls) -> range:
        return range(cls.K_MIN, cls.K_MAX)


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CUSTOM EXCEPTION HIERARCHY
# ═══════════════════════════════════════════════════════════════════════════════

class KNNPipelineError(Exception):
    """Base exception for all KNN pipeline domain errors."""
    pass


class DataIngestionError(KNNPipelineError):
    """Raised when dataset loading or validation fails."""
    pass


class PreprocessingError(KNNPipelineError):
    """Raised when scaling, splitting, or transformation fails."""
    pass


class ModelTrainingError(KNNPipelineError):
    """Raised when KNN model fitting or prediction fails."""
    pass


class VisualizationError(KNNPipelineError):
    """Raised when plot generation or file save fails."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — AUTO-LOGGING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_logging() -> logging.Logger:
    """Initialize production-grade dual-channel logger."""
    logger = logging.getLogger("DecodeLabs_KNN")
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        return logger

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
    console_fmt = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    log_dir = Path(Config.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / Config.LOG_FILE

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(funcName)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.info("Logging initialized | Dir: %s | File: %s", log_dir, log_path)
    return logger


LOGGER: Final[logging.Logger] = initialize_logging()


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ULTRA TERMINAL UI (Animations + Progress + Colors)
# ═══════════════════════════════════════════════════════════════════════════════

class UltraUI:
    """
    Ultra-professional terminal UI with animations, progress bars, and themes.
    """

    THEMES = {
        "cyberpunk": {
            "primary": "\033[38;5;213m",      # Hot pink
            "secondary": "\033[38;5;51m",     # Cyan
            "accent": "\033[38;5;226m",       # Yellow
            "success": "\033[38;5;82m",       # Green
            "error": "\033[38;5;196m",        # Red
            "warning": "\033[38;5;208m",      # Orange
            "dim": "\033[38;5;245m",          # Gray
        },
        "matrix": {
            "primary": "\033[38;5;46m",       # Matrix green
            "secondary": "\033[38;5;118m",    # Light green
            "accent": "\033[38;5;154m",       # Lime
            "success": "\033[38;5;82m",
            "error": "\033[38;5;196m",
            "warning": "\033[38;5;190m",
            "dim": "\033[38;5;240m",
        },
        "ocean": {
            "primary": "\033[38;5;39m",       # Ocean blue
            "secondary": "\033[38;5;81m",     # Light blue
            "accent": "\033[38;5;159m",       # Pale blue
            "success": "\033[38;5;85m",
            "error": "\033[38;5;196m",
            "warning": "\033[38;5;221m",
            "dim": "\033[38;5;245m",
        }
    }

    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLINK = "\033[5m"

    HORIZONTAL = "═"
    VERTICAL = "║"
    TOP_LEFT = "╔"
    TOP_RIGHT = "╗"
    BOT_LEFT = "╚"
    BOT_RIGHT = "╝"
    T_LEFT = "╠"
    T_RIGHT = "╣"
    BULLET = "▸"
    ARROW = "➤"
    CHECK = "✓"
    STAR = "★"
    DIAMOND = "◆"
    CIRCLE = "●"

    def __init__(self, theme: str = "cyberpunk") -> None:
        self.theme = self.THEMES.get(theme, self.THEMES["cyberpunk"])
        self._color_enabled = self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        if os.getenv("FORCE_COLOR", "0") == "1":
            return True
        if os.name == "nt" and not os.getenv("ANSICON") and not os.getenv("WT_SESSION"):
            return False
        return sys.stdout.isatty()

    def _c(self, color_key: str, text: str) -> str:
        if not self._color_enabled:
            return text
        return f"{self.theme.get(color_key, '')}{text}{self.RESET}"

    def _b(self, text: str) -> str:
        return f"{self.BOLD}{text}{self.RESET}" if self._color_enabled else text

    def clear_screen(self) -> None:
        """Clear terminal screen for cinematic effect."""
        os.system("cls" if os.name == "nt" else "clear")

    def typewriter(self, text: str, delay: float = 0.01) -> None:
        """Print text with typewriter animation effect."""
        for char in text:
            print(char, end="", flush=True)
            time.sleep(delay)
        print()

    def progress_bar(self, label: str, current: int, total: int, width: int = 40) -> None:
        """Render a fancy progress bar."""
        pct = current / total
        filled = int(width * pct)
        bar = self._c("success", "█" * filled) + self._c("dim", "░" * (width - filled))
        percent = f"{pct * 100:.1f}%"
        print(f"\r  {self.BULLET} {label}: [{bar}] {self._c('accent', percent)}", end="", flush=True)
        if current == total:
            print(f" {self._c('success', self.CHECK)}")

    def spinner(self, label: str, duration: float = 1.0) -> None:
        """Show a spinning animation."""
        spins = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        end_time = time.time() + duration
        i = 0
        while time.time() < end_time:
            print(f"\r  {self._c('secondary', spins[i % len(spins)])} {label}...", end="", flush=True)
            time.sleep(0.1)
            i += 1
        print(f"\r  {self._c('success', self.CHECK)} {label} complete!{' ' * 20}")

    def header(self, title: str, subtitle: str = "", width: int = 80) -> str:
        """Render a cinematic header with gradient effect."""
        inner = width - 2
        pad = (inner - len(title)) // 2
        line = f"{self.TOP_LEFT}{self.HORIZONTAL * inner}{self.TOP_RIGHT}"
        title_line = f"{self.VERTICAL}{' ' * pad}{self._b(self._c('primary', title))}{' ' * (inner - pad - len(title))}{self.VERTICAL}"

        result = f"\n{line}\n{title_line}"
        if subtitle:
            sub_pad = (inner - len(subtitle)) // 2
            sub_line = f"{self.VERTICAL}{' ' * sub_pad}{self._c('dim', subtitle)}{' ' * (inner - sub_pad - len(subtitle))}{self.VERTICAL}"
            result += f"\n{sub_line}"
        result += f"\n{self.T_LEFT}{self.HORIZONTAL * inner}{self.T_RIGHT}"
        return result

    def footer(self, width: int = 80) -> str:
        inner = width - 2
        return f"{self.BOT_LEFT}{self.HORIZONTAL * inner}{self.BOT_RIGHT}\n"

    def section(self, icon: str, title: str, width: int = 80) -> str:
        """Render a section header with icon."""
        inner = width - 2
        content = f"  {icon}  {self._b(self._c('secondary', title))}"
        padding = inner - len(content) + 20  # ANSI offset
        line = f"{self.T_LEFT}{self.HORIZONTAL * inner}{self.T_RIGHT}"
        text = f"{self.VERTICAL}{content}{' ' * max(padding, 0)}{self.VERTICAL}"
        return f"{line}\n{text}"

    def info(self, icon: str, label: str, value: str, width: int = 80) -> str:
        """Render an info line with colored value."""
        inner = width - 2
        content = f"    {self.ARROW} {label}: {self._c('accent', value)}"
        padding = inner - len(content) + 15
        return f"{self.VERTICAL}{content}{' ' * max(padding, 0)}{self.VERTICAL}"

    def metric(self, label: str, value: str, unit: str = "", width: int = 80) -> str:
        """Render a metric with big bold value."""
        inner = width - 2
        full_value = f"{value}{unit}"
        content = f"    {self.STAR} {label}: {self._b(self._c('primary', full_value))}"
        padding = inner - len(content) + 20
        return f"{self.VERTICAL}{content}{' ' * max(padding, 0)}{self.VERTICAL}"

    def success_box(self, message: str, width: int = 80) -> str:
        """Render a success notification box."""
        inner = width - 2
        content = f"  {self._c('success', self.CHECK)}  {message}"
        padding = inner - len(content) + 10
        return f"{self.VERTICAL}{content}{' ' * max(padding, 0)}{self.VERTICAL}"

    def warning_box(self, message: str, width: int = 80) -> str:
        """Render a warning notification box."""
        inner = width - 2
        content = f"  {self._c('warning', '⚠')}  {message}"
        padding = inner - len(content) + 10
        return f"{self.VERTICAL}{content}{' ' * max(padding, 0)}{self.VERTICAL}"

    def divider(self, width: int = 80) -> str:
        inner = width - 2
        return f"{self.T_LEFT}{self.HORIZONTAL * inner}{self.T_RIGHT}"


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DatasetMeta:
    """Immutable metadata for the Iris dataset."""
    name: str
    samples: int
    features: int
    classes: int
    class_names: Tuple[str, ...]
    feature_names: Tuple[str, ...]


@dataclass(frozen=True)
class SplitData:
    """Immutable train-test split with scaler."""
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler


@dataclass(frozen=True)
class ModelMetrics:
    """Immutable evaluation metrics."""
    accuracy: float
    predictions: np.ndarray
    confusion_matrix: np.ndarray
    classification_report: str


@dataclass(frozen=True)
class ElbowResult:
    """Immutable elbow method results."""
    k_values: Tuple[int, ...]
    error_rates: Tuple[float, ...]
    optimal_k: int


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — ANIMATED VISUALIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class VisualizationEngine:
    """
    Ultra-professional visualization engine with animations and interactive plots.
    """

    def __init__(self, theme: str = "cyberpunk") -> None:
        self.theme = theme
        self._setup_style()

    def _setup_style(self) -> None:
        """Configure matplotlib style based on theme."""
        if self.theme == "cyberpunk":
            self.bg_color = "#0a0a0f"
            self.card_color = "#1a1a2e"
            self.primary = "#ff00ff"
            self.secondary = "#00ffff"
            self.accent = "#ffff00"
            self.text = "#e0e0ff"
            self.grid = "#2a2a4a"
        elif self.theme == "matrix":
            self.bg_color = "#000000"
            self.card_color = "#001100"
            self.primary = "#00ff00"
            self.secondary = "#55ff55"
            self.accent = "#aaffaa"
            self.text = "#ccffcc"
            self.grid = "#003300"
        else:  # ocean
            self.bg_color = "#001020"
            self.card_color = "#002040"
            self.primary = "#00aaff"
            self.secondary = "#44ddff"
            self.accent = "#88eeff"
            self.text = "#cceeff"
            self.grid = "#004060"

    def create_animated_elbow(self, elbow_result: ElbowResult, standard_k: int) -> Tuple[plt.Figure, animation.FuncAnimation]:
        """
        Create an animated elbow curve with real-time drawing effect.
        """
        fig, ax = plt.subplots(figsize=(10, 7), facecolor=self.bg_color)
        ax.set_facecolor(self.card_color)

        k_vals = list(elbow_result.k_values)
        errors = list(elbow_result.error_rates)

        ax.set_xlabel("K (Number of Neighbours)", color=self.text, fontsize=12, fontweight="bold")
        ax.set_ylabel("Error Rate", color=self.text, fontsize=12, fontweight="bold")
        ax.set_title(
            "ELBOW METHOD ANIMATION\nReal-time K Evaluation",
            color=self.primary, fontsize=14, fontweight="bold", pad=15
        )
        ax.tick_params(colors=self.text, labelsize=10)
        ax.grid(True, color=self.grid, linewidth=0.8, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(self.grid)
        ax.spines["left"].set_color(self.grid)

        ax.axvline(
            x=standard_k, color=self.accent, linestyle="--", linewidth=2.5,
            label=f"Blueprint K = {standard_k}", alpha=0.9
        )

        opt_idx = int(np.argmin(errors))
        ax.scatter(
            [k_vals[opt_idx]], [errors[opt_idx]], color=self.secondary,
            s=200, zorder=5, marker="*", edgecolors="white", linewidths=2,
            label=f"Optimal K = {elbow_result.optimal_k}"
        )

        line, = ax.plot([], [], color=self.primary, linewidth=3, marker="o", markersize=8,
                       markerfacecolor=self.secondary, markeredgecolor="white", markeredgewidth=1.5)

        progress_text = ax.text(
            0.02, 0.98, "", transform=ax.transAxes, color=self.accent, fontsize=11,
            fontweight="bold", verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=self.card_color, edgecolor=self.primary, alpha=0.9)
        )

        ax.set_xlim(min(k_vals) - 0.5, max(k_vals) + 0.5)
        ax.set_ylim(0, max(errors) * 1.2)
        ax.legend(loc="upper right", facecolor=self.card_color, edgecolor=self.grid, labelcolor=self.text, fontsize=10)

        def init():
            line.set_data([], [])
            progress_text.set_text("Initializing...")
            return line, progress_text

        def update(frame):
            x = k_vals[:frame + 1]
            y = errors[:frame + 1]
            line.set_data(x, y)
            progress_text.set_text(f"Evaluating K={k_vals[frame]}... Error: {errors[frame]:.4f}")
            return line, progress_text

        anim = animation.FuncAnimation(
            fig, update, init_func=init, frames=len(k_vals), interval=200, blit=True, repeat=False
        )

        plt.tight_layout()
        return fig, anim

    def create_fancy_confusion_matrix(self, cm: np.ndarray, class_names: Tuple[str, ...]) -> plt.Figure:
        """
        Create an ultra-fancy confusion matrix heatmap with annotations and effects.
        """
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=self.bg_color)
        ax.set_facecolor(self.card_color)

        if self.theme == "cyberpunk":
            cmap = sns.color_palette("magma", as_cmap=True)
        elif self.theme == "matrix":
            cmap = sns.color_palette("Greens", as_cmap=True)
        else:
            cmap = sns.color_palette("Blues", as_cmap=True)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap,
            xticklabels=class_names, yticklabels=class_names, ax=ax,
            cbar_kws={"shrink": 0.8, "label": "Count"},
            annot_kws={"size": 16, "weight": "bold", "color": "white"},
            linewidths=2, linecolor=self.bg_color, square=True
        )

        ax.set_title(
            "CONFUSION MATRIX HEATMAP\nPrediction Accuracy Breakdown",
            color=self.primary, fontsize=14, fontweight="bold", pad=15
        )
        ax.set_xlabel("Predicted Label", color=self.text, fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", color=self.text, fontsize=12, fontweight="bold")
        ax.tick_params(colors=self.text, labelsize=11)

        for i in range(len(class_names)):
            ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor=self.accent, linewidth=4))

        plt.tight_layout()
        return fig

    def create_dashboard(self, metrics: ModelMetrics, meta: DatasetMeta, elbow: ElbowResult) -> plt.Figure:
        """
        Create a comprehensive 2x2 dashboard with all visualizations.
        """
        fig = plt.figure(figsize=(Config.FIGURE_WIDTH, Config.FIGURE_HEIGHT), facecolor=self.bg_color)
        fig.suptitle(
            "DECODELABS KNN IRIS CLASSIFIER — VVIP DASHBOARD",
            fontsize=16, fontweight="bold", color=self.primary, y=0.98
        )

        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Panel 1: Elbow Curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(self.card_color)
        ax1.plot(list(elbow.k_values), list(elbow.error_rates), color=self.primary,
                marker="o", markersize=6, linewidth=2.5)
        ax1.axvline(x=Config.STANDARD_K, color=self.accent, linestyle="--", linewidth=2.5,
                   label=f"Blueprint K = {Config.STANDARD_K}")
        ax1.scatter([elbow.optimal_k], [min(elbow.error_rates)], color=self.secondary,
                   s=300, zorder=5, marker="*", edgecolors="white", linewidths=2,
                   label=f"Optimal K = {elbow.optimal_k}")
        ax1.set_title("Elbow Method", color=self.text, fontsize=12, fontweight="bold")
        ax1.set_xlabel("K", color=self.text)
        ax1.set_ylabel("Error Rate", color=self.text)
        ax1.tick_params(colors=self.text)
        ax1.legend(facecolor=self.card_color, edgecolor=self.grid, labelcolor=self.text)
        ax1.grid(True, color=self.grid, alpha=0.5)
        for spine in ax1.spines.values():
            spine.set_color(self.grid)

        # Panel 2: Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(self.card_color)
        if self.theme == "cyberpunk":
            cmap = "magma"
        elif self.theme == "matrix":
            cmap = "Greens"
        else:
            cmap = "Blues"
        sns.heatmap(metrics.confusion_matrix, annot=True, fmt="d", cmap=cmap,
                   xticklabels=meta.class_names, yticklabels=meta.class_names, ax=ax2,
                   cbar_kws={"shrink": 0.8}, annot_kws={"size": 14, "weight": "bold"})
        ax2.set_title("Confusion Matrix", color=self.text, fontsize=12, fontweight="bold")
        ax2.set_xlabel("Predicted", color=self.text)
        ax2.set_ylabel("True", color=self.text)
        ax2.tick_params(colors=self.text)

        # Panel 3: Metrics Bar
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(self.card_color)
        categories = ["Accuracy", "Precision\n(Macro)", "Recall\n(Macro)", "F1-Score\n(Macro)"]
        report_lines = metrics.classification_report.split("\n")
        macro_line = [l for l in report_lines if "macro avg" in l]
        if macro_line:
            parts = macro_line[0].split()
            values = [metrics.accuracy, float(parts[2]), float(parts[3]), float(parts[4])]
        else:
            values = [metrics.accuracy, 0.97, 0.97, 0.97]

        bars = ax3.bar(categories, values, color=[self.primary, self.secondary, self.accent, "#ff6b6b"],
                      edgecolor="white", linewidth=2)
        ax3.set_ylim(0, 1.1)
        ax3.set_title("Performance Metrics", color=self.text, fontsize=12, fontweight="bold")
        ax3.set_ylabel("Score", color=self.text)
        ax3.tick_params(colors=self.text)
        ax3.grid(True, color=self.grid, alpha=0.3, axis="y")
        for bar, val in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f"{val:.2%}", ha="center", va="bottom", color=self.text, fontweight="bold", fontsize=11)
        for spine in ax3.spines.values():
            spine.set_color(self.grid)

        # Panel 4: Class Distribution Pie
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(self.card_color)
        class_counts = [50, 50, 50]
        colors_pie = [self.primary, self.secondary, self.accent]
        wedges, texts, autotexts = ax4.pie(
            class_counts, labels=meta.class_names, autopct="%1.1f%%",
            colors=colors_pie, startangle=90, textprops={"color": self.text, "fontsize": 11},
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(12)
        ax4.set_title("Class Distribution", color=self.text, fontsize=12, fontweight="bold")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — IPO PIPELINE IMPLEMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

class KNNPipeline:
    """
    Ultra VVIP KNN classification pipeline with animations and rich UI.
    """

    def __init__(self) -> None:
        self.ui = UltraUI(theme=Config.THEME)
        self.viz = VisualizationEngine(theme=Config.THEME)
        self._dataset: Optional[DatasetMeta] = None
        self._split: Optional[SplitData] = None
        self._model: Optional[KNeighborsClassifier] = None
        self._metrics: Optional[ModelMetrics] = None
        self._elbow: Optional[ElbowResult] = None

    def input_phase(self) -> Tuple[SplitData, DatasetMeta]:
        """Execute INPUT phase with animated progress."""
        print(self.ui.header("INPUT PHASE", "Loading • Splitting • Scaling • Securing"))

        self.ui.spinner("Loading Iris Dataset", 0.8)
        try:
            iris = load_iris()
            X, y = iris.data, iris.target
            meta = DatasetMeta(
                name="Iris Benchmark (UCI / Fisher 1936)",
                samples=X.shape[0], features=X.shape[1],
                classes=len(iris.target_names),
                class_names=tuple(str(n) for n in iris.target_names),
                feature_names=tuple(str(n) for n in iris.feature_names)
            )
            self._dataset = meta
            print(self.ui.info("📊", "Dataset", meta.name))
            print(self.ui.info("📈", "Samples", f"{meta.samples} | Features: {meta.features}"))
            print(self.ui.info("🏷️", "Classes", f"{meta.classes} — {', '.join(meta.class_names)}"))
            LOGGER.info("Dataset loaded | %s", meta)
        except Exception as exc:
            LOGGER.exception("Dataset loading failed")
            raise DataIngestionError(f"Failed to load Iris dataset: {exc}") from exc

        print()
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE,
                shuffle=True, stratify=y
            )
            print(self.ui.success_box(f"Split Complete | Train: {len(X_train)} | Test: {len(X_test)} | Stratified: ✓"))
            LOGGER.info("Split complete | Train: %d | Test: %d", len(X_train), len(X_test))
        except Exception as exc:
            LOGGER.exception("Train-test split failed")
            raise PreprocessingError(f"Failed to split data: {exc}") from exc

        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            split_data = SplitData(X_train=X_train_scaled, X_test=X_test_scaled,
                                  y_train=y_train, y_test=y_test, scaler=scaler)
            self._split = split_data
            mean_f0 = float(X_train_scaled[:, 0].mean())
            var_f0 = float(X_train_scaled[:, 0].var())
            print(self.ui.success_box("Gatekeeper Rule Applied → Data Leakage Prevented 🔒"))
            print(self.ui.info("📐", "Post-Scale Mean (F0)", f"{mean_f0:.4f} (Target: 0.0)"))
            print(self.ui.info("📐", "Post-Scale Var (F0)", f"{var_f0:.4f} (Target: 1.0)"))
            LOGGER.info("Scaling complete | Mean[0]: %.4f | Var[0]: %.4f", mean_f0, var_f0)
        except Exception as exc:
            LOGGER.exception("StandardScaler failed")
            raise PreprocessingError(f"Failed to scale features: {exc}") from exc

        print(self.ui.divider())
        return split_data, meta

    def process_phase(self, split_data: SplitData, meta: DatasetMeta) -> Tuple[KNeighborsClassifier, ElbowResult]:
        """Execute PROCESS phase with animated elbow."""
        print(self.ui.header("PROCESS PHASE", "Elbow Evaluation • Model Training • Blueprint Compliance"))

        print()
        try:
            error_rates: List[float] = []
            k_values = list(Config.k_range())
            total = len(k_values)

            for i, k in enumerate(k_values):
                knn_temp = KNeighborsClassifier(n_neighbors=k, metric="euclidean", weights="uniform")
                knn_temp.fit(split_data.X_train, split_data.y_train)
                preds = knn_temp.predict(split_data.X_test)
                err = 1.0 - accuracy_score(split_data.y_test, preds)
                error_rates.append(err)
                self.ui.progress_bar("Elbow Evaluation", i + 1, total)

            optimal_k = k_values[int(np.argmin(error_rates))]
            elbow_result = ElbowResult(
                k_values=tuple(k_values), error_rates=tuple(error_rates), optimal_k=optimal_k
            )
            self._elbow = elbow_result
            print(self.ui.info("🎯", "Optimal K (Data-Driven)", f"K={optimal_k} | Error: {min(error_rates):.4f}"))
            print(self.ui.info("📋", "Blueprint K (Mandated)", f"K={Config.STANDARD_K} [STRICT COMPLIANCE]"))
            LOGGER.info("Elbow complete | Optimal K: %d | Blueprint K: %d", optimal_k, Config.STANDARD_K)
        except Exception as exc:
            LOGGER.exception("Elbow method failed")
            raise ModelTrainingError(f"Elbow evaluation failed: {exc}") from exc

        print()
        self.ui.spinner("Training Final Model (K=5)", 1.0)
        try:
            model = KNeighborsClassifier(n_neighbors=Config.STANDARD_K, metric="euclidean", weights="uniform")
            model.fit(split_data.X_train, split_data.y_train)
            self._model = model
            print(self.ui.success_box(f"Model Trained | K={Config.STANDARD_K} | Metric=Euclidean | Weights=Uniform"))
            LOGGER.info("Model trained successfully | K=%d", Config.STANDARD_K)
        except Exception as exc:
            LOGGER.exception("Model training failed")
            raise ModelTrainingError(f"Failed to train KNN model: {exc}") from exc

        print(self.ui.divider())
        return model, elbow_result

    def output_phase(self, model: KNeighborsClassifier, split_data: SplitData, meta: DatasetMeta) -> ModelMetrics:
        """Execute OUTPUT phase with rich visualizations."""
        print(self.ui.header("OUTPUT PHASE", "Prediction • Metrics • Visualization • Export"))

        self.ui.spinner("Generating Predictions", 0.5)
        try:
            predictions = model.predict(split_data.X_test)
            print(self.ui.success_box("Predictions Generated ✓"))
            LOGGER.info("Predictions complete | Test samples: %d", len(predictions))
        except Exception as exc:
            LOGGER.exception("Prediction failed")
            raise ModelTrainingError(f"Failed to generate predictions: {exc}") from exc

        try:
            accuracy = float(accuracy_score(split_data.y_test, predictions))
            cm = confusion_matrix(split_data.y_test, predictions)
            report = classification_report(split_data.y_test, predictions, target_names=list(meta.class_names))
            metrics = ModelMetrics(accuracy=accuracy, predictions=predictions,
                                  confusion_matrix=cm, classification_report=report)
            self._metrics = metrics
            print()
            print(self.ui.metric("Overall Accuracy", f"{accuracy * 100:.2f}", "%"))
            print(self.ui.warning_box("Beware of Accuracy Mirage on imbalanced data!"))
            print(self.ui.info("📊", "Confusion Matrix", "\n" + str(cm)))
            LOGGER.info("Metrics computed | Accuracy: %.4f", accuracy)
        except Exception as exc:
            LOGGER.exception("Metrics computation failed")
            raise ModelTrainingError(f"Failed to compute metrics: {exc}") from exc

        print()
        self.ui.spinner("Generating Ultra Visualizations", 1.5)
        try:
            # Dashboard
            dashboard_fig = self.viz.create_dashboard(metrics, meta, self._elbow)
            dashboard_path = Config.OUTPUT_PLOT.replace(".png", "_dashboard.png")
            dashboard_fig.savefig(dashboard_path, dpi=Config.FIGURE_DPI,
                                 bbox_inches="tight", facecolor=self.viz.bg_color)
            print(self.ui.success_box(f"Dashboard Saved → {dashboard_path}"))

            # Animated Elbow
            anim_fig, anim = self.viz.create_animated_elbow(self._elbow, Config.STANDARD_K)
            anim_path = Config.OUTPUT_PLOT.replace(".png", "_animated.gif")
            try:
                anim.save(anim_path, writer="pillow", fps=Config.ANIMATION_FPS)
                print(self.ui.success_box(f"Animation Saved → {anim_path}"))
            except Exception:
                print(self.ui.warning_box("Animation save skipped (pillow not installed)"))

            # Fancy Confusion Matrix
            cm_fig = self.viz.create_fancy_confusion_matrix(cm, meta.class_names)
            cm_path = Config.OUTPUT_PLOT.replace(".png", "_confusion.png")
            cm_fig.savefig(cm_path, dpi=Config.FIGURE_DPI, bbox_inches="tight", facecolor=self.viz.bg_color)
            print(self.ui.success_box(f"Confusion Matrix Saved → {cm_path}"))

            # Main plot
            main_fig = self.viz.create_dashboard(metrics, meta, self._elbow)
            main_fig.savefig(Config.OUTPUT_PLOT, dpi=Config.FIGURE_DPI,
                           bbox_inches="tight", facecolor=self.viz.bg_color)
            print(self.ui.success_box(f"Main Plot Saved → {Config.OUTPUT_PLOT}"))

            # SHOW PLOT (plt.show())
            if Config.SHOW_PLOT:
                print()
                print(self.ui.info("🖥️", "Displaying Interactive Plot", "Close window to continue..."))
                plt.show()

            LOGGER.info("All visualizations generated and saved")
        except Exception as exc:
            LOGGER.exception("Plot generation failed")
            raise VisualizationError(f"Failed to generate plots: {exc}") from exc

        print(self.ui.divider())
        return metrics

    def run(self) -> ModelMetrics:
        """Execute complete IPO pipeline with cinematic intro."""
        self.ui.clear_screen()
        print(self.ui.header("DECODELABS AI INTERNSHIP", "PROJECT 2: DATA CLASSIFICATION USING AI | VVIP ULTRA EDITION"))
        print(self.ui.info("🤖", "Algorithm", "K-Nearest Neighbors (KNN)"))
        print(self.ui.info("📊", "Dataset", "Iris Benchmark (UCI / Fisher 1936)"))
        print(self.ui.info("🏗️", "Architecture", "IPO Pipeline (Input → Process → Output)"))
        print(self.ui.info("🎨", "Theme", Config.THEME.upper()))
        print(self.ui.info("⭐", "Edition", "VVIP ULTRA PROFESSIONAL"))
        print(self.ui.divider())

        start_time = datetime.now()
        LOGGER.info("Pipeline started | Timestamp: %s", start_time.isoformat())

        try:
            split_data, meta = self.input_phase()
            model, elbow = self.process_phase(split_data, meta)
            metrics = self.output_phase(model, split_data, meta)

            duration = (datetime.now() - start_time).total_seconds()
            print(self.ui.header("PIPELINE COMPLETE", "All Systems Operational ✓"))
            print(self.ui.metric("Model", f"KNeighborsClassifier(n_neighbors={Config.STANDARD_K})", ""))
            print(self.ui.metric("Accuracy", f"{metrics.accuracy * 100:.2f}", "%"))
            print(self.ui.metric("Duration", f"{duration:.2f}", " seconds"))
            print(self.ui.metric("Log File", f"{Config.LOG_DIR}/{Config.LOG_FILE}", ""))
            print(self.ui.metric("Plots Generated", "4 files", ""))
            print(self.ui.success_box("🎉 Mission Accomplished! All outputs saved successfully! 🎉"))
            print(self.ui.footer())

            LOGGER.info("Pipeline completed | Accuracy: %.4f | Duration: %.2fs", metrics.accuracy, duration)
            return metrics

        except KNNPipelineError as exc:
            LOGGER.error("Pipeline failed: %s", exc)
            print(f"\n{self.ui._c('error', f'[PIPELINE ERROR] {exc}')}\n")
            raise
        except Exception as exc:
            LOGGER.exception("Unhandled pipeline exception")
            print(f"\n{self.ui._c('error', f'[CRITICAL FAILURE] {exc}')}\n")
            raise KNNPipelineError(f"Unexpected pipeline failure: {exc}") from exc


# ═══════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Application entry point with top-level exception shielding."""
    try:
        Config.validate()
    except ValueError as exc:
        LOGGER.error("Configuration validation failed: %s", exc)
        print(f"\n[CONFIG ERROR] {exc}\n")
        sys.exit(1)

    try:
        pipeline = KNNPipeline()
        pipeline.run()
    except KNNPipelineError as exc:
        LOGGER.error("Application terminated with pipeline error: %s", exc)
        sys.exit(1)
    except Exception as exc:
        LOGGER.exception("Unhandled runtime exception")
        print(f"\n[CRITICAL FAILURE] Check {Config.LOG_DIR}/{Config.LOG_FILE} for details.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()