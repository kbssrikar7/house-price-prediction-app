"""Utility functions for the ML portfolio project."""

import logging
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt


def create_directory_structure(base_path: Path = Path(".")) -> None:
    """
    Create the standard project directory structure.

    Args:
        base_path: Root directory for the project
    """
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src",
        "models",
        "reports",
        "figures",
        "tests",
    ]

    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create __init__.py files
    (base_path / "src" / "__init__.py").touch()
    (base_path / "tests" / "__init__.py").touch()

    logging.info(f"Created directory structure at {base_path}")


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Configure logging for the project.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("ml_portfolio")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (optional)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def save_figure(
    fig,
    filename: str,
    output_dir: Path = Path("figures/"),
    formats: List[str] = None
) -> None:
    """
    Save a matplotlib figure to disk.

    Args:
        fig: Figure object to save
        filename: Name of the file (without extension)
        output_dir: Directory to save figures
        formats: List of formats to save (png, pdf, svg)
    """
    if formats is None:
        formats = ["png"]

    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        logging.info(f"Saved figure to {filepath}")


def get_feature_names(
    pipeline,
    original_features: List[str]
) -> List[str]:
    """
    Extract feature names from a fitted sklearn pipeline.

    Args:
        pipeline: Fitted sklearn Pipeline with ColumnTransformer
        original_features: Original feature column names

    Returns:
        List of transformed feature names
    """
    try:
        # For sklearn >= 1.0
        return pipeline.get_feature_names_out().tolist()
    except AttributeError:
        # Fallback for older versions
        return original_features
