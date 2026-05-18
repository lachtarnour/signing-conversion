from __future__ import annotations

import logging
import warnings


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def setup_warnings() -> None:
    """Silence known-benign third-party warnings (torchaudio stft resize, HF Hub anon)."""
    warnings.filterwarnings(
        "ignore",
        message="An output with one or more elements was resized since it had shape",
        category=UserWarning,
    )
