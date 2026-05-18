from __future__ import annotations

import logging
import warnings


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def setup_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"An output with one or more elements was resized since it had shape.*",
        category=UserWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You are sending unauthenticated requests to the HF Hub.*",
    )
