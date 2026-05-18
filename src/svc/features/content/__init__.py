from __future__ import annotations

from svc.features.content.base import ContentEncoder, ContentEncoderConfig
from svc.features.content.hubert_soft import HuBERTSoftEncoder

__all__ = ["ContentEncoder", "ContentEncoderConfig", "HuBERTSoftEncoder", "build_content_encoder"]


def build_content_encoder(
    name: str,
    pretrained: bool = True,
    device: str | None = None,
) -> ContentEncoder:
    normalized = name.replace("-", "_").lower()
    if normalized == "hubert_soft":
        return HuBERTSoftEncoder(pretrained=pretrained, device=device)
    raise ValueError(f"Unknown content encoder: {name}")
