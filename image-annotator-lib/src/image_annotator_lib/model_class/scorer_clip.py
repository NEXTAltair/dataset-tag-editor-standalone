"""CLIP ベースの Scorer モデルの実装。"""

import logging
from typing import Any

from ..core.base import ClipBaseAnnotator

logger = logging.getLogger(__name__)

# --- Improved Aesthetic / Waifu Aesthetic ---


class ImprovedAesthetic(ClipBaseAnnotator):
    """Improved Aesthetic Predictor v2 モデル。"""

    def __init__(self, model_name: str, **kwargs: Any):
        """ImprovedAestheticPredictor を初期化します。"""
        # ClipBaseAnnotator の __init__ で必要な設定は読み込まれる
        super().__init__(model_name=model_name, **kwargs)
        logger.debug(f"ImprovedAestheticPredictor '{model_name}' initialized.")

    def _get_score_tag(self, score: float) -> str:
        """スコアをタグ形式の文字列に変換します (例: [IAP]score_7)。"""
        score_int = max(1, min(round(score), 10))
        return f"[IAP]score_{score_int}"


class WaifuAesthetic(ClipBaseAnnotator):
    """Waifu Diffusion Aesthetic Predictor v2 モデル。"""

    def __init__(self, model_name: str, **kwargs: Any):
        """WaifuAestheticPredictor を初期化します。"""
        super().__init__(model_name=model_name, **kwargs)
        logger.debug(f"WaifuAestheticPredictor '{model_name}' initialized.")

    def _get_score_tag(self, score: float) -> str:
        """スコアをタグ形式の文字列に変換します (例: [WAIFU]score_7)。"""
        # スコアを 0-10 の範囲の整数に変換
        # FIXME: 計算式がおかしい後で直す
        score_int = max(0, min(round(score * 10), 100))
        return f"[WAIFU]score_{score_int}"
