"""Pipeline ベースの Aesthetic Score モデルの実装。"""

import logging
from typing import Any

# PipelineBaseAnnotator をインポート
from image_annotator_lib.core.base import PipelineBaseAnnotator

# AnnotationResult は型ヒントで使用する可能性があるため残す (が、直接は使わない)

logger = logging.getLogger(__name__)


class AestheticShadow(PipelineBaseAnnotator):
    """Aesthetic Shadow モデル (v1, v2) の共通処理を含む基底クラス (内部用)。

    Hugging Face Pipeline を使用して美的スコアを計算します。
    """

    def __init__(self, model_name: str):  # kwargs は不要
        """AestheticShadow を初期化します。"""
        super().__init__(model_name=model_name)
        logger.debug(f"AestheticShadow '{model_name}' initialized.")
        self.SCORE_THRESHOLDS = {
            "very aesthetic": 0.71,
            "aesthetic": 0.45,
            "displeasing": 0.27,
        }

    def _format_predictions(self, raw_outputs: list[list[dict[str, Any]]]) -> list[dict[str, float]]:
        """Pipeline の生出力リストから、各画像のhqとlqスコアを辞書形式で返します。

        Args:
            raw_outputs: モデルからの生の出力リスト
                例: [[{'label': 'hq', 'score': 0.9}, {'label': 'lq', 'score': 0.1}], ...]

        Returns:
            list[dict[str, float]]: 各画像のスコアを含む辞書のリスト
                例: [{'hq': 0.9, 'lq': 0.1}, ...]
        """
        scores = []
        for single_output in raw_outputs:
            final_scores: dict[str, float] = {}

            for item in single_output:
                label = item["label"]
                if label in ["hq", "lq"]:
                    final_scores[label] = float(item["score"])

            scores.append(final_scores)

        return scores

    def _generate_tags(self, formatted_score: dict[str, float]) -> list[str]:
        """スコア値に基づいてスコアタグを返します。

        Args:
            formatted_score: hqとlqのスコアを含む辞書

        Returns:
            list[str]: 生成されたタグのリスト
        """
        if "hq" in formatted_score:
            hq_score = formatted_score["hq"]
            for tag, threshold in self.SCORE_THRESHOLDS.items():
                if hq_score >= threshold:
                    return [tag]
        return ["very displeasing"]


# --- Cafe Aesthetic ---
class CafePredictor(PipelineBaseAnnotator):
    """Cafe Aesthetic モデル ("model_name/cafe_aesthetic")。

    Hugging Face Pipeline を使用して美的スコアを計算します。
    """

    def __init__(self, model_name: str):
        """CafePredictor を初期化します。"""
        super().__init__(model_name=model_name)
        self.score_prefix = "[CAFE]_"
        logger.debug(f"CafePredictor '{model_name}' initialized with prefix: '{self.score_prefix}'")

    # _format_predictions を実装 (PipelineBaseAnnotator の抽象メソッド)
    def _format_predictions(self, raw_outputs: list[list[dict[str, Any]]]) -> list[float]:
        """Pipeline の生出力リストから、各画像の最終スコア (float) のリストを生成します。
        'aesthetic' ラベルのスコアを抽出します。
        """
        scores = []
        for single_output in raw_outputs:
            # 各画像の出力 (例: [{'label': 'aesthetic', 'score': 0.67}, {'label': 'not_aesthetic', 'score': 0.33}])
            # から 'aesthetic' スコアを抽出するロジック
            if not isinstance(single_output, list):
                logger.warning(
                    f"予期しない single_output の型: {type(single_output)}。スコア 0.0 を使用します。"
                )
                scores.append(0.0)
                continue

            score_found = False
            for entry in single_output:
                if isinstance(entry, dict) and entry.get("label") == "aesthetic" and "score" in entry:
                    try:
                        scores.append(float(entry["score"]))
                        score_found = True
                        break  # aesthetic スコアが見つかったらループを抜ける
                    except (TypeError, ValueError):
                        self.logger.error(f"モデルからの戻り値 'aesthetic' のスコアが不正です: {entry}")
                        scores.append(0.0)
                        score_found = True
                        break
            if not score_found:
                # 'aesthetic' ラベルが見つからなかった場合
                self.logger.warning(f"出力に 'aesthetic' ラベルが見つかりませんでした: {single_output}")
                scores.append(0.0)

        return scores

    # _calculate_score メソッドは _format_predictions に統合されたため削除

    # _generate_tags は BaseAnnotator._generate_results から呼び出される
    def _generate_tags(self, score: float) -> list[str]:
        """スコア値に基づいてスコアタグを生成します (例: cafe_score_6)。"""
        score_level = int(score * 10)  # 0-1のスコアを0-10のスケールに変換して切り捨て
        return [f"{self.score_prefix}score_{score_level}"]


class AestheticShadowV2(AestheticShadow):
    """Aesthetic Shadow V2 モデル ("NEXTAltair/cache_aestheic-shadow-v2")。"""

    # model_name は config ファイルで "aesthetic-shadow-v2" のように定義されることを想定
    pass  # 実装は AestheticShadow に依存


# 他の Aesthetic 系モデルも同様に PipelineBaseAnnotator を継承して実装可能
