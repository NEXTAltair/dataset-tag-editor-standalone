from scorer_wrapper_lib.core.base import PipelineModel


class CafePredictor(PipelineModel):
    """
    Cafe Aesthetic モデルのラッパークラス

    HuggingFace の "model_name/cafe_aesthetic" モデルを利用して画像の美的スコアを計算する。
    """

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)

    def _calculate_score(self, raw_output: list[dict]) -> float:
        """
        モデルの出力から審美的スコアを計算する

        Args:
            raw_output (list[dict]]): 例
                [{'label': 'aesthetic', 'score': 0.67}, {'label': 'not_aesthetic', 'score': 0.06}]

        Returns:
            float: 'aesthetic' ラベルのスコア値
        """
        score_by_label = {}
        for entry in raw_output:
            score_by_label[entry["label"]] = entry["score"]
        score = score_by_label["aesthetic"]
        return score

    def _get_score_tag(self, score: float) -> str:
        return f"{self.config['score_prefix']}score_{int(score * 10)}"
