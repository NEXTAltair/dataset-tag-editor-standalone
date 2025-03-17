from scorer_wrapper_lib.core.base import PipelineModel

SCORE_THRESHOLDS = {
    "very aesthetic": 0.71,
    "aesthetic": 0.45,
    "displeasing": 0.27,
}


class AestheticShadowV1(PipelineModel):
    """
    Aesthetic Shadow V1 モデルのラッパークラス

    HuggingFace の "shadowlilac/aesthetic-shadow" モデルを利用して画像の美的スコアを計算する。
    このモデルは、'hq' と 'lq' の 2 種類のラベルのスコアを出力
    """

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)

    def _calculate_score(self, raw_output: list[dict]) -> float:
        # raw_output は list[dict] (例: [{'label': 'hq', 'score': 0.76}, ...]) を期待
        final = {}
        for item in raw_output:
            final[item["label"]] = item["score"]
        hq = final["hq"]
        return hq

    def _get_score_tag(self, score: float) -> str:
        for k, v in SCORE_THRESHOLDS.items():
            if score > v:
                return k
        return "very displeasing"


class AestheticShadowV2(PipelineModel):
    """
    Aesthetic Shadow V2 モデルのラッパークラス

    HuggingFace の "NEXTAltair/cache_aestheic-shadow-v2" モデルを利用して画像の美的スコアを計算する。
    """

    # model_path = "NEXTAltair/cache_aestheic-shadow-v2"

    def __init__(self, model_name: str):
        """コンストラクタ"""
        super().__init__(model_name=model_name)

    def _calculate_score(self, raw_output: list[dict]) -> float:
        # raw_output は常に list[dict] (例: [{'label': 'hq', 'score': 0.76}, ...]) を期待
        final = {}
        for item in raw_output:
            final[item["label"]] = item["score"]
        hq = final["hq"]
        return hq

    def _get_score_tag(self, score: float) -> str:
        for k, v in SCORE_THRESHOLDS.items():
            if score > v:
                return k
        return "very displeasing"


if __name__ == "__main__":
    # 簡易テストコード
    config = {"device": "cpu"}
    model_name = "test_model"
    dummy_output = [{"label": "hq", "score": 0.8}, {"label": "lq", "score": 0.2}]

    print("Testing AestheticShadowV1")
    model_v1 = AestheticShadowV1(model_name)
    score_v1 = model_v1._calculate_score(dummy_output)
    tag_v1 = model_v1._get_score_tag(score_v1)
    print(f"Score V1: {score_v1}, Tag V1: {tag_v1}")

    print("Testing AestheticShadowV2")
    model_v2 = AestheticShadowV2(model_name)
    score_v2 = model_v2._calculate_score(dummy_output)
    tag_v2 = model_v2._get_score_tag(score_v2)
    print(f"Score V2: {score_v2}, Tag V2: {tag_v2}")
