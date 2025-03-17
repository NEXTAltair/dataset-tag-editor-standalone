from scorer_wrapper_lib.core.base import ClipModel


class ImprovedAesthetic(ClipModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)

    def _get_score_tag(self, score_float: float) -> str:
        score = int(score_float)
        return f"[IAP]score_{score}"


class WaifuAesthetic(ClipModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)

    def _get_score_tag(self, score_float: float) -> str:
        score = int(score_float * 10)
        return f"[WAIFU]score_{score}"
