# Scorerと違って要らなそうだが､一応registerの動作に必要なので定義しておく
# OPTIMIZE: このファイルは削除してもいいかも
# TODO: リファクタリング候補
from tagger_wrapper_lib.core.base import ONNXModel


class WDTagger(ONNXModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
