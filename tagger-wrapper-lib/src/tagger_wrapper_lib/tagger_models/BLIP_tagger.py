# Scorerと違って要らなそうだが､一応registerの動作に必要なので定義しておく
from tagger_wrapper_lib.core.base import TransformerModel


class BLIPTagger(TransformerModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)


class BLIP2Tagger(TransformerModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)


class GITTagger(TransformerModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
