# This code is using the image classification "aesthetic-shadow-v2" by shadowlilac (https://huggingface.co/shadowlilac/aesthetic-shadow-v2)
# and "aesthetic-shadow-v2" is licensed under CC-BY-NC 4.0 (https://spdx.org/licenses/CC-BY-NC-4.0)
# 配布元公開停止に伴い、手持ちのModelミラー(https://huggingface.co/NEXTAltair/cache_aestheic-shadow-v2)したものに移行しました。

from PIL import Image
from transformers import pipeline

from scripts import devices, settings
from scripts.tagger import Tagger

# brought and modified from https://huggingface.co/spaces/cafeai/cafe_aesthetic_demo/blob/main/app.py

# I'm not sure if this is really working
BATCH_SIZE = 1

# tags used in Animagine-XL
SCORE_N = {
    'very aesthetic': 0.71,
    'aesthetic': 0.45,
    'displeasing': 0.27,
    'very displeasing': -float('inf'),
}

def get_aesthetic_tag(score: float):
    for k, v in SCORE_N.items():
        if score > v:
            return f"v2_{k}"

class AestheticShadowV2(Tagger):
    def load(self):
        self.pipe_aesthetic = pipeline("image-classification", "NEXTAltair/cache_aestheic-shadow-v2", device=devices.device, batch_size=BATCH_SIZE)

    def unload(self):
        if not settings.current.interrogator_keep_in_memory:
            self.pipe_aesthetic = None
            devices.torch_gc()

    def start(self):
        self.load()
        return self

    def stop(self):
        self.unload()

    def _get_score(self, data):
        final = {}
        for d in data:
            final[d["label"]] = d["score"]
        hq = final['hq']
        return [get_aesthetic_tag(hq)]

    def predict(self, image: Image.Image, threshold=None):
        data = self.pipe_aesthetic(image)
        if self._is_wrapper_call(): # ScorerWrapper経由の呼び出しの場合
            return (data, self._get_score(data))
        return self._get_score(data)

    def predict_pipe(self, data: list[Image.Image], threshold=None):
        if data is None:
            return
        for out in self.pipe_aesthetic(data, batch_size=BATCH_SIZE):
            if self._is_wrapper_call(): # ScorerWrapper経由の呼び出しの場合
                yield (out, self._get_score(out))
            yield self._get_score(out)

    def name(self):
        return "aesthetic shadow v2"