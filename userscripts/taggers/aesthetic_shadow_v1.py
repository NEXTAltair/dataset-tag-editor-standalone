# This code is using the image classification "aesthetic-shadow" by shadowlilac (https://huggingface.co/shadowlilac/aesthetic-shadow)
# and "aesthetic-shadow" is licensed under CC-BY-NC 4.0 (https://spdx.org/licenses/CC-BY-NC-4.0)
# v2配布元公開停止に伴い、v1を使用する

from PIL import Image
from transformers import pipeline

from scripts import devices, settings
from scripts.tagger import Tagger

# brought and modified from https://huggingface.co/spaces/cafeai/cafe_aesthetic_demo/blob/main/app.py

# I'm not sure if this is really working
BATCH_SIZE = 1

# tags used in Animagine-XL
# TODO: Animation-XLはv2のスコア基準なので値が異なる
SCORE_N = {
    'very aesthetic': 0.71,
    'aesthetic': 0.45,
    'displeasing': 0.27,
    'very displeasing': -float('inf'),
}

def get_aesthetic_tag(score: float):
    for k, v in SCORE_N.items():
        if score > v:
            return k

class AestheticShadow(Tagger):
    def load(self):
        self.pipe_aesthetic = pipeline("image-classification", "shadowlilac/aesthetic-shadow", device=devices.device, batch_size=BATCH_SIZE)

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
            return data
        return self._get_score(data)  # 元の処理

    def predict_pipe(self, data: list[Image.Image], threshold=None):
        if data is None:
            return
        for out in self.pipe_aesthetic(data, batch_size=BATCH_SIZE):
            if self._is_wrapper_call(): # ScorerWrapper経由の呼び出しの場合
                yield out
            else:
                yield self._get_score(out)  # 元の処理

    def name(self):
        return "aesthetic shadow v1"