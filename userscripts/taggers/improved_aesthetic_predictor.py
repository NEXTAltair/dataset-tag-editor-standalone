import math

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from scripts import devices, model_loader, settings
from scripts.paths import paths
from scripts.tagger import Tagger


# brought from https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py and modified
class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


# brought and modified from https://github.com/waifu-diffusion/aesthetic/blob/main/aesthetic.py
def image_embeddings(image: Image, model: CLIPModel, processor: CLIPProcessor):
    inputs = processor(images=image, return_tensors="pt")["pixel_values"]
    inputs = inputs.to(devices.device)
    result: np.ndarray = (
        model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    )
    return (result / np.linalg.norm(result)).squeeze(axis=0)


class ImprovedAestheticPredictor(Tagger):
    def load(self):
        MODEL_VERSION = "sac+logos+ava1-l14-linearMSE"
        file = model_loader.load(
            model_path=paths.models_path / "aesthetic" / f"{MODEL_VERSION}.pth",
            model_url=f"https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/{MODEL_VERSION}.pth",
        )
        CLIP_REPOS = "openai/clip-vit-large-patch14"
        self.model = Classifier(768)
        self.model.load_state_dict(torch.load(file))
        self.model = self.model.to(devices.device)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_REPOS)
        self.clip_model = (
            CLIPModel.from_pretrained(CLIP_REPOS).to(devices.device).eval()
        )

    def unload(self):
        if not settings.current.interrogator_keep_in_memory:
            self.model = None
            self.clip_processor = None
            self.clip_model = None
            devices.torch_gc()

    def start(self):
        self.load()
        return self

    def stop(self):
        self.unload()

    def _get_score(self, data):
        data: torch.Tensor = self.model(
            torch.from_numpy(data).float().to(devices.device)
        )
        return [f"[IAP]score_{math.floor(data.item())}"]

    def predict(self, image: Image.Image, threshold=None):
        data = image_embeddings(image, self.clip_model, self.clip_processor)
        if self._is_wrapper_call():
            return (data, self._get_score(data))
        return self._get_score(data)

    def predict_pipe(self, images: list[Image.Image], batch_size=32, threshold=None):
        """ NOTE: トランスフォーマーのPipelineが使えないのでバッチ処理はここで行う

        Args:
            images: 処理する画像のリスト
            batch_size: バッチサイズ（デフォルト: 32）
            threshold: 閾値（未使用）

        Yields:
            特徴量または特徴量とスコアのタプル
        """
        # NOTE: dte_logic.py の load_dataset メソッドは、
        # predict_pipe メソッドが実装されているかをテストするために、最初に None を引数として呼び出している
        # そのため、None が渡された場合は32で上書きする
        # load_dataset はなぜこんな処理をしてるのかわからないからいじらない
        if batch_size is None:
            batch_size = 32

        def chunks(lst: list, n: int):
            """リストをn個ずつのバッチに分割するジェネレータ

            Args:
                lst: 分割するリスト
                n: バッチサイズ

            Yields:
                バッチ単位のリスト
            """
            if lst is None or not lst:
                return []
            return [lst[i:i + n] for i in range(0, len(lst), n)]

        if images is None or not isinstance(images, list) or not images:
            print("Invalid input: empty, None, or non-list input")
            return

        for batch in chunks(images, batch_size):
            # バッチ内の画像を同時に処理
            inputs = self.clip_processor(
                images=batch, return_tensors="pt", padding=True
            )["pixel_values"]
            inputs = inputs.to(devices.device)

            # CLIPで特徴量を一括抽出
            features = self.clip_model.get_image_features(pixel_values=inputs)
            features = features.cpu().detach().numpy()

            # 正規化
            features = features / np.linalg.norm(features, axis=1, keepdims=True)

            # バッチ内の各画像について処理
            for feature in features:
                if self._is_wrapper_call():
                    yield (feature, self._get_score(feature))
                else:
                    yield self._get_score(feature)

    def name(self):
        return "Improved Aesthetic Predictor"
