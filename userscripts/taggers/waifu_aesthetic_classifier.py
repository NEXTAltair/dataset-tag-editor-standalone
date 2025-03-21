from PIL import Image
import torch
import numpy as np
import math

from transformers import CLIPModel, CLIPProcessor

from scripts import model_loader, devices, settings
from scripts.paths import paths
from scripts.tagger import Tagger

# brought from https://github.com/waifu-diffusion/aesthetic/blob/main/aesthetic.py
class Classifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.fc3 = torch.nn.Linear(hidden_size//2, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x:torch.Tensor):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

# brought and modified from https://github.com/waifu-diffusion/aesthetic/blob/main/aesthetic.py
def image_embeddings(image:Image, model:CLIPModel, processor:CLIPProcessor):
    inputs = processor(images=image, return_tensors='pt')['pixel_values']
    inputs = inputs.to(devices.device)
    result:np.ndarray = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    return (result / np.linalg.norm(result)).squeeze(axis=0)


class WaifuAesthetic(Tagger):
    def load(self):
        file = model_loader.load(
            model_path=paths.models_path / "aesthetic" / "aes-B32-v0.pth",
            model_url='https://huggingface.co/hakurei/waifu-diffusion-v1-4/resolve/main/models/aes-B32-v0.pth'
        )
        CLIP_REPOS = 'openai/clip-vit-base-patch32'
        self.model = Classifier(512, 256, 1)
        self.model.load_state_dict(torch.load(file))
        self.model = self.model.to(devices.device)
        self.clip_processor = CLIPProcessor.from_pretrained(CLIP_REPOS)
        self.clip_model = CLIPModel.from_pretrained(CLIP_REPOS).to(devices.device).eval()

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

    def _get_score(self, prediction):
        """モデルの生出力をタグ形式に変換"""
        score = math.floor(prediction.item() * 10)  # 0-1のスコアを0-10のスケールに変換して切り捨て
        return [f"[WAIFU]score_{score}"]

    def predict(self, image: Image.Image, threshold=None):
        image_embeds = image_embeddings(image, self.clip_model, self.clip_processor)
        prediction:torch.Tensor = self.model(torch.from_numpy(image_embeds).float().to(devices.device))
        # edit here to change tag
        if self._is_wrapper_call(): # ScorerWrapper経由の呼び出しの場合
            return (prediction.item(), self._get_score(prediction))
        return self._get_score(prediction)

    def predict_pipe(self, data: list[Image.Image], threshold=None):
        if data is None:
            return

        for image in data:
            # 現在のpredictメソッドの処理を流用
            image_embeds = image_embeddings(image, self.clip_model, self.clip_processor)
            prediction = self.model(torch.from_numpy(image_embeds).float().to(devices.device))
            if self._is_wrapper_call():
                yield (prediction.item(), self._get_score(prediction))  # 生のスコア（0-1）とタグのリスト
            else:
                yield self._get_score(prediction)

    def name(self):
        return "wd aesthetic classifier"
