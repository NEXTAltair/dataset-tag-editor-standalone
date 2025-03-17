# """
# * Adapted from BLIP (https://github.com/salesforce/BLIP)
# * Using transformers library for BLIP model implementation
# """

# import json
# import logging
# from typing import Any, Dict, List, Optional

# import torch
# import torch.nn as nn
# from PIL import Image
# from transformers import (
#     BlipModel,
#     BlipProcessor,
# )

# from ..core.base import BaseScorer
# from ..core.utils import load_file, load_model_config

# logger = logging.getLogger(__name__)


# class MLP(nn.Module):
#     """
#     ImageRewardで使用されるMLPモデル。
#     特徴量からスコアを計算するための多層パーセプトロン。
#     """

#     def __init__(self, input_size, hidden_sizes=None, dropout_rates=None):
#         super().__init__()
#         self.input_size = input_size

#         if hidden_sizes is None:
#             hidden_sizes = [1024, 128, 64, 16, 1]

#         if dropout_rates is None:
#             dropout_rates = [0.2, 0.2, 0.1, 0.0, 0.0]

#         # レイヤーの構築
#         layers = []
#         prev_size = input_size

#         for i, (size, dropout) in enumerate(zip(hidden_sizes, dropout_rates, strict=False)):
#             layers.append(nn.Linear(prev_size, size))
#             if i < len(hidden_sizes) - 1 and dropout > 0:
#                 layers.append(nn.Dropout(dropout))
#             prev_size = size

#         self.layers = nn.Sequential(*layers)

#         # パラメータの初期化
#         for name, param in self.layers.named_parameters():
#             if "weight" in name:
#                 nn.init.normal_(param, mean=0.0, std=1.0 / (self.input_size + 1))
#             if "bias" in name:
#                 nn.init.constant_(param, val=0)

#     def forward(self, input):
#         """順伝播の実行"""
#         return self.layers(input)


# class ImageRewardModel(nn.Module):
#     """
#     ImageRewardModel は BLIP と MLP を組み合わせたモデルです。
#     画像の美的評価を行います。
#     """

#     def __init__(
#         self,
#         device: str,
#         config: Optional[Dict[str, Any]] = None,
#     ):
#         super().__init__()
#         self.device = device

#         # 設定がない場合はデフォルト値を使用
#         if config is None:
#             config = {
#                 "mlp_input_size": 768,
#                 "mlp_hidden_sizes": [1024, 128, 64, 16, 1],
#                 "mlp_dropout_rates": [0.2, 0.2, 0.1, 0.0, 0.0],
#                 "reward_mean": 0.16717362830052426,
#                 "reward_std": 1.0333394966054072,
#             }

#         # BLIPモデルとプロセッサを初期化
#         self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
#         self.blip = BlipModel.from_pretrained("Salesforce/blip-image-captioning-large")

#         # MLPの設定
#         mlp_input_size = config.get("mlp_input_size", 768)
#         mlp_hidden_sizes = config.get("mlp_hidden_sizes", [1024, 128, 64, 16, 1])
#         mlp_dropout_rates = config.get("mlp_dropout_rates", [0.2, 0.2, 0.1, 0.0, 0.0])

#         self.mlp = MLP(mlp_input_size, mlp_hidden_sizes, mlp_dropout_rates)

#         # ImageRewardモデルの正規化パラメータ
#         self.mean = config.get("reward_mean", 0.16717362830052426)
#         self.std = config.get("reward_std", 1.0333394966054072)

#         # デバイスに移動
#         self.to(device)

#     def forward(self, **inputs):
#         """
#         画像とプロンプトからスコアを計算します。

#         Args:
#             inputs: BLIPモデルへの入力（pixel_values, input_ids, attention_maskなど）

#         Returns:
#             スコアテンソル
#         """
#         # BLIPモデルで特徴量を抽出
#         outputs = self.blip(**inputs, return_dict=True)

#         # テキスト特徴量を取得
#         # BlipModelの出力からtext_embedsを取得
#         text_features = outputs.text_embeds

#         # スコアの計算
#         score = self.mlp(text_features)
#         score = (score - self.mean) / self.std

#         return score

#     def to(self, device):
#         """
#         モデルを指定されたデバイスに移動します。

#         Args:
#             device: 移動先のデバイス

#         Returns:
#             モデル自身
#         """
#         self.device = device
#         self.blip.to(device)
#         self.mlp.to(device)
#         return super().to(device)


# class ImageRewardScorer(BaseScorer):
#     """
#     ImageRewardScorer は ImageReward モデルのラッパークラスです。
#     画像の美的評価スコアを計算します。
#     """

#     def __init__(self, model_name: str, device: str = "cpu", **kwargs) -> None:
#         """
#         ImageRewardScorer の初期化

#         Args:
#             model_name: モデル名
#             device: 使用するデバイス（"cpu" または "cuda"）
#             **kwargs: その他の引数
#         """
#         super().__init__(model_name=model_name)
#         self.device = device
#         self.model: Dict[str, Any] = {}
#         self.is_model_loaded = False

#         # 設定を読み込む
#         all_configs = load_model_config()
#         self.config = all_configs.get(model_name, {})
#         if not self.config:
#             raise ValueError(f"Configuration not found for model: {model_name}")

#         # 設定からMLPとスコア正規化パラメータを取得
#         self.mlp_config = {
#             "input_size": self.config.get("mlp_input_size", 768),
#             "hidden_sizes": self.config.get("mlp_hidden_sizes", [1024, 128, 64, 16, 1]),
#             "dropout_rates": self.config.get("mlp_dropout_rates", [0.2, 0.2, 0.1, 0.0, 0.0]),
#         }

#         self.mean = self.config.get("reward_mean", 0.16717362830052426)
#         self.std = self.config.get("reward_std", 1.0333394966054072)
#         self.score_prefix = self.config.get("score_prefix", "[IR]")

#     def _load_model(self) -> None:
#         """モデルをロードします。"""
#         if self.is_model_loaded:
#             return

#         # 設定ファイルからBLIPの設定を読み込む
#         blip_config_path = self.config.get("config_path")
#         blip_config = {}

#         if blip_config_path:
#             try:
#                 with open(blip_config_path, "r") as f:
#                     blip_config = json.load(f)
#             except Exception as e:
#                 logger.warning(f"Failed to load BLIP config from {blip_config_path}: {e}")

#         # モデルを作成
#         model_dict = create_blip_image_reward_model(
#             {
#                 "device": self.device,
#                 "model_path": self.config.get("model_path"),
#                 "base_model": self.config.get("base_model"),
#                 "config_path": blip_config_path,
#                 "mlp_input_size": self.mlp_config["input_size"],
#                 "mlp_hidden_sizes": self.mlp_config["hidden_sizes"],
#                 "mlp_dropout_rates": self.mlp_config["dropout_rates"],
#                 "reward_mean": self.mean,
#                 "reward_std": self.std,
#                 **self.config,
#             }
#         )

#         self.model = model_dict
#         self.is_model_loaded = True

#     def _calculate_score(self, raw_output: float) -> float:
#         """
#         生のモデル出力を正規化されたスコアに変換します。

#         Args:
#             raw_output: モデルからの生の出力値

#         Returns:
#             正規化されたスコア（0-10の範囲）
#         """
#         # スコアを0-10の範囲に変換
#         score = (raw_output * self.std + self.mean) * 10
#         return max(0, min(score, 10))  # 0-10の範囲にクリップ

#     def _get_score_tag(self, score: float) -> str:
#         """
#         スコアをタグ形式の文字列に変換します。

#         Args:
#             score: 計算されたスコア

#         Returns:
#             タグ形式のスコア文字列（例: "[IR]score_7"）
#         """
#         score_int = max(0, int(score))  # 負の値を0にクリップ
#         return f"{self.score_prefix}score_{score_int}"

#     def predict(self, images: List[Image.Image], prompt: str = "") -> List[Dict[str, Any]]:
#         """
#         画像の美的スコアを計算します。

#         Args:
#             images: 評価する画像のリスト
#             prompt: オプションのプロンプト文字列

#         Returns:
#             各画像のスコア情報を含む辞書のリスト
#         """
#         if not self.is_model_loaded:
#             self._load_model()

#         model = self.model["model"]
#         processor = self.model["processor"]

#         scores = []
#         for image in images:
#             # 画像とテキストを前処理
#             inputs = processor(images=image, text=prompt, return_tensors="pt")
#             inputs = {k: v.to(self.device) for k, v in inputs.items()}

#             with torch.no_grad():
#                 # モデルで予測
#                 outputs = model(**inputs)

#                 # スコアを取得
#                 raw_score = outputs.item()
#                 calculated_score = self._calculate_score(raw_score)
#                 score_tag = self._get_score_tag(calculated_score)
#                 scores.append(self._generate_result(raw_score, score_tag))

#         return scores

#     def predict_pipe(
#         self, images: List[Image.Image], prompt: str = "", batch_size: int = 8
#     ) -> List[List[Dict[str, Any]]]:
#         """
#         複数画像のバッチ予測を実施します。

#         Args:
#             images: 評価する画像のリスト
#             prompt: オプションのプロンプト文字列
#             batch_size: バッチサイズ

#         Returns:
#             各画像のスコア情報を含む辞書のリストのリスト
#         """
#         scores = []
#         for image in images:
#             scores.append(self.predict([image], prompt=prompt))
#         return scores

#     def name(self) -> str:
#         """モデルの名前を返します。"""
#         return "ImageReward Scorer"


# def create_blip_image_reward_model(config: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     BLIP Image Reward モデルを作成します。

#     Args:
#         config: モデル設定

#     Returns:
#         モデル、プロセッサなどを含む辞書
#     """
#     device = config["device"]

#     # BLIPプロセッサを初期化
#     processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")

#     # ImageRewardモデルを初期化
#     model = ImageRewardModel(
#         device=device,
#         config={
#             "mlp_input_size": config.get("mlp_input_size", 768),
#             "mlp_hidden_sizes": config.get("mlp_hidden_sizes", [1024, 128, 64, 16, 1]),
#             "mlp_dropout_rates": config.get("mlp_dropout_rates", [0.2, 0.2, 0.1, 0.0, 0.0]),
#             "reward_mean": config.get("reward_mean", 0.16717362830052426),
#             "reward_std": config.get("reward_std", 1.0333394966054072),
#         },
#     )

#     # モデル重みをダウンロード
#     image_reward_model_path = config["model_path"]
#     checkpoint_file = load_file(image_reward_model_path)
#     checkpoint = torch.load(checkpoint_file, map_location="cpu")

#     # BLIPモデルの重みはすでにロードされているので、MLPの重みだけをロード
#     mlp_state_dict = {k: v for k, v in checkpoint.items() if "reward_predictor" in k}

#     # MLPのレイヤー名を変換
#     mlp_state_dict_rename = {}
#     for k, v in mlp_state_dict.items():
#         new_key = k.replace("reward_predictor.", "")
#         mlp_state_dict_rename[new_key] = v

#     # MLPの重みをロード
#     model.mlp.load_state_dict(mlp_state_dict_rename, strict=False)

#     # モデルを評価モードに設定
#     model.eval()

#     return {"model": model, "processor": processor}


# # model_factory.py から呼び出される関数
# def create_blip_sfr_vision_language_research_model(config: Dict[str, Any]) -> Dict[str, Any]:
#     """
#     BLIP SFR Vision Language Research モデルを作成します。

#     Args:
#         config: モデル設定

#     Returns:
#         モデル、プロセッサなどを含む辞書
#     """
#     return create_blip_image_reward_model(config)
