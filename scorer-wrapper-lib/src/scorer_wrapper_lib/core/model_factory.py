import logging
from typing import Any, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, pipeline

from . import utils

logger = logging.getLogger(__name__)


class Classifier(nn.Module):
    """画像特徴量を入力として、分類スコアを出力する柔軟な分類器。

    Args:
        input_size (int): 入力特徴量の次元数
        hidden_sizes (list[int], optional): 各隠れ層のユニット数のリスト
        output_size (int, optional): 出力層のユニット数（通常は1）
        dropout_rates (list[float], optional): 各隠れ層のドロップアウト率
        use_activation (bool, optional): 活性化関数を使用するかどうか
        activation (Type[nn.Module], optional): 使用する活性化関数
        use_final_activation (bool, optional): 最終層に活性化関数を使用するかどうか
        final_activation (Type[nn.Module], optional): 最終層に使用する活性化関数
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Optional[list[int]] = None,
        output_size: int = 1,
        dropout_rates: Optional[list[float]] = None,
        use_activation: bool = False,
        activation: Type[nn.Module] = nn.ReLU,
        use_final_activation: bool = False,
        final_activation: Type[nn.Module] = nn.Sigmoid,
    ) -> None:
        super().__init__()

        # デフォルト値の設定
        if hidden_sizes is None:
            hidden_sizes = [1024, 128, 64, 16]

        if dropout_rates is None:
            dropout_rates = [0.2, 0.2, 0.1, 0.0]

        # ドロップアウト率のリストの長さを調整
        if len(dropout_rates) < len(hidden_sizes):
            dropout_rates = dropout_rates + [0.0] * (len(hidden_sizes) - len(dropout_rates))

        # レイヤーの構築
        layers: list[nn.Module] = []
        prev_size = input_size

        for i, (size, drop) in enumerate(zip(hidden_sizes, dropout_rates, strict=False)):
            layers.append(nn.Linear(prev_size, size))

            if use_activation:
                layers.append(activation())

            if drop > 0:
                layers.append(nn.Dropout(drop))

            prev_size = size

        # 出力層
        layers.append(nn.Linear(prev_size, output_size))

        # 最終活性化関数
        if use_final_activation:
            layers.append(final_activation())

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ネットワークの順伝播を実行します。

        Args:
            x (torch.Tensor): 入力テンソル

        Returns:
            torch.Tensor: 処理された出力テンソル
        """
        return self.layers(x)  # type: ignore


# image_embeddings 関数 (WaifuAestheticで使用されているものを流用)
def image_embeddings(
    image: Image.Image, model: CLIPModel, processor: CLIPProcessor
) -> np.ndarray[Any, np.dtype[Any]]:
    """画像からCLIPモデルを使用して埋め込みベクトルを生成します。

    Args:
        image (Image.Image): 埋め込みを生成するための入力画像
        model (CLIPModel): 特徴抽出に使用するCLIPモデル
        processor (CLIPProcessor): 画像の前処理を行うCLIPプロセッサ

    Returns:
        np.ndarray: 正規化された画像の埋め込みベクトル
    """
    inputs = processor(images=image, return_tensors="pt")["pixel_values"]
    inputs = inputs.to(model.device)
    result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
    # 正規化された埋め込みを返す
    normalized_result: np.ndarray[Any, np.dtype[Any]] = result / np.linalg.norm(result)
    return normalized_result


# def create_blip_mlp_model(config: dict[str, Any]) -> dict[str, Any]:
#     """
#     BLIP モデルを作成します。

#     Args:
#         config (dict[str, Any]): モデルの設定。

#     Returns:
#         dict[str, Any]: モデル、プロセッサ、BLIPモデルなどを含む辞書

#     Note:
#         この関数は現在実装されていません。
#     """
# TODO: 将来的に実装予定
#   return {}  # 暫定的に空の辞書を返す


# def create_blip_sfr_vision_language_research_model(config: dict[str, Any]) -> dict[str, Any]:
#     """
#     BLIP SFR Vision Language Research モデルを作成します。

#     Args:
#         config (dict[str, Any]): モデルの設定。

#     Returns:
#         dict[str, Any]: モデル、プロセッサ、BLIPモデルなどを含む辞書
#     """
#     # モジュールが存在しない場合のエラー処理
#     try:
#         from ..score_models.imagereward import (
#             create_blip_sfr_vision_language_research_model as create_model_func,
#         )

#         return create_model_func(config)
#     except (ImportError, AttributeError):
#         logger.warning("ImageRewardモデルが見つかりません。空の辞書を返します。")
#         return {}


def create_clip_model(config: dict[str, Any]) -> dict[str, Any]:
    """どのCLIPモデルでも使用可能なモデルを作成します。

    Args:
        config (dict[str, Any]): モデルの設定。必要なキー:
                             - "base_model": CLIPモデルの名前またはパス
                             - "model_path": モデルの重みファイルのパス
                             - "device": モデルを実行するデバイス ("cuda" または "cpu")

    Returns:
        dict[str, Any]: {
            "model": Classifier モデルインスタンス,
            "processor": CLIPプロセッサインスタンス,
            "clip_model": CLIPモデルインスタンス
        }
    """
    # 共通のCLIPモデルとプロセッサを初期化
    base_model = config["base_model"]
    clip_processor = CLIPProcessor.from_pretrained(base_model)
    clip_model = CLIPModel.from_pretrained(base_model).to(config["device"]).eval()

    # 入力サイズを自動検出
    input_size = clip_model.config.projection_dim
    logger.debug(f"CLIPモデル {base_model} の特徴量次元: {input_size}")

    # モデルの重みをロード
    file = utils.load_file(config["model_path"])
    state_dict = torch.load(file)

    # state_dictの構造から正しいhidden_featuresを推測する
    hidden_features = []
    layer_idx = 0

    # レイヤーの重みキーを探索して構造を推測
    while True:
        weight_key = f"layers.{layer_idx}.weight"
        if weight_key not in state_dict:
            break
        weight = state_dict[weight_key]
        hidden_features.append(weight.shape[0])
        # 活性化関数やドロップアウトがあるかに応じてスキップ量を調整
        # 基本的には線形層だけを考慮
        next_idx = layer_idx + 1
        while f"layers.{next_idx}.weight" not in state_dict and next_idx < layer_idx + 5:
            next_idx += 1
        layer_idx = next_idx

    # 最後の出力層を除外（必要な場合）
    if hidden_features and len(hidden_features) > 1:
        hidden_features = hidden_features[:-1]

    if not hidden_features:
        # 構造を推測できなかった場合はモデルタイプによってデフォルト値を設定
        if "large" in base_model:
            hidden_features = [1024, 128, 64, 16]
        else:
            hidden_features = [512, 128, 64, 16]  # 小さいモデル用に調整

    logger.info(f"推測されたhidden_features: {hidden_features}")

    # 活性化関数の設定マップ
    activation_map = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        # NOTE: 必要になれば追加､ でも今更Pipeline非対応をさらに追加したくはない
    }

    # 設定から活性化関数のパラメータを取得
    activation_type = config.get("activation_type", None)
    use_activation = activation_type is not None
    activation_func = activation_map.get(activation_type, nn.ReLU) if use_activation else nn.ReLU

    final_activation_type = config.get("final_activation_type", None)
    use_final_activation = final_activation_type is not None
    final_activation_func = (
        activation_map.get(final_activation_type, nn.Sigmoid) if use_final_activation else nn.Sigmoid
    )

    # モデル初期化
    logger.info("モデル初期化開始...")
    model = Classifier(
        input_size=input_size,
        hidden_sizes=hidden_features,
        output_size=1,
        dropout_rates=[0.2, 0.2, 0.1, 0.0],
        use_activation=use_activation,
        activation=activation_func,
        use_final_activation=use_final_activation,
        final_activation=final_activation_func,
    )
    logger.debug("モデル初期化完了、重みロード開始...")
    model.load_state_dict(state_dict, strict=False)
    logger.debug("重みロード完了、デバイス転送開始...")
    model = model.to(config["device"])
    logger.debug("デバイス転送完了")

    return {"model": model, "processor": clip_processor, "clip_model": clip_model}


def create_model(config: dict[str, Any]) -> dict[str, Any]:
    """
    指定された設定に基づいてモデルを作成します。

    Args:
        config (dict[str, Any]): モデルの設定情報を含む辞書。
                             "type"キーはモデルの種類を指定します。
                             ("pipeline", "clip", "blip_mlp"のいずれか)

    Returns:
        dict[str, Any]: モデル、プロセッサ、その他必要なコンポーネントを含む辞書

    Raises:
        ValueError: 不明なモデルタイプが指定された場合
    """
    model_type = config["type"]
    model_path = config["model_path"]
    device = config["device"]
    BATCH_SIZE = 8  # NOTE: 暫定的な設定ユーザーに設定求めるほどのものではない

    if model_type == "pipeline":
        pipeline_obj = pipeline(
            model=model_path,
            device=device,
            batch_size=BATCH_SIZE,
            use_fast=True,
        )
        return {"pipeline": pipeline_obj}

    elif model_type == "clip":
        return create_clip_model(config)

    # elif model_type == "blip_mlp":
    #     # NOTE: 実装が特殊なので、モデルのクラスを指定する
    #     if config["class"] == "ImageRewardScorer":
    #         return create_blip_sfr_vision_language_research_model(config)
    #     return create_blip_mlp_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
