import logging
from typing import Any, Optional, Type

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, pipeline

from . import utils

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Args:
        in_features (int): 入力次元
        hidden_features (list[int], optional): 各中間層のユニット数のリスト。指定がない場合はデフォルトの構成を使用
        out_features (int, optional): 出力次元 (デフォルトは1)
        act_layer (nn.Module, optional): 活性化関数（デフォルトは nn.GELU）
        drop (float, optional): 各層のドロップアウト率 (デフォルトは0.1)
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[list[int]] = None,
        out_features: int = 1,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_features is None:
            # 参考実装と現在の実装の妥協構成
            hidden_features = [1024, 128, 64, 16]
        layers: list[nn.Module] = []  # nn.Moduleのリストとして型アノテーション
        prev_features = in_features
        for hf in hidden_features:
            layers.append(nn.Linear(prev_features, hf))
            layers.append(act_layer())
            layers.append(nn.Dropout(drop))
            prev_features = hf
        layers.append(nn.Linear(prev_features, out_features))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播を実行します。

        Args:
            x (torch.Tensor): 入力テンソル

        Returns:
            torch.Tensor: 処理された出力テンソル
        """
        return self.layers(x)


class Classifier(torch.nn.Module):
    """画像特徴量を入力として、分類スコアを出力するシンプルな分類器。

    Args:
        input_size (int): 入力特徴量の次元数
        hidden_size (int): 隠れ層のユニット数
        output_size (int): 出力層のユニット数（通常は1）
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(Classifier, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = torch.nn.Linear(hidden_size // 2, output_size)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ネットワークの順伝播を実行します。

        Args:
            x (torch.Tensor): 入力テンソル

        Returns:
            torch.Tensor: シグモイド活性化された出力テンソル
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x


# image_embeddings 関数 (WaifuAestheticで使用されているものを流用)
def image_embeddings(image: Image.Image, model: CLIPModel, processor: CLIPProcessor) -> np.ndarray:
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
    return result / np.linalg.norm(result)


def create_clip_mlp_model(config: dict[str, Any]) -> dict[str, Any]:
    """
    CLIP+MLP モデルを作成します。

    Args:
        config (dict[str, Any]): モデルの設定。必要なキー:
                             - "base_model": CLIPモデルの名前またはパス
                             - "model_path": MLPモデルの重みファイルのパス
                             - "device": モデルを実行するデバイス ("cuda" または "cpu")

    Returns:
        dict[str, Any]: {
            "model": MLP モデルインスタンス,
            "processor": CLIPプロセッサインスタンス,
            "clip_model": CLIPモデルインスタンス
        }
    """
    # 共通のCLIPモデルとプロセッサを初期化
    base_model = config["base_model"]
    clip_processor = CLIPProcessor.from_pretrained(base_model)
    clip_model = CLIPModel.from_pretrained(base_model).to(config["device"]).eval()
    clip_config = clip_model.config

    # モデルの重みをロード
    file = utils.load_file(config["model_path"])
    model = MLP(clip_config.projection_dim)

    # ImprovedAestheticモデルまたはその他のモデルで構造の不一致がある場合は緩和条件でロード
    try:
        model.load_state_dict(torch.load(file), strict=False)  # strict=Falseで緩和条件ロード
    except RuntimeError as e:
        logger.warning(f"モデルロード中にエラーが発生しました: {e}")

    model = model.to(config["device"])

    return {"model": model, "processor": clip_processor, "clip_model": clip_model}


def create_clip_classifier_model(config: dict[str, Any]) -> dict[str, Any]:
    """
    CLIP+Classifier モデルを作成します。

    Args:
        config (dict[str, Any]): モデルの設定。必要なキー:
                             - "base_model": CLIPモデルの名前またはパス
                             - "model_path": Classifierモデルの重みファイルのパス
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
    clip_config = clip_model.config

    # モデルの重みをロード
    file = utils.load_file(config["model_path"])
    model = Classifier(clip_config.projection_dim, 256, 1)  # Classifierを使用
    model.load_state_dict(torch.load(file))
    model = model.to(config["device"])

    return {"model": model, "processor": clip_processor, "clip_model": clip_model}


def create_blip_mlp_model(config: dict[str, Any]) -> dict[str, Any]:
    """
    BLIP+MLP モデルを作成します。

    Args:
        config (dict[str, Any]): モデルの設定。

    Returns:
        dict[str, Any]: モデル、プロセッサ、BLIPモデルなどを含む辞書

    Note:
        この関数は現在実装されていません。
    """
    # TODO: 将来的に実装予定
    return {}  # 暫定的に空の辞書を返す


def create_blip_sfr_vision_language_research_model(config: dict[str, Any]) -> dict[str, Any]:
    """
    BLIP SFR Vision Language Research モデルを作成します。

    Args:
        config (dict[str, Any]): モデルの設定。

    Returns:
        dict[str, Any]: モデル、プロセッサ、BLIPモデルなどを含む辞書
    """
    # モジュールが存在しない場合のエラー処理
    try:
        from ..score_models.imagereward import (
            create_blip_sfr_vision_language_research_model as create_model_func,
        )

        return create_model_func(config)
    except (ImportError, AttributeError):
        logger.warning("ImageRewardモデルが見つかりません。空の辞書を返します。")
        return {}


def create_model(config: dict[str, Any]) -> dict[str, Any]:
    """
    指定された設定に基づいてモデルを作成します。

    Args:
        config (dict[str, Any]): モデルの設定情報を含む辞書。
                             "type"キーはモデルの種類を指定します。
                             ("pipeline", "clip_mlp", "clip_classifier", "blip_mlp"のいずれか)

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
        )
        return {"pipeline": pipeline_obj}

    elif model_type == "clip_mlp":
        return create_clip_mlp_model(config)
    elif model_type == "clip_classifier":
        return create_clip_classifier_model(config)
    elif model_type == "blip_mlp":
        # 実装が特殊なので
        if config["class"] == "ImageRewardScorer":
            return create_blip_sfr_vision_language_research_model(config)
        return create_blip_mlp_model(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
