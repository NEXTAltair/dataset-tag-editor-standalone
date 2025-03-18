import logging
from typing import Any, Optional, Type

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, pipeline

from . import utils

logger = logging.getLogger(__name__)


class Classifier(nn.Module):
    """画像特徴量を入力として、分類スコアを出力する柔軟な分類器。

    Args:
        input_size (int): 入力特徴量の次元数
        hidden_sizes (list[int], optional): 各隠れ層のユニット数のリスト
        output_size (int, optional): 出力層のユニット数 (通常は1)
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

        for _, (size, drop) in enumerate(zip(hidden_sizes, dropout_rates, strict=False)):
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


# def create_blip_mlp_model(config: dict[str, Any]) -> dict[str, Any]:
#     """
#     BLIP モデルを作成します。
#
#     Args:
#     config (dict[str, Any]): モデルの設定。
#
#     Returns:
#     dict[str, Any]: モデル、プロセッサ、BLIP モデルなどを含む辞書
#
#     Note:
#     この関数は現在実装されていません。
#     """
# TODO: 将来的に実装予定
# return {}  # 暫定的に空の辞書を返す


# def create_blip_sfr_vision_language_research_model(config: dict[str, Any]) -> dict[str, Any]:
#     """
#     BLIP SFR Vision Language Research モデルを作成します。
#
#     Args:
#     config (dict[str, Any]): モデルの設定。
#
#     Returns:
#     dict[str, Any]: モデル、プロセッサ、BLIP モデルなどを含む辞書
#     """
#     # モジュールが存在しない場合のエラー処理
#     try:
#         from ..score_models.imagereward import (
#             create_blip_sfr_vision_language_research_model as create_model_func,
#         )
#
#         return create_model_func(config)
#     except (ImportError, AttributeError):
#         logger.warning("ImageReward モデルが見つかりません。空の辞書を返します。")
#         return {}


def create_clip_model(
    base_model: str,
    model_path: str,
    device: str,
    activation_type: Optional[str] = None,
    final_activation_type: Optional[str] = None,
) -> dict[str, Any]:
    """どの CLIP モデルでも使用可能なモデルを作成します。

    Args:
        base_model (str): CLIP モデルの名前またはパス
        model_path (str): モデルの重みファイルのパス
        device (str): モデルを実行するデバイス ("cuda" または "cpu")
        activation_type (str): 活性化関数のタイプ ("ReLU", "GELU", "Sigmoid", "Tanh")
        final_activation_type (str): 最終層の活性化関数のタイプ ("ReLU", "GELU", "Sigmoid", "Tanh")

    Returns:
        dict[str, Any]: {
            "model": Classifier モデルインスタンス,
            "processor": CLIP プロセッサインスタンス,
            "clip_model": CLIP モデルインスタンス
        }
    """
    # 共通の CLIP モデルとプロセッサを初期化
    clip_processor = CLIPProcessor.from_pretrained(base_model)
    clip_model = CLIPModel.from_pretrained(base_model).to(device).eval()

    # 入力サイズを自動検出
    input_size = clip_model.config.projection_dim
    logger.debug(f"CLIP モデル {base_model} の特徴量次元: {input_size}")

    # モデルの重みをロード
    file = utils.load_file(model_path)
    state_dict = torch.load(file)

    # state_dict の構造から正しい hidden_features を推測する
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

    # 最後の出力層を除外 (必要な場合)
    if hidden_features and len(hidden_features) > 1:
        hidden_features = hidden_features[:-1]

    if not hidden_features:
        # 構造を推測できなかった場合はモデルタイプによってデフォルト値を設定
        logger.warning(f"CLIP モデル {base_model} の構造を推測できませんでした。デフォルト値を設定します。")
        if "large" in base_model:
            hidden_features = [1024, 128, 64, 16]
        else:
            hidden_features = [512, 128, 64, 16]  # 小さいモデル用に調整

    logger.info(f"推測された hidden_features: {hidden_features}")

    # 活性化関数の設定マップ
    activation_map = {
        "ReLU": nn.ReLU,
        "GELU": nn.GELU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        # NOTE: 必要になれば追加､ でも今更 Pipeline 非対応をさらに追加したくはない
    }

    # 設定から活性化関数のパラメータを取得
    use_activation = activation_type is not None
    if use_activation and activation_type in activation_map:
        activation_func = activation_map[activation_type]
    else:
        activation_func = nn.ReLU

    use_final_activation = final_activation_type is not None
    if use_final_activation and final_activation_type in activation_map:
        final_activation_func = activation_map[final_activation_type]
    else:
        final_activation_func = nn.Sigmoid

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
    model = model.to(device)
    logger.debug("デバイス転送完了")

    return {"model": model, "processor": clip_processor, "clip_model": clip_model}


class ModelLoad:
    _MODEL_STATES: dict[str, str] = {}
    logger = logging.getLogger(__name__)


    @staticmethod
    def pipeline_model_load(
        model_name: str, model_path: str, batch_size: int, device: str
    ) -> Optional[dict[str, Any]]:
        if model_name in ModelLoad._MODEL_STATES:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に読み込まれています。")
            return None
        pipeline_obj = pipeline(
            model=model_path,
            device=device,
            batch_size=batch_size,
            use_fast=True,
        )
        ModelLoad._MODEL_STATES[model_name] = f"on_{device}"
        return {"pipeline": pipeline_obj}

    @staticmethod
    def clip_model_load(
        model_name: str,
        base_model: str,
        model_path: str,
        device: str,
        activation_type: Optional[str] = None,
        final_activation_type: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        if model_name in ModelLoad._MODEL_STATES:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に読み込まれています。")
            return None
        model_dict = create_clip_model(
            base_model=base_model,
            model_path=model_path,
            device=device,
            activation_type=activation_type,
            final_activation_type=final_activation_type,
        )

        ModelLoad._MODEL_STATES[model_name] = f"on_{device}"
        return model_dict

    @staticmethod
    def cache_to_main_memory(model_name: str, model: dict[str, Any]) -> dict[str, Any]:
        """モデルを CPU メモリにキャッシュします。

        モデルのすべてのコンポーネントを GPU から CPU メモリに移動します。
        これにより、GPU 上のメモリは解放されますが、モデル自体は保持されるため、
        後で `restore_model_to_cuda` を呼び出して再利用できます。

        主な用途:
        - モデル自体は保持したまま GPU リソースを解放したい場合

        Note:
            このメソッドはモデルを破棄しません。モデルを完全に解放するには
            `release_model` を使用してください。
        """
        if ModelLoad._MODEL_STATES[model_name] == "on_cpu":
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に CPU にあります。")
            return model

        for component_name, component in model.items():
            if component_name == "pipeline":
                # パイプラインの場合は内部モデルを移動
                if hasattr(component, "model"):
                    component.model.to("cpu")
                ModelLoad.logger.debug(f"パイプライン '{component_name}' を CPU に移動しました")
            elif hasattr(component, "to"):  # to メソッドを持つ場合のみ CPU に移動
                component.to("cpu")
                ModelLoad.logger.debug(f"コンポーネント '{component_name}' を CPU に移動しました")

        ModelLoad._MODEL_STATES[model_name] = "on_cpu"
        return model

    @staticmethod
    def restore_model_to_cuda(model_name: str, device: str, model: dict[str, Any]) -> dict[str, Any]:
        """モデルを指定 CUDA に復元します。"""
        if ModelLoad._MODEL_STATES[model_name] == "on_cpu" and "cuda" in device:
            for component_name, component in model.items():
                if component_name == "pipeline":
                    if hasattr(component, "model"):
                        component.model.to("cuda")

                elif hasattr(component, "to"):
                    component.to("cuda")

                ModelLoad._MODEL_STATES[model_name] = "on_cuda"
                ModelLoad.logger.info(f"モデル '{model_name}' をメインメモリから復元しました。")
            return model

        return model

    @staticmethod
    def release_model(model_name: str) -> None:
        """
        モデルを _LOADED_SCORERS から削除 メモリから解放し、GPU キャッシュをクリアします。
        """
        if model_name in ModelLoad._MODEL_STATES:
            del ModelLoad._MODEL_STATES[model_name]

        # GPU メモリをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ModelLoad.logger.info(f"モデル '{model_name}' を解放しました。")
