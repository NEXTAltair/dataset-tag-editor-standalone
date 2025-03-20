import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, TypedDict

import huggingface_hub
import numpy as np
import onnxruntime as ort
import polars as pl
import torch
from PIL import Image

logger = logging.getLogger(__name__)

config = {
    "model_path": "SmilingWolf/wd-swinv2-tagger-v3",
    "device": "cuda",
}


# 前処理された画像データを表す型定義
class PreprocessedImage(TypedDict):
    # TransformerModel
    pixel_values: Optional[torch.Tensor]

    # ONNXModelTagger
    input_data: Optional[np.ndarray]


# model_factory によって生成されるモデルのコンポーネントを表す型定義
class ModelComponents(TypedDict):
    # ONNXModelTagger
    session: Optional[ort.InferenceSession]  # ONNXセッション
    csv_path: Optional[str]  # ローカルに保存されたCSVファイルのパス


# 画像の推論結果を人が読める形式に変換したものを表す型定義
class predictions(TypedDict):
    # BLIP BLIP2 GIT
    caption: Optional[list[str]]

    # ONNX
    output_data: Optional[dict[str, dict[str, float]]]


def download_wd_tagger_model(model_repo: str) -> tuple[str, str]:
    MODEL_FILENAME = "model.onnx"
    LABEL_FILENAME = "selected_tags.csv"
    csv_path = huggingface_hub.hf_hub_download(
        model_repo,
        LABEL_FILENAME,
    )
    model_path = huggingface_hub.hf_hub_download(
        model_repo,
        MODEL_FILENAME,
    )
    return csv_path, model_path


class ModelLoad:
    _MODEL_STATES: dict[str, str] = {}
    logger = logging.getLogger(__name__)

    @staticmethod
    def load_onnx_components(model_name: str, model_repo: str, device: str) -> Optional[dict[str, Any]]:
        if model_name in ModelLoad._MODEL_STATES:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に読み込まれています。")
            return None
        # ONNXランタイムセッションの作成
        csv_path, model_path = download_wd_tagger_model(model_repo)

        # 利用可能なプロバイダーを取得
        available_providers = ort.get_available_providers()
        ModelLoad.logger.debug(f"利用可能なプロバイダー: {available_providers}")

        # デバイスに基づいてプロバイダーを選択
        if device == "cuda" and "CUDAExecutionProvider" in available_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        ModelLoad.logger.info(f"ONNXモデル '{model_path}' をロードしています...")
        session = ort.InferenceSession(model_path, providers=providers)

        ModelLoad._MODEL_STATES[model_name] = f"on_{device}"
        return {"model": session, "csv_path": csv_path, "model_path": model_path}

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


class BaseTagger(ABC):
    def __init__(self, model_name: str):
        """BaseTagger を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = config

        self.model_path = self.config["model_path"]
        self.device = self.config.get("device", "cuda")

        self.components: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def __enter__(self) -> "BaseTagger":
        pass

    @abstractmethod
    def __exit__(self, exception_type: type[Exception], exception_value: Exception, traceback: Any) -> None:
        pass

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像からタグを予測します。"""
        results = []
        for image in images:
            try:
                processed_image = self._preprocess_image(image)
                # モデル推論
                raw_output = self._run_inference(processed_image)
                # 推論結果をフォーマット
                formatted_output = self._format_predictions(raw_output)
                # タグを生成
                tags = self._generate_tags(formatted_output)
                # 結果を標準形式で追加
                results.append(self._generate_result(formatted_output, tags))
            except ValueError as e:
                self.logger.error(f"推論生成中のエラー: {e}")
                raise
        return results

    @abstractmethod
    def _preprocess_image(self, image: Image.Image) -> Any:
        """画像を前処理してモデル入力形式に変換します。"""
        pass

    @abstractmethod
    def _run_inference(self, processed_image: Any) -> Any:
        """モデル推論を実行します。"""
        pass

    @abstractmethod
    def _format_predictions(self, raw_output: Any) -> Any:
        """モデルの生出力をフォーマットします。"""
        pass

    @abstractmethod
    def _generate_tags(self, raw_output: Any) -> list[str]:
        """モデルの生出力からタグを生成します。"""
        pass

    def _generate_result(self, model_output: Any, annotation_list: list[str]) -> dict[str, Any]:
        """標準化された結果の辞書を生成します。

        Args:
            model_name (str): モデルの名前。
            model_output: モデルの出力。
            annotation_list (list[str]): 各サブクラスで

        Returns:
            dict: モデル出力、モデル名、タグリストを含む辞書。
        """
        return {
            "model_name": self.model_name,
            "model_output": model_output,
            "annotation": annotation_list,
        }


class ONNXModelTagger(BaseTagger):
    """ONNXランタイムを使用するモデル用の抽象クラス。
    本質的にはWD-Tagger用のクラス
    """

    def __init__(self, model_name: str):
        """ONNXModelTagger を初期化します。
        Args:
            model_name (str): モデルの名前。
        """
        super().__init__(model_name=model_name)
        # 設定ファイルから追加パラメータを取得
        self.labels: list[str] = []  # __enter__でロード

        # タグカテゴリ用のインデックス
        self.rating_indexes: list[int] = []
        self.general_indexes: list[int] = []
        self.character_indexes: list[int] = []

    def __enter__(self) -> "ONNXModelTagger":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        loaded_model = ModelLoad.load_onnx_components(
            self.model_name,
            self.model_path,
            self.device,
        )
        if loaded_model is not None:
            self.components = loaded_model
        self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)

        # ラベルとカテゴリインデックスをロード
        self._load_labels()

        return self

    def __exit__(self, exception_type: type[Exception], exception_value: Exception, traceback: Any) -> None:
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

    def _load_labels(self):
        """ラベル情報をロードし、カテゴリごとのインデックスを設定します。"""
        # ラベルファイルをpolarsで読み込み
        tags_df = pl.read_csv(self.components["csv_path"])

        # ラベル名を取得
        self.labels = tags_df["name"].to_list()

        # カテゴリインデックスを設定
        self.rating_indexes = [i for i, cat in enumerate(tags_df["category"].to_list()) if cat == 9]
        self.general_indexes = [i for i, cat in enumerate(tags_df["category"].to_list()) if cat == 0]
        self.character_indexes = [i for i, cat in enumerate(tags_df["category"].to_list()) if cat == 4]

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        # 透明部分の処理（条件分岐方式）
        canvas = Image.new("RGB", image.size, (255, 255, 255))
        if image.mode == "RGBA":
            canvas.paste(image, mask=image.split()[3])
        else:
            canvas.paste(image)

        # アスペクト比保持処理
        width, height = canvas.size
        max_dim = max(width, height)
        padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded.paste(canvas, ((max_dim - width) // 2, (max_dim - height) // 2))

        # モデル入力サイズを取得してリサイズ
        input_shape = self.components["model"].get_inputs()[0].shape
        target_size = input_shape[2:4] if input_shape[0] == 1 else input_shape[1:3]
        resized = padded.resize(target_size, Image.Resampling.LANCZOS)

        # BGRに変換してバッチ次元追加
        img_array = np.array(resized, dtype=np.float32)[:, :, ::-1]
        return np.expand_dims(img_array, axis=0)

    def _run_inference(self, input_data: np.ndarray) -> dict[str, dict[str, float]]:
        # 入出力名の取得
        input_name = self.components["model"].get_inputs()[0].name
        label_name = self.components["model"].get_outputs()[0].name

        # 推論実行
        results = self.components["model"].run([label_name], {input_name: input_data})
        # 推論結果をフォーマット
        return results[0]

    def _format_predictions(self, raw_output: np.ndarray) -> dict[str, dict[str, float]]:
        """
        モデルの生出力からタグを計算します。

        Args:
            raw_output: モデルからの生出力。

        Returns:
             dict[str, dict[str, float]]:
            評価タグ、一般タグ、キャラクタータグの辞書。
        # NOTE: 閾値は model_output の中に含まれている値からライブラリ使う方で取り出したほうが良い気がする
        # デフォルト閾値 は annotation を直接タグに使う用の設定
        """
        # カテゴリ別の処理を分離
        ratings = self._extract_ratings(raw_output)
        general_tags = self._extract_general_tags(raw_output)
        character_tags = self._extract_character_tags(raw_output)

        return {
            "ratings": ratings,
            "general": general_tags,
            "character": character_tags,
        }

    def _generate_tags(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """全てのタグから一般タグとキャラクタータグをしきい値に基づいてタグを取得し、
        一つのリストとして返します。

        Args:
            raw_output (dict[str, dict[str, float]]):
                'general'と'character'をキーとする予測結果の辞書。
                各値は、タグ名と確率値のペアを含む辞書。

        Returns:
            list[str]: 選択されたタグのリスト。エスケープ処理済み。
        """
        # 一般タグの処理
        general_tags = formatted_output["general"]
        general_probs = np.array(list(general_tags.values()))
        general_threshold = self._calculate_mcut_threshold(general_probs)
        general_threshold = max(0.35, general_threshold)  # 最低閾値を保証

        # キャラクタータグの処理
        character_tags = formatted_output["character"]
        character_probs = np.array(list(character_tags.values()))
        character_threshold = self._calculate_mcut_threshold(character_probs)
        character_threshold = max(0.85, character_threshold)  # 最低閾値を保証

        # 閾値以上のタグを選択
        selected_general = [tag for tag, prob in general_tags.items() if prob > general_threshold]
        selected_character = [tag for tag, prob in character_tags.items() if prob > character_threshold]

        # 全てのタグを結合
        all_selected_tags = selected_general + selected_character

        # エスケープ処理を適用
        escaped_tags = [tag.replace("(", r"\(").replace(")", r"\)") for tag in all_selected_tags]

        # 結果を返す
        return escaped_tags

    def _extract_ratings(self, output_data: np.ndarray) -> dict[str, float]:
        """評価タグを抽出します。"""
        # ラベルと予測値をマッピング
        labels = list(zip(self.labels, output_data[0].astype(float)))
        # 評価タグのみ取得
        ratings_names = [labels[i] for i in self.rating_indexes]
        return dict(ratings_names)

    def _extract_general_tags(self, output_data: np.ndarray) -> dict[str, float]:
        """一般タグを抽出します。"""
        # ラベルと予測値をマッピング
        labels = list(zip(self.labels, output_data[0].astype(float)))
        # 一般タグのみ取得
        general_names = [labels[i] for i in self.general_indexes]
        return dict(general_names)

    def _extract_character_tags(self, output_data: np.ndarray) -> dict[str, float]:
        """キャラクタータグを抽出します。"""
        # ラベルと予測値をマッピング
        labels = list(zip(self.labels, output_data[0].astype(float)))
        # キャラクタータグのみ取得
        character_names = [labels[i] for i in self.character_indexes]
        return dict(character_names)

    def _calculate_mcut_threshold(self, probs: np.ndarray) -> float:
        """Maximum Cut Thresholding (MCut)アルゴリズムで閾値を計算します。"""
        sorted_probs = probs[probs.argsort()[::-1]]
        if len(sorted_probs) <= 1:
            return 0.0
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = difs.argmax()
        threshold = (sorted_probs[t] + sorted_probs[t + 1]) / 2
        return threshold


if __name__ == "__main__":
    with ONNXModelTagger("wd-swinv2-tagger-v3") as tagger:
        image = Image.open(Path("tests/resources/img/1_img/file01.webp"))
        results = tagger.predict([image])
        print(results[0]["annotation"])
