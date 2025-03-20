import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, TypedDict

import numpy as np
import onnxruntime as ort
import polars as pl
import torch
from PIL import Image
from transformers import AutoProcessor

from .model_factory import ModelLoad
from .utils import load_model_config

logger = logging.getLogger(__name__)


# model_factory によって生成されるモデルのコンポーネントを表す型定義
class ModelComponents(TypedDict):
    # TransformerModel
    model: Optional[torch.nn.Module]  # PyTorchモデルのインスタンス
    processor: Optional[AutoProcessor]  # AutoProcessorのインスタンス

    # ONNXModel
    session: Optional[ort.InferenceSession]  # ONNXセッション
    csv_path: Optional[str]  # ローカルに保存されたCSVファイルのパス


class BaseTagger(ABC):
    def __init__(self, model_name: str):
        """BaseTagger を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = load_model_config()[model_name]

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
                annotation_list = self._generate_tags(formatted_output)
                # 結果を標準形式で追加
                results.append(self._generate_result(formatted_output, annotation_list))
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


class TransformerModel(BaseTagger):
    """Transformersライブラリを使用するモデル用の抽象クラス。
    BLIP、BLIP2、GITなどのHugging Face Transformersベースのモデルの基底クラスとして機能します。
    """

    def __init__(self, model_name: str):
        """TransformerModel を初期化します。
        Args:
            model_name (str): モデルの名前。
        """
        super().__init__(model_name)
        # 設定ファイルから追加パラメータを取得
        self.max_length = self.config.get("max_length", 75)
        self.processor_path = self.config.get("processor_path", self.model_path)

    def __enter__(self) -> "TransformerModel":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        loaded_model = ModelLoad.load_transformer_components(
            self.model_name,
            self.model_path,
            self.device,
        )
        if loaded_model is not None:
            self.components = loaded_model
        self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
        return self

    def __exit__(self, exception_type: type[Exception], exception_value: Exception, traceback: Any) -> None:
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

    def _preprocess_image(self, image: Image.Image) -> dict[str, Any]:
        """画像を前処理してモデル入力形式に変換します。
        Args:
            image (Image.Image): 入力画像

        Returns:
            ProcessorOutput: モデル用に処理された入力データ（pixel_valuesなどのテンソルを含む辞書）
        """
        if self.components["processor"] is None:
            raise ValueError("画像をTensorに変換するためのProcessorが初期化されていません。")

        processor = self.components["processor"]
        # プロセッサの出力を取得してデバイスに移動
        processed_output: dict[str, Any] = processor(images=image, return_tensors="pt").to(self.device)

        self.logger.debug(f"辞書のキー: {processed_output.keys()}")
        for key, tensor in processed_output.items():
            self.logger.debug(f"キー: {key}, デバイス: {tensor.device}, 形状: {tensor.shape}")

        return processed_output

    def _run_inference(self, processed_image: dict[str, Any]) -> torch.Tensor:
        """モデル推論を実行します。
        Args:
            processed_image (dict[str, Any]): モデルへの入力データ

        Returns:
            torch.Tensor: モデルからの出力
        """
        model = self.components["model"]

        model_out: torch.Tensor = model.generate(**processed_image)
        self.logger.debug(f"推論結果のデバイス: {model_out.device}, 形状: {model_out.shape}")
        return model_out

    def _format_predictions(self, token_ids: torch.Tensor) -> list[str]:
        # BLIP モデルの出力後処理を実装
        processor = self.components["processor"]
        annotations_list: list[str] = processor.batch_decode(token_ids, skip_special_tokens=True)
        return annotations_list

    def _generate_tags(self, formatted_output: list[str]) -> list[str]:
        """
        キャプションなのでこの処理は不要
        """
        return formatted_output


class ONNXModel(BaseTagger):
    """ONNXランタイムを使用するモデル用の抽象クラス。
    本質的にはWD-Tagger用のクラス
    """

    def __init__(self, model_name: str):
        """ONNXModel を初期化します。
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

    def __enter__(self) -> "ONNXModel":
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

    def _load_labels(self) -> None:
        """ラベル情報をロードし、カテゴリごとのインデックスを設定します。"""
        # ラベルファイルをpolarsで読み込み
        tags_df = pl.read_csv(self.components["csv_path"])

        # ラベル名を取得
        self.labels = tags_df["name"].to_list()

        # カテゴリインデックスを設定
        self.rating_indexes = [i for i, cat in enumerate(tags_df["category"].to_list()) if cat == 9]
        self.general_indexes = [i for i, cat in enumerate(tags_df["category"].to_list()) if cat == 0]
        self.character_indexes = [i for i, cat in enumerate(tags_df["category"].to_list()) if cat == 4]

    def _preprocess_image(self, image: Image.Image) -> np.ndarray[Any, np.dtype[np.float32]]:
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

    def _run_inference(self, input_data: np.ndarray[Any, np.dtype[Any]]) -> np.ndarray[Any, np.dtype[Any]]:
        # 入出力名の取得
        input_name = self.components["model"].get_inputs()[0].name
        label_name = self.components["model"].get_outputs()[0].name

        # 推論実行
        raw_output: list[np.ndarray[Any, np.dtype[Any]]] = self.components["model"].run(
            [label_name], {input_name: input_data}
        )
        # 推論結果をフォーマット
        return raw_output[0]

    def _format_predictions(
        self, raw_output: np.ndarray[Any, np.dtype[Any]]
    ) -> dict[str, dict[str, float]]:
        """
        モデルの生出力からタグを計算します。

        Args:
            raw_output: モデルからの生出力。

        Returns:
             dict[str, dict[str, float]]:
            評価タグ、一般タグ、キャラクタータグの辞書。
        """
        # NOTE: 閾値は model_output の中に含まれている値からライブラリ使う方で取り出したほうが良い気がする
        # デフォルト閾値 は annotation を直接タグに使う用の設定
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

    def _extract_ratings(self, raw_output: np.ndarray[Any, np.dtype[Any]]) -> dict[str, float]:
        """評価タグを抽出します。"""
        # ラベルと予測値をマッピング
        labels = list(zip(self.labels, raw_output[0].astype(float), strict=False))
        # 評価タグのみ取得
        ratings_names = [labels[i] for i in self.rating_indexes]
        return dict(ratings_names)

    def _extract_general_tags(self, raw_output: np.ndarray[Any, np.dtype[Any]]) -> dict[str, float]:
        """一般タグを抽出します。"""
        # ラベルと予測値をマッピング
        labels = list(zip(self.labels, raw_output[0].astype(float), strict=False))
        # 一般タグのみ取得
        general_names = [labels[i] for i in self.general_indexes]
        return dict(general_names)

    def _extract_character_tags(self, raw_output: np.ndarray[Any, np.dtype[Any]]) -> dict[str, float]:
        """キャラクタータグを抽出します。"""
        # ラベルと予測値をマッピング
        labels = list(zip(self.labels, raw_output[0].astype(float), strict=False))
        # キャラクタータグのみ取得
        character_names = [labels[i] for i in self.character_indexes]
        return dict(character_names)

    def _calculate_mcut_threshold(self, probs: np.ndarray[Any, np.dtype[Any]]) -> float:
        """Maximum Cut Thresholding (MCut)アルゴリズムで閾値を計算します。"""
        sorted_probs = probs[probs.argsort()[::-1]]
        if len(sorted_probs) <= 1:
            return 0.0
        difs = sorted_probs[:-1] - sorted_probs[1:]
        t = difs.argmax()
        threshold = (sorted_probs[t] + sorted_probs[t + 1]) / 2
        return float(threshold)


## こっからさき参考程度なので変更しない

# class PipelineModel(BaseTagger):
#     def __init__(self, model_name: str):
#         """PipelineModel を初期化します。

#         Args:
#             model_name (str): モデルの名前。
#         """
#         super().__init__(model_name=model_name)
#         self.batch_size = self.config.get("batch_size", 8)

#     def __enter__(self) -> "PipelineModel":
#         """
#         モデルの状態に基づいて、必要な場合のみロードまたは復元
#         """
#         loaded_model = ModelLoad.pipeline_model_load(
#             self.model_name,
#             self.model_path,
#             self.batch_size,
#             self.device,
#         )
#         if loaded_model is not None:
#             self.components = loaded_model
#         self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
#         return self

#     def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
#         """パイプラインモデルで画像リストのアノテーション結果を予測します。

#         Args:
#             images (list[Image.Image]): アノテーション対象の画像リスト。

#         Returns:
#             list[dict]: アノテーション結果の辞書リスト。
#         """
#         results = []
#         for image in images:
#             pipeline_model = self.components["pipeline"]
#             raw_output = pipeline_model(image)
#             self.logger.debug(f"モデル '{self.model_name}' に処理された生の出力結果: {raw_output}")
#             annotation = self._calculate_annotation(raw_output)
#             annotation_tag = self._get_annotation_tag(annotation)
#             results.append(
#                 {
#                     "model_name": self.model_name,
#                     "model_output": raw_output,
#                     "annotation_tag": annotation_tag,
#                 }
#             )
#         return results

#     @abstractmethod
#     def _calculate_annotation(self, raw_output: Any) -> float:
#         """出力からアノテーションを計算します。

#         Args:
#             raw_output: モデルからの生出力。

#         Returns:
#             float: 計算されたアノテーション。
#         """
#         pass

#     @abstractmethod
#     def _get_annotation_tag(self, annotation_float: float) -> list[str]:
#         """パイプラインモデル用のアノテーションタグを生成します。

#         Args:
#             annotation_float (float): 計算されたアノテーション。

#         Returns:
#             list[str]: 生成されたアノテーションタグ。
#         """
#         pass


# class ClipModel(BaseTagger):
#     """CLIPモデルを使用するアノテーションラークラス。

#     任意のCLIPモデルと組み合わせたアノテーションリングモデルを扱います。
#     activation_typeとfinal_activation_typeの設定に基づいて
#     活性化関数の有無を動的に決定します。

#     Args:
#         model_name (str): モデルの名前
#     """

#     def __init__(self, model_name: str):
#         super().__init__(model_name=model_name)
#         self.base_model = self.config["base_model"]
#         self.activation_type = self.config.get("activation_type", None)
#         self.final_activation_type = self.config.get("final_activation_type", None)

#     def __enter__(self) -> "ClipModel":
#         """
#         モデルの状態に基づいて、必要な場合のみロードまたは復元
#         """
#         loaded_model = ModelLoad.clip_model_load(
#             self.model_name,
#             self.base_model,
#             self.model_path,
#             self.device,
#             self.activation_type,
#             self.final_activation_type,
#         )
#         if loaded_model is not None:
#             self.components = loaded_model
#         self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
#         return self

#     # image_embeddings 関数 (WaifuAesthetic で使用されているものを流用)
#     def image_embeddings(self, image: Image.Image) -> np.ndarray[Any, np.dtype[Any]]:
#         """画像から CLIP モデルを使用して埋め込みベクトルを生成します。

#         Args:
#             image (Image.Image): 埋め込みを生成するための入力画像
#             model (CLIPModel): 特徴抽出に使用する CLIP モデル
#             processor (CLIPProcessor): 画像の前処理を行う CLIP プロセッサ

#         Returns:
#             np.ndarray: 正規化された画像の埋め込みベクトル
#         """
#         processor = self.components["processor"]
#         model = self.components["clip_model"]
#         inputs = processor(images=image, return_tensors="pt")["pixel_values"]
#         inputs = inputs.to(model.device)
#         result = model.get_image_features(pixel_values=inputs).cpu().detach().numpy()
#         # 正規化された埋め込みを返す
#         normalized_result: np.ndarray[Any, np.dtype[Any]] = result / np.linalg.norm(result)
#         return normalized_result

#     def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
#         """CLIPモデルで画像リストのアノテーション結果を予測します。

#         Args:
#             images (list[Image.Image]): アノテーション対象の画像リスト。

#         Returns:
#             list[dict]: アノテーション結果の辞書リスト。
#         """

#         results = []
#         for image in images:
#             with torch.no_grad():  # NOTE: あると推論速度が早くなる
#                 # 画像の特徴量抽出
#                 image_embedding = self.preprocess(image)
#                 tensor_input = self._prepare_tensor(image_embedding)

#                 # モデル推論
#                 raw_annotation = self.components["model"](tensor_input).item()
#                 self.logger.debug(f"モデル '{self.model_name}' に処理された生の出力結果: {raw_annotation}")

#             # アノテーション計算とタグ生成
#             calculated_annotation = self._calculate_annotation(raw_annotation)
#             self.logger.debug(
#                 f"モデル '{self.model_name}' 生の結果はTensorなのでfloatに変換: {calculated_annotation}"
#             )
#             annotation_list = self._get_annotation_tag(calculated_annotation)
#             results.append(self._generate_result(raw_annotation, annotation_list))

#         return results

#     def _calculate_annotation(self, raw_annotation: torch.Tensor) -> float:
#         """モデルの生出力からアノテーションを計算します。
#         Args:
#             raw_annotation: モデルからの生出力。

#         Returns:
#             annotation_float (float): 計算されたアノテーション。
#         """
#         return float(raw_annotation)

#     def preprocess(self, image: Image.Image) -> torch.Tensor:
#         """画像をCLIPモデル用のテンソルに前処理します。

#         Args:
#             image (Image.Image): 入力画像。

#         Returns:
#             torch.Tensor: 前処理済みのテンソル。
#         """
#         return torch.tensor(self.image_embeddings(image))

#     def _prepare_tensor(self, model_output: Any) -> torch.Tensor:
#         """テンソルを設定されたデバイスへ移動させます。

#         Args:
#             model_output: 変換対象の出力。

#         Returns:
#             torch.Tensor: 指定デバイス上のテンソル。
#         """
#         if isinstance(model_output, torch.Tensor):
#             tensor = model_output.float()
#         else:
#             tensor = torch.from_numpy(model_output).float()
#         return tensor.to(self.device)
