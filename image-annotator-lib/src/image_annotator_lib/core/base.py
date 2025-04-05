"""画像アノテーションライブラリの基底クラスと型定義。

このモジュールは、画像アノテーション(タギング、スコアリングなど)を行う
すべてのモデルクラスの基底となる抽象クラス `BaseAnnotator` と、
関連する型定義(`AnnotationResult`, `ModelComponents`, `TagConfidence`)、
およびフレームワーク固有の基底クラスを提供します。
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Self,
    TypedDict,
)

# --- 依存ライブラリのインポート ---
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoProcessor

# --- ローカルインポート ---
from ..exceptions.errors import ModelLoadError, OutOfMemoryError
from .model_factory import ModelLoad
from .utils import load_model_config, setup_logger

# ロガーの初期化
logger = setup_logger(__name__)

# --- 型定義 ---


class ModelComponents(TypedDict, total=False):
    """モデルコンポーネントを表す型定義。

    `ModelLoad` クラスの各 `load_..._components` メソッドが返す辞書のキーを定義します。
    フレームワークやモデルタイプによって実際に含まれるキーは異なります。

    Attributes:
        model: ロードされたモデルオブジェクト (PyTorch nn.Module, TF Keras Model/Module, Classifier)。
        processor: Transformers のプロセッサオブジェクト (AutoProcessor 互換)。
        session: ONNX Runtime の推論セッション (ort.InferenceSession)。
        csv_path: ONNX モデルで使用されるタグ情報 CSV ファイルへのパス (str)。
        clip_model: CLIP ベースモデルオブジェクト (PyTorch nn.Module)。
        pipeline: Transformers のパイプラインオブジェクト。
        model_dir: TensorFlow SavedModel のディレクトリパス (Path)。
    """

    model: Any
    processor: Any
    session: ort.InferenceSession | None
    csv_path: str | None
    clip_model: Any
    pipeline: Any
    model_dir: Path | None


class AnnotationResult(TypedDict, total=False):
    """単一画像の標準化されたアノテーション結果。

    `BaseAnnotator.predict` メソッドの戻り値リストの要素型です。

    Attributes:
        phash: 画像の知覚ハッシュ (str)。計算失敗時は None。
        tags: アノテーション結果の主要な文字列リスト (list[str])。
               タガーの場合はタグ、スコアラーの場合はスコアタグ、
               キャプショナーの場合はキャプションが入ります。
        formatted_output: 整形済み出力 (Any)。`_format_predictions` の戻り値。
                          デバッグや詳細分析に使用できます。
        error: 処理中に発生したエラーメッセージ (str)。エラーがない場合は None。
    """

    phash: str | None
    tags: list[str]
    formatted_output: Any
    error: str | None


class TagConfidence(TypedDict):
    """タグとその信頼度、情報源を保持する型定義。"""

    confidence: float
    source: str


# --- 基底クラス ---


class BaseAnnotator(ABC):
    """画像アノテーションモデルの抽象基底クラス。

    すべてのアノテーター (Tagger, Scorer, Captioner など) はこのクラスを継承します。
    共通の初期化処理、コンテキスト管理のインターフェース、予測処理の骨格を提供します。

    Attributes:
        model_name (str): モデル設定ファイル (`models.toml`) 内のモデル名。
        DEFAULT_CHUNK_SIZE (int): `predict` メソッドでのデフォルトのチャンクサイズ。
        logger (logging.Logger): このクラスインスタンス用のロガー。
        config (dict[str, Any]): `models.toml` からロードされたこのモデルの設定。
        model_path (str): モデルファイルまたはディレクトリへのパス (必須設定)。
        device (str): 推論に使用するデバイス ("cuda", "cpu" など)。
        chunk_size (int): 一度に処理する画像の数。
        components (dict[str, Any]): ロードされたモデルコンポーネントを保持する辞書。
                                     キーと値の型は `ModelComponents` TypedDict を参照。
    """

    DEFAULT_CHUNK_SIZE: int = 8

    def __init__(self, model_name: str):
        """BaseAnnotator を初期化します。

        モデル名に基づいて設定ファイルを読み込み、共通の属性を設定します。

        Args:
            model_name: `models.toml` 内のモデル名。

        Raises:
            ValueError: 設定ファイルが見つからない、または必須キー (`model_path`) が
                        設定ファイル内に存在しない場合。
            RuntimeError: 初期化中に予期せぬエラーが発生した場合。
        """
        self.model_name = model_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"{self.__class__.__name__} をモデル '{model_name}' で初期化中...")

        try:
            all_configs = load_model_config()
            if model_name not in all_configs:
                raise ValueError(f"モデル '{model_name}' の設定が設定ファイルに見つかりません。")
            self.config: dict[str, Any] = all_configs[model_name]

            try:
                self.model_path = self.config["model_path"]
                self.logger.debug(f"モデルパス: {self.model_path}")
            except KeyError:
                message = f"モデル '{model_name}' の設定に必須キー 'model_path' がありません。"
                self.logger.error(message)
                raise ValueError(message) from None

            # オプション設定の取得とデフォルト値設定
            self.device = self.config.get("device", "cuda")
            self.chunk_size = self.config.get("chunk_size", self.DEFAULT_CHUNK_SIZE)

            # 属性の初期化
            self.components: dict[str, Any] = {}

            self.logger.debug(f"{self.__class__.__name__} '{model_name}' の初期化完了。")

        except Exception as e:
            self.logger.exception(f"モデル '{model_name}' の初期化中にエラーが発生しました: {e}")
            raise RuntimeError(f"モデル '{model_name}' の初期化に失敗しました。") from e

    @abstractmethod
    def __enter__(self) -> Self:
        """コンテキストマネージャのエントリポイント。

        モデルコンポーネントのロード/復元処理を行います。
        実際の処理は `ModelLoad` クラスに委譲されます。
        サブクラスはこのメソッドを実装し、適切な `ModelLoad.load_..._components` を呼び出し、
        その結果を `self.components` に設定する必要があります。

        Returns:
            自身のインスタンス (Self)。
        """
        raise NotImplementedError("サブクラスは __enter__ を実装する必要があります。")

    @abstractmethod
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """コンテキストマネージャの終了ポイント。

        モデルコンポーネントのキャッシュ/解放処理を行います。
        実際の処理は `ModelLoad` クラスに委譲されます。
        サブクラスはこのメソッドを実装し、`ModelLoad.cache_to_main_memory` または
        `ModelLoad.release_model` を適切に呼び出す必要があります。

        Args:
            exception_type: 発生した例外の型 (なければ None)。
            exception_value: 発生した例外インスタンス (なければ None)。
            traceback: トレースバックオブジェクト (なければ None)。
        """
        raise NotImplementedError("サブクラスは __exit__ を実装する必要があります。")

    @abstractmethod
    def _preprocess_images(self, images: list[Image.Image]) -> Any:
        """画像バッチをモデル入力に適した形式に前処理します。

        フレームワーク固有の処理を実装します。

        Args:
            images: 前処理対象の PIL Image オブジェクトのリスト。

        Returns:
            モデルの `_run_inference` メソッドが受け付ける形式の前処理済みデータ。
            形式はフレームワークやモデルによって異なります (例: PyTorch Tensor, NumPy Array)。
        """
        raise NotImplementedError

    @abstractmethod
    def _run_inference(self, processed: Any) -> Any:
        """前処理済みデータを使用してモデル推論を実行します。

        フレームワーク固有の推論処理を実装します。

        Args:
            processed: `_preprocess_images` から返された前処理済みデータ。

        Returns:
            モデルからの生の出力。形式はフレームワークやモデルによって異なります。
        """
        raise NotImplementedError

    @abstractmethod
    def _format_predictions(self, raw_outputs: Any) -> list[Any]:
        """モデルの生出力バッチを、後続処理に適したリスト形式にフォーマットします。

        バッチ内の各画像に対応する整形済み結果を要素とするリストを返します。
        リスト要素の型はモデルの種類によって異なります。

        Args:
            raw_outputs: `_run_inference` から返された生出力。

        Returns:
            整形済み予測結果のリスト。
            例:
            - タガー (カテゴリ別): `list[dict[str, dict[str, float]]]`
            - キャプショナー: `list[str]`
            - スコアラー: `list[float]`
        """
        raise NotImplementedError

    @abstractmethod
    def _generate_tags(self, formatted_output: Any) -> list[str]:
        """整形済み出力から最終的なタグリスト (`list[str]`) を生成します。

        タガー、スコアラー、キャプショナーなど、モデルの種類に応じて、
        `_format_predictions` の結果を解釈し、`AnnotationResult.tags` に
        格納するための文字列リストを作成します。

        Args:
            formatted_output: `_format_predictions` の戻り値リストの単一要素。

        Returns:
            タグ、スコアタグ、またはキャプションを含む文字列リスト。
        """
        raise NotImplementedError("サブクラスは _generate_tags を実装する必要があります。")

    def _generate_result(
        self,
        phash: str | None,
        tags: list[str],
        formatted_output: Any,
        error: str | None = None,
    ) -> AnnotationResult:
        """標準化された AnnotationResult 辞書を生成します。

        このメソッドは BaseAnnotator で共通実装を提供するため、
        サブクラスでオーバーライドしないでください。

        Args:
            phash: 画像の知覚ハッシュ。
            tags: 生成されたタグ、スコアタグ、またはキャプションのリスト。
            formatted_output: 整形済みのモデル出力。
            error: エラーメッセージ (あれば)。

        Returns:
            AnnotationResult 型の辞書。
        """
        return {
            "phash": phash,
            "tags": tags if isinstance(tags, list) else [tags],
            "formatted_output": formatted_output,
            "error": error,
        }

    @torch.no_grad()
    def predict(self, images: list[Image.Image], phash_list: list[str]) -> list[AnnotationResult]:
        """画像リストに対して予測を実行し、結果を返します。チャンクに分割してバッチ処理します。"""
        all_results: list[AnnotationResult] = []
        num_images = len(images)
        chunk_size = self.chunk_size if hasattr(self, "chunk_size") else self.DEFAULT_CHUNK_SIZE

        self.logger.info(
            f"モデル '{self.model_name}' で {num_images} 枚の画像をチャンクサイズ {chunk_size} で処理します。"
        )

        # 画像リストをチャンクに分割してループ処理
        for i in range(0, num_images, chunk_size):
            chunk_images = images[i : i + chunk_size]
            chunk_phash_list = phash_list[i : i + chunk_size] if i < len(phash_list) else []
            current_chunk_size = len(chunk_images)

            self.logger.debug(
                f"チャンク {i // chunk_size + 1}/{(num_images + chunk_size - 1) // chunk_size} (サイズ: {current_chunk_size}) を処理中..."
            )

            try:
                # 1. 前処理
                processed_batch = self._preprocess_images(chunk_images)

                # 2. 推論
                raw_outputs = self._run_inference(processed_batch)

                # 3. フォーマット (バッチ全体の結果をリストで返す)
                formatted_outputs = self._format_predictions(raw_outputs)

                # 4. 各画像ごとにタグを生成して結果を作成
                for j, formatted_output in enumerate(formatted_outputs):
                    # 個々の画像に対してタグを生成
                    tags = self._generate_tags(formatted_output)

                    # 対応するpHashを取得
                    phash = chunk_phash_list[j] if j < len(chunk_phash_list) else None

                    # 結果を生成
                    result = self._generate_result(
                        phash=phash, tags=tags, formatted_output=formatted_output, error=None
                    )
                    all_results.append(result)

            except Exception as e:
                self.logger.error(f"チャンク {i // chunk_size + 1} の処理中にエラーが発生: {e}")

                # エラーが発生した場合は、チャンク内の各画像にエラー結果を追加
                for j, _ in enumerate(chunk_images):
                    phash = chunk_phash_list[j] if j < len(chunk_phash_list) else None
                    result = self._generate_result(
                        phash=phash, tags=[], formatted_output=None, error=str(e)
                    )
                    all_results.append(result)

                raise

        self.logger.debug(
            f"モデル '{self.model_name}' の全チャンク処理が完了しました。合計 {len(all_results)} 件の結果を生成しました。"
        )
        return all_results

    def _extract_category_tags(
        self, attr_name: str, labels_with_probs: list[tuple[str, float]]
    ) -> dict[str, float]:
        """カテゴリータグを抽出するヘルパー関数 (ONNX/TF タガー用)。"""
        category_tags: dict[str, float] = {}
        indexes = getattr(self, attr_name, [])
        for i in indexes:
            if 0 <= i < len(labels_with_probs):
                tag_name, prob = labels_with_probs[i]
                category_tags[tag_name] = prob
            else:
                self.logger.warning(f"インデックス {i} が範囲外です (ラベル数: {len(self.labels)})。")
        return category_tags

    def _format_predictions_single(
        self, raw_output: np.ndarray[Any, np.dtype[Any]]
    ) -> dict[str, dict[str, float]]:
        """単一の生出力をカテゴリ別にフォーマットします (ONNX/TF タガー用)。"""
        result: dict[str, dict[str, float]] = {}
        if not hasattr(self, "labels") or not self.labels:
            self.logger.warning("ラベルがロードされていません。フォーマットできません。")
            return {"error": {}}
        if raw_output.ndim == 2 and raw_output.shape[0] == 1:
            predictions = raw_output[0].astype(float)
        elif raw_output.ndim == 1:
            predictions = raw_output.astype(float)
        else:
            self.logger.error(f"予期しない生出力形状: {raw_output.shape}")
            return {"error": {}}
        if len(self.labels) != len(predictions):
            self.logger.error(
                f"ラベル数 ({len(self.labels)}) と予測数 ({len(predictions)}) が一致しません。"
            )
            return {"error": {}}
        labels_with_probs = list(zip(self.labels, predictions, strict=True))
        if not hasattr(self, "_category_attr_map") or not self._category_attr_map:
            self.logger.warning(
                "_category_attr_map がサブクラスで定義されていません。カテゴリ分類なしでフォーマットします。"
            )
            result["general"] = {label: float(prob) for label, prob in labels_with_probs}
            return result

        for category_key, attr_name in self._category_attr_map.items():
            category_tags = self._extract_category_tags(attr_name, labels_with_probs)
            if category_tags:
                result[category_key] = category_tags
        if "rating" in result and "ratings" not in result:
            result["ratings"] = result.pop("rating")
        return result

    def _generate_tags_single(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """フォーマットされた単一出力からタグリストを生成します (ONNX/TF タガー用)。"""
        tags = []
        if not formatted_output or "error" in formatted_output:
            return []

        for category, tag_dict in formatted_output.items():
            if category == "error":
                continue
            for tag, confidence in tag_dict.items():
                if confidence >= self.tag_threshold:
                    tags.append((tag, confidence))

        unique_tags: dict[str, float] = {}
        for tag, conf in tags:
            if tag not in unique_tags or conf > unique_tags[tag]:
                unique_tags[tag] = conf

        return [tag for tag, _ in sorted(unique_tags.items(), key=lambda x: x[1], reverse=True)]


# --- フレームワーク別基底クラス ---


class TransformersBaseAnnotator(BaseAnnotator):
    """Transformers ライブラリを使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        """TransformerModel を初期化します。
        Args:
            model_name (str): モデルの名前。
        """
        super().__init__(model_name)
        # 設定ファイルから追加パラメータを取得
        self.max_length = self.config.get("max_length", 75)
        self.processor_path = self.config.get("processor_path", self.model_path)

    def __enter__(self) -> "TransformersBaseAnnotator":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        メモリ不足エラーをハンドリングし、VRAM使用量をログに出力
        """
        try:
            # --- モデルロード処理 ---
            logger.info(f"モデルコンポーネントのロード試行: {self.model_name} をデバイス {self.device} へ")
            loaded_model = ModelLoad.load_transformers_components(
                self.model_name,
                self.model_path,
                self.device,
            )
            if loaded_model:
                self.components = loaded_model
                logger.info(f"モデルコンポーネントのロード成功: {self.model_name}")

            # --- CUDAへの復元処理 ---
            logger.debug(f"モデル {self.model_name} を {self.device} へ復元試行")
            self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.components, self.device)
            logger.debug(f"モデル {self.model_name} の {self.device} への復元成功")
        except Exception as e:
            # その他のロード/復元時エラー
            logger.exception(f"モデル {self.model_name} のロード/復元に失敗: {e}")
            raise

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components, self.device)

    def _preprocess_image(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像バッチを前処理します。各画像を個別に処理して結果をリストで返します。"""
        results = []
        for image in images:
            # プロセッサの出力を取得してデバイスに移動
            processed_output = self.components["processor"](images=image, return_tensors="pt").to(
                self.device
            )
            self.logger.debug(f"辞書のキー: {processed_output.keys()}")
            results.append(processed_output)
        return results

    def _run_inference(self, processed: list[dict[str, torch.Tensor]]) -> list[torch.Tensor]:
        """前処理済みバッチで推論を実行します (Transformers用)。"""
        if "model" not in self.components or self.components["model"] is None:
            raise RuntimeError("Transformer モデルがロードされていません。")
        model: Any = self.components["model"]
        outputs = []
        try:
            with torch.no_grad():
                for processed_image in processed:
                    if hasattr(model, "generate"):
                        model_out = model.generate(**processed_image, max_length=self.max_length)
                    else:
                        model_out = model(**processed_image)
                        if hasattr(model_out, "last_hidden_state"):
                            model_out = model_out.last_hidden_state
                        elif hasattr(model_out, "logits"):
                            model_out = model_out.logits
                    outputs.append(model_out)
            return outputs
        except torch.cuda.OutOfMemoryError as e:
            error_message = f"CUDAメモリ不足: モデル '{self.model_name}' の推論実行中"
            self.logger.error(error_message)
            try:
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    self.logger.error(torch.cuda.memory_summary(device=self.device))
            except Exception as mem_e:
                self.logger.error(f"CUDAメモリサマリーの取得に失敗: {mem_e}")
            raise OutOfMemoryError(error_message) from e
        except Exception as e:
            self.logger.exception(f"モデル '{self.model_name}' の推論実行中にエラーが発生: {e}")
            raise RuntimeError(f"推論エラー: {e}") from e

    def _format_predictions(self, token_ids_list: list[torch.Tensor]) -> list[str]:
        """生出力バッチをフォーマットします (Transformers用、テキストデコード)。"""
        if "processor" not in self.components or self.components["processor"] is None:
            raise RuntimeError("Transformer プロセッサがロードされていません。")
        processor: AutoProcessor = self.components["processor"]
        all_formatted = []
        try:
            for token_ids in token_ids_list:
                decoded_texts: list[str] | str = processor.batch_decode(token_ids, skip_special_tokens=True)
                if isinstance(decoded_texts, str):
                    all_formatted.append(decoded_texts)
                else:
                    all_formatted.append(decoded_texts[0] if decoded_texts else "")
            return all_formatted
        except Exception as e:
            self.logger.exception(f"予測結果のフォーマット中にエラー発生: {e}")
            raise ValueError(f"予測結果のフォーマット失敗: {e}") from e

    def _generate_tags(self, formatted_output: str) -> list[str]:
        """キャプション文字列を単一要素のリストに変換します。"""
        return [formatted_output]


class TensorflowBaseAnnotator(BaseAnnotator):
    """TensorFlow モデルを使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        if tf is None:
            raise ImportError("TensorFlow がインストールされていません。")
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                self.logger.debug("TensorFlow GPU メモリ成長を有効化しました。")
            else:
                self.logger.debug("TensorFlow: 利用可能な GPU が見つかりません。")
        except Exception as gpu_e:
            self.logger.warning(f"TensorFlow GPU 設定中にエラー: {gpu_e}")
        self.model_format: str = self.config.get("model_format", "h5")

    def __enter__(self) -> "TensorflowBaseAnnotator":
        """TensorFlow モデルコンポーネントをロードします。状態管理は ModelLoad に委譲します。"""
        self.logger.debug(f"Entering context for TensorFlow model '{self.model_name}'")
        try:
            if not self.model_path:
                raise ValueError("'model_path' が設定されていません。")
            self.logger.info(
                f"Loading/Restoring TensorFlow components: model='{self.model_path}', format='{self.model_format}'"
            )
            loaded_components = ModelLoad.load_tensorflow_components(
                self.model_name,
                self.model_path,
                self.device,
                self.model_format,
            )
            if loaded_components is None:
                raise ModelLoadError(f"モデル '{self.model_name}' のロード/復元に失敗しました。")
            self.components = loaded_components
            self._load_tags()  # TFモデル固有のタグロード処理
            self.logger.info(f"モデル '{self.model_name}' を正常にロードしました")
        except (ModelLoadError, OutOfMemoryError, FileNotFoundError, ValueError) as e:
            self.logger.error(f"TensorFlow モデル '{self.model_name}' のロード/準備中にエラー: {e}")
            self.components = {}
            raise
        except Exception as e:
            self.logger.exception(
                f"TensorFlow モデル '{self.model_name}' のロード/準備中に予期せぬエラー: {e}"
            )
            self.components = {}
            raise ModelLoadError(f"予期せぬロードエラー: {e}") from e
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """TensorFlow モデルのリソースを解放します。"""
        self.logger.debug(
            f"Exiting context for TensorFlow model '{self.model_name}' (exception: {exc_type})"
        )
        if self.components:
            try:
                components_to_release = self.components
                self.components = ModelLoad.release_model_components(self.model_name, components_to_release)
                self.logger.debug("TensorFlow Keras セッションクリアを試行 (必要な場合)。")
                if tf:
                    tf.keras.backend.clear_session()
            except Exception as e:
                self.logger.exception(f"TensorFlow モデル '{self.model_name}' の解放中にエラー: {e}")
            finally:
                self.components = {}
        if exc_type:
            self.logger.error(
                f"TensorFlow モデル '{self.model_name}' のコンテキスト内で例外発生: {exc_val}"
            )

    @abstractmethod
    def _load_tags(self) -> None:
        """モデル固有のタグ情報 (例: tags.txt) をロードします。"""
        raise NotImplementedError("サブクラスは _load_tags を実装する必要があります。")

    def _load_tag_file(self, tags_path: Path) -> list[str]:
        """タグファイルを読み込み、タグのリストを返します。"""
        if not tags_path.is_file():
            self.logger.error(f"タグファイルが見つかりません: {tags_path}")
            return []
        try:
            with open(tags_path, encoding="utf-8") as f:
                tags = [line.strip() for line in f if line.strip()]
                self.logger.debug(f"{tags_path.name} から {len(tags)} 個のタグをロードしました。")
                return tags
        except Exception as e:
            self.logger.exception(f"タグファイル '{tags_path}' の読み込みエラー: {e}")
            return []

    @abstractmethod
    def _preprocess_images(self, images: list[Image.Image]) -> np.ndarray[Any, np.dtype[np.float32]]:
        """画像リストを前処理し、単一の NumPy 配列バッチを返します。"""
        raise NotImplementedError("TensorFlow サブクラスは _preprocess_images を実装する必要があります。")

    def _run_inference(self, processed: np.ndarray[Any, np.dtype[Any]]) -> tf.Tensor:
        """前処理済みバッチで推論を実行します (TensorFlow用)。"""
        return self._run_inference_tf(processed)

    @abstractmethod
    def _format_predictions(self, raw_output: tf.Tensor) -> list[Any]:
        """モデルの生出力バッチをフォーマットします。"""
        raise NotImplementedError("TensorFlow サブクラスは _format_predictions を実装する必要があります。")

    def _run_inference_tf(self, processed: np.ndarray[Any, np.dtype[Any]]) -> tf.Tensor:
        """TensorFlow モデルでバッチ推論を実行します。"""
        if "model" not in self.components or self.components["model"] is None:
            raise RuntimeError("TensorFlow モデルがロードされていません。")
        tf_model = self.components["model"]
        try:
            self.logger.debug(f"TF 推論実行: 入力形状={processed.shape}")
            raw_output = tf_model(processed, training=False)
            self.logger.debug(f"TF 推論完了: 出力形状={raw_output.shape}")
            return raw_output
        except tf.errors.ResourceExhaustedError as e:
            error_message = f"TensorFlow リソース枯渇 (OOM?) : モデル '{self.model_name}' の推論実行中"
            self.logger.error(error_message)
            raise OutOfMemoryError(error_message) from e
        except Exception as e:
            self.logger.exception(f"TensorFlow モデル '{self.model_name}' の推論実行中にエラーが発生: {e}")
            raise RuntimeError(f"TensorFlow 推論エラー: {e}") from e

    def _generate_tags(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """フォーマットされた単一出力からタグリストを生成します (ONNX/TF タガー用)。"""
        return self._generate_tags_single(formatted_output)


class ClipBaseAnnotator(BaseAnnotator):
    """CLIP モデルをベースとする Scorer 用の基底クラス。"""

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name=model_name)
        self.base_model = self.config.get("base_model")
        if not self.base_model:
            raise ValueError(f"モデル '{model_name}' の設定に 'base_model' (CLIPモデルID) が必要です。")
        if not self.model_path:
            raise ValueError(
                f"モデル '{model_name}' の設定に 'model_path' (分類器ヘッドのパス) が必要です。"
            )
        logger.debug(
            f"ClipBaseAnnotator '{model_name}' initialized. Base CLIP: {self.base_model}, Head: {self.model_path}"
        )

    def __enter__(self) -> Self:
        """CLIP モデルと分類器ヘッドをロードします。"""
        self.logger.debug(f"Entering context for CLIP Scorer '{self.model_name}'")
        try:
            loaded_components = ModelLoad.load_clip_components(
                model_name=self.model_name,
                base_model=self.base_model,
                model_path=self.model_path,
                device=self.device,
                activation_type=self.config.get("activation_type"),
                final_activation_type=self.config.get("final_activation_type"),
            )
            if loaded_components:
                self.components = loaded_components
            self.logger.info(f"CLIP Scorer '{self.model_name}' の準備完了。")

        except (ModelLoadError, OutOfMemoryError, FileNotFoundError, ValueError) as e:
            self.logger.error(f"CLIP Scorer '{self.model_name}' のロード/復元中にエラー: {e}")
            self.components = {}
            raise
        except Exception as e:
            self.logger.exception(f"CLIP Scorer '{self.model_name}' のロード/復元中に予期せぬエラー: {e}")
            self.components = {}
            raise ModelLoadError(f"予期せぬロードエラー: {e}") from e
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """CLIP Scorer モデルをキャッシュします。"""
        self.logger.debug(
            f"Exiting context for CLIP Scorer model '{self.model_name}' (exception: {exc_type})"
        )
        try:
            if self.components:
                self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)
            else:
                ModelLoad.release_model(self.model_name)
        except Exception:
            ModelLoad.release_model(self.model_name)

    def _preprocess_images(self, images: list[Image.Image]) -> dict[str, torch.Tensor]:
        """画像を CLIP プロセッサで前処理します。"""
        if "processor" not in self.components or self.components["processor"] is None:
            raise RuntimeError("CLIP プロセッサがロードされていません。")
        processor = self.components["processor"]
        try:
            inputs = processor(images=images, return_tensors="pt", padding=True, truncation=True)
            return {k: v.to(self.device) for k, v in inputs.items()}
        except Exception as e:
            logger.exception(f"CLIP 画像の前処理中にエラー: {e}")
            raise ValueError(f"CLIP 画像の前処理失敗: {e}") from e

    def _run_inference(self, processed: dict[str, torch.Tensor]) -> torch.Tensor:
        """CLIP モデルで画像特徴量を抽出し、分類器ヘッドでスコアを計算します。"""
        if "clip_model" not in self.components or self.components["clip_model"] is None:
            raise RuntimeError("CLIP ベースモデルがロードされていません。")
        if "model" not in self.components or self.components["model"] is None:
            raise RuntimeError("分類器ヘッド (model) がロードされていません。")

        clip_model = self.components["clip_model"]
        classifier_head = self.components["model"]

        try:
            with torch.no_grad():
                image_features = clip_model.get_image_features(**processed)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                raw_scores = classifier_head(image_features)
                return raw_scores.squeeze(-1)
        except torch.cuda.OutOfMemoryError as e:
            error_message = f"CUDA OOM: CLIP Scorer '{self.model_name}' 推論中"
            self.logger.error(error_message)
            raise OutOfMemoryError(error_message) from e
        except Exception as e:
            logger.exception(f"CLIP Scorer '{self.model_name}' 推論中にエラー: {e}")
            raise RuntimeError(f"CLIP Scorer 推論エラー: {e}") from e

    def _format_predictions(self, raw_outputs: torch.Tensor) -> list[float]:
        """生のスコアテンソルを float のリストに変換します。"""
        try:
            scores = raw_outputs.cpu().numpy().tolist()
            return [float(s) for s in scores]
        except Exception as e:
            logger.exception(f"スコアテンソルのフォーマット中にエラー: {e}")
            try:
                batch_size = raw_outputs.shape[0]
                return [0.0] * batch_size
            except Exception:
                return []

    @abstractmethod
    def _get_score_tag(self, score: float) -> str:
        """スコア値に基づいてスコアタグ文字列を生成します (サブクラスで実装)。"""
        raise NotImplementedError("サブクラスは _get_score_tag を実装する必要があります。")

    def _generate_tags(self, formatted_output: float) -> list[str]:
        """スコア値からスコアタグを生成します。"""
        return [self._get_score_tag(formatted_output)]


class PipelineBaseAnnotator(BaseAnnotator):
    """Hugging Face Pipeline を使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.batch_size = self.config.get("batch_size", 8)

    def __enter__(self) -> "PipelineBaseAnnotator":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        loaded_components = ModelLoad.pipeline_model_load(
            self.model_name,
            self.model_path,
            self.batch_size,
            self.device,
        )
        if loaded_components:
            self.components = loaded_components
        self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Pipeline モデルをキャッシュします。"""
        self.logger.debug(f"Exiting context for Pipeline model '{self.model_name}' (exception: {exc_type})")
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Pipeline は PIL Image を直接受け付けるため、前処理は不要。"""
        return images

    def _format_predictions(self, raw_outputs: list[list[dict[str, Any]]]) -> Any:
        """
        Pipeline の生出力は人間が読めるので不要
        """
        return raw_outputs

    def _run_inference(self, processed: list[Image.Image]) -> list[list[dict[str, Any]]]:
        """Pipeline を使用して推論を実行します。"""
        try:
            raw_outputs = self.components["pipeline"](processed)
            return raw_outputs
        except Exception as e:
            logger.exception(f"Pipeline 推論中にエラーが発生: {e}")
            raise

    @abstractmethod
    def _format_predictions(self, raw_outputs: Any) -> list[Any]:
        """Pipeline の生出力バッチをフォーマットします (サブクラスで実装)。"""
        raise NotImplementedError("Pipeline サブクラスは _format_predictions を実装する必要があります。")


class ONNXBaseAnnotator(BaseAnnotator):
    """ONNX Runtime を使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.labels: list[str] = []
        self.target_size: tuple[int, int] | None = None
        self.is_nchw_expected: bool = False

    def __enter__(self) -> Self:
        """
        ModelLoad を使用して ONNX モデルコンポーネントをロードします。
        """
        try:
            self.components = ModelLoad.load_onnx_components(self.model_name, self.model_path, self.device)
            self._load_labels()
            self._analyze_model_input_format()

        except OutOfMemoryError as e:
            raise e
        except Exception as e:
            logger.exception(f"ONNXモデル {self.model_name} の準備中にエラーが発生: {e}")
            raise

        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """ONNX モデルのリソースを解放します。"""
        self.logger.debug(f"Exiting context for ONNX model '{self.model_name}' (exception: {exc_type})")
        if self.components:
            self.components = ModelLoad.release_model_components(self.model_name, self.components)
        if exc_type:
            self.logger.error(f"ONNX モデル '{self.model_name}' のコンテキスト内で例外発生: {exc_val}")

    @abstractmethod
    def _load_labels(self) -> None:
        """ラベル情報をロードし、必要に応じてカテゴリインデックスを設定します (サブクラスで実装)。"""
        raise NotImplementedError("ONNX サブクラスは _load_labels を実装する必要があります。")

    def _analyze_model_input_format(self) -> None:
        """モデル入力形式を分析し、ターゲットサイズと次元形式を判定・保存する"""
        if "session" not in self.components or self.components["session"] is None:
            raise RuntimeError("ONNX セッションがロードされていません。")
        session = self.components["session"]
        input_shape = session.get_inputs()[0].shape

        target_size: tuple[int, int] | None = None
        is_nchw = False
        if len(input_shape) == 4:
            if (
                isinstance(input_shape[1], int)
                and input_shape[1] == 3
                and isinstance(input_shape[2], int)
                and isinstance(input_shape[3], int)
            ):
                target_size = (input_shape[2], input_shape[3])
                is_nchw = True
            elif (
                isinstance(input_shape[3], int)
                and input_shape[3] == 3
                and isinstance(input_shape[1], int)
                and isinstance(input_shape[2], int)
            ):
                target_size = (input_shape[1], input_shape[2])
                is_nchw = False
            else:
                if isinstance(input_shape[1], int) and isinstance(input_shape[2], int):
                    self.logger.warning(
                        f"モデル {self.model_name} の不明な入力形状フォーマット: {input_shape}。ターゲットサイズとしてNHWC (インデックス 1, 2) を想定します。"
                    )
                    target_size = (input_shape[1], input_shape[2])
                    is_nchw = False

        if target_size is None:
            raise ValueError(f"入力形状 {input_shape} から有効なターゲットサイズ (H, W) を決定できません。")

        self.target_size = target_size
        self.is_nchw_expected = is_nchw

        self.logger.debug(
            f"モデル {self.model_name} の入力形状: {input_shape}, ターゲットサイズ: {self.target_size}, NCHW形式: {self.is_nchw_expected}"
        )

    def _preprocess_images(self, images: list[Image.Image]) -> list[np.ndarray[Any, np.dtype[np.float32]]]:
        """画像バッチを前処理します。各画像を個別に処理して結果をリストで返します。"""
        if self.target_size is None:
            raise ValueError(f"モデル {self.model_name} の target_size が設定されていません。")

        results = []
        for image in images:
            if image.mode == "RGBA":
                canvas = Image.new("RGB", image.size, (255, 255, 255))
                canvas.paste(image, mask=image.split()[3])
                img_rgb = canvas
            elif image.mode != "RGB":
                img_rgb = image.convert("RGB")
            else:
                img_rgb = image

            width, height = img_rgb.size
            max_dim = max(width, height)
            pad_width = (max_dim - width) // 2
            pad_height = (max_dim - height) // 2
            padded = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            padded.paste(img_rgb, (pad_width, pad_height))

            resized = padded.resize(self.target_size, Image.Resampling.LANCZOS)
            img_array = np.array(resized, dtype=np.float32)[:, :, ::-1]

            if self.is_nchw_expected:
                input_data = np.transpose(img_array, (2, 0, 1))
            else:
                input_data = img_array

            input_data = np.expand_dims(input_data, axis=0)
            results.append(input_data.astype(np.float32))
        return results

    def _run_inference(
        self, processed: list[np.ndarray[Any, np.dtype[Any]]]
    ) -> list[np.ndarray[Any, np.dtype[Any]]]:
        """バッチの各画像に対してONNX推論を実行します。"""
        if "session" not in self.components or self.components["session"] is None:
            raise RuntimeError("ONNX セッションがロードされていません。")
        session = self.components["session"]
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        results = []
        for input_data in processed:
            try:
                raw_output = session.run([output_name], {input_name: input_data})
                results.append(raw_output[0])
            except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
                if "Failed to allocate memory" in str(e) or "out of memory" in str(e).lower():
                    error_message = f"ONNX Runtime メモリ不足: モデル {self.model_name} の推論中"
                    logger.error(error_message)
                    logger.error(f"元のONNX Runtimeエラー: {e}")
                    raise OutOfMemoryError(error_message) from e
                else:
                    logger.exception(f"ONNX Runtime エラー: モデル {self.model_name} の推論中: {e}")
                    raise
            except Exception as e:
                logger.exception(f"予期せぬエラー: モデル {self.model_name} のONNX推論中: {e}")
                raise
        return results

    def _format_predictions(self, raw_outputs: list[np.ndarray[Any, np.dtype[Any]]]) -> list[Any]:
        """バッチ出力結果のナマの値をカテゴリ別にフォーマットします。"""
        result_list = []
        for raw_output in raw_outputs:
            formatted = self._format_predictions_single(raw_output)
            result_list.append(formatted)
        return result_list

    def _generate_tags(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """フォーマットされた単一出力からタグリストを生成します (ONNX/TF タガー用)。"""
        return self._generate_tags_single(formatted_output)
