"""画像アノテーションライブラリの基底クラスと型定義。

このモジュールは、画像アノテーション(タギング、スコアリングなど)を行う
すべてのモデルクラスの基底となる抽象クラス `BaseAnnotator` と、
関連する型定義(`AnnotationResult`, `ModelComponents`, `TagConfidence`)、
およびフレームワーク固有の基底クラスを提供します。
"""

import asyncio
import json
import logging
import re
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, NoReturn, Self, TypedDict

import numpy as np
import onnxruntime as ort
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoProcessor

# --- ローカルインポート ---
from ..exceptions.errors import (
    ApiAuthenticationError,
    ApiRateLimitError,
    ApiRequestError,
    ApiServerError,
    ApiTimeoutError,
    ConfigurationError,
    InsufficientCreditsError,
    ModelLoadError,
    OutOfMemoryError,
    WebApiError,
)
from .config import config_registry
from .model_factory import ModelLoad
from .utils import setup_logger

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


# Web API Annotator 用の型定義を追加
class FormattedOutput(TypedDict):
    """フォーマット済み出力を格納する辞書型"""

    annotation: dict[str, Any] | None  # {"tags": list[str], "captions": list[str], "score": float}
    error: str | None


# Web API Annotator 用の型定義を追加
class WebApiAnnotationOutput(TypedDict):
    """WebApiBaseAnnotator._format_predictions の戻り値の型定義。

    APIからの生のレスポンスを解析した結果を格納します。

    Attributes:
        annotation: 解析されたアノテーション情報 (タグ、キャプション、スコアなど) を含む辞書。
                    解析に成功した場合に設定され、失敗した場合は None。
        error: 解析中にエラーが発生した場合のエラーメッセージ文字列。
               エラーがない場合は None。
    """

    annotation: dict[str, Any] | None  # {"tags": list[str], "captions": list[str], "score": float}
    error: str | None


# --- 基底クラス ---


class BaseAnnotator(ABC):
    """画像アノテーションモデルの抽象基底クラス。

    すべてのアノテーター (Tagger, Scorer, Captioner など) はこのクラスを継承します。
    共通の初期化処理、コンテキスト管理のインターフェース、予測処理の骨格を提供します。

    Attributes:
        model_name (str): モデル設定ファイル (`models.toml`) 内のモデル名。
        DEFAULT_CHUNK_SIZE (int): `predict` メソッドでのデフォルトのチャンクサイズ。
        logger (logging.Logger): このクラスインスタンス用のロガー。
        model_path (str): モデルファイルまたはディレクトリへのパス。
        device (str): 推論に使用するデバイス ("cuda", "cpu")。
        chunk_size (int): 一度に処理する画像の数 (バッチサイズ)。
        components (dict[str, Any]): ロードされたモデルコンポーネントを保持する辞書。
                                     キーと値の型は `ModelComponents` TypedDict を参照。
    """

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
            self.model_path = config_registry.get(self.model_name, "model_path")
            self.logger.debug(f"モデルパス: {self.model_path}")

            # device と chunk_size (オプション、デフォルト値指定)
            self.device = config_registry.get(self.model_name, "device", "cuda")
            # chunk_size を int にキャストして型を保証
            chunk_size_val = config_registry.get(self.model_name, "chunk_size", 8)
            try:
                self.chunk_size = int(chunk_size_val)
            except (ValueError, TypeError):
                self.logger.warning(
                    f"chunk_size に不正な値 {chunk_size_val} が設定されました。デフォルトの 8 を使用します。"
                )
                self.chunk_size = 8

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
        tags: list[str] | str,
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
        chunk_size = self.chunk_size

        self.logger.info(
            f"モデル '{self.model_name}' で {num_images} 枚の画像をチャンクサイズ {chunk_size} で処理します。"
        )

        # 画像リストをチャンクに分割してループ処理
        for i in range(0, num_images, chunk_size):
            chunk_images = images[i : i + chunk_size]
            chunk_phash_list = phash_list[i : i + chunk_size] if phash_list and i < len(phash_list) else []
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
                    # formatted_output がエラーを含むかチェック (TypedDict を想定)
                    error_in_format = None
                    if isinstance(formatted_output, dict):
                        error_in_format = formatted_output.get("error")

                    # エラーがない場合のみタグを生成
                    tags: list[str] = []
                    if error_in_format is None:
                        try:
                            tags = self._generate_tags(formatted_output)  # この行を try ブロック内に移動
                        except Exception as tag_gen_e:
                            self.logger.error(f"タグ生成中にエラー: {tag_gen_e}")
                            error_in_format = f"タグ生成エラー: {tag_gen_e}"
                    else:
                        # フォーマット段階でエラーがあれば、タグ生成はスキップ
                        self.logger.debug(f"フォーマットエラーのためタグ生成をスキップ: {error_in_format}")

                    # 対応するpHashを取得
                    phash = chunk_phash_list[j] if j < len(chunk_phash_list) else None

                    # 結果を生成 (エラー情報を渡す)
                    result = self._generate_result(
                        phash=phash, tags=tags, formatted_output=formatted_output, error=error_in_format
                    )
                    all_results.append(result)

            except (OutOfMemoryError, MemoryError, OSError) as e:
                error_message = "メモリ不足エラー"
                self.logger.error(f"チャンク {i // chunk_size + 1} の処理中にメモリ不足エラーが発生: {e}")
                for j, _ in enumerate(chunk_images):
                    phash = chunk_phash_list[j] if j < len(chunk_phash_list) else None
                    result = self._generate_result(phash=phash, tags=[], formatted_output=None, error=error_message)
                    all_results.append(result)
                # メモリ不足の場合は後続チャンクの処理を継続するため raise しない
            except Exception as e:
                error_message = str(e)
                self.logger.error(f"チャンク {i // chunk_size + 1} の処理中に予期せぬエラーが発生: {e}")
                for j, _ in enumerate(chunk_images):
                    phash = chunk_phash_list[j] if j < len(chunk_phash_list) else None
                    result = self._generate_result(phash=phash, tags=[], formatted_output=None, error=error_message)
                    all_results.append(result)
                # 予期せぬエラーの場合も後続チャンクの処理を継続するため raise しない (必要に応じて再検討)

        self.logger.debug(
            f"モデル '{self.model_name}' の全チャンク処理が完了しました。合計 {len(all_results)} 件の結果を生成しました。"
        )
        return all_results


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
        self.max_length = config_registry.get(self.model_name, "max_length", 75)
        self.processor_path = config_registry.get(self.model_name, "processor_path")

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
            self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
            logger.debug(f"モデル {self.model_name} の {self.device} への復元成功")
        except (OutOfMemoryError, MemoryError, OSError) as mem_e:
            # メモリ関連エラーはそのまま上位に伝播させる
            raise mem_e
        except Exception as e:
            # メモリ関連以外の予期せぬエラー
            logger.exception(f"モデル {self.model_name} のロード/復元中に予期せぬエラーが発生: {e}")
            # 予期せぬエラーは ModelLoadError でラップして再送出するなどの検討も可能だが、
            # ここでは元の挙動に合わせてそのまま raise する
            raise

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

    def _preprocess_images(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像バッチを前処理します。各画像を個別に処理して結果をリストで返します。"""
        results = []
        for image in images:
            # プロセッサの出力を取得してデバイスに移動
            processed_output = self.components["processor"](images=image, return_tensors="pt").to(self.device)
            self.logger.debug(f"辞書のキー: {processed_output.keys()}")
            results.append(processed_output)
        return results

    def _run_inference(self, processed: list[dict[str, torch.Tensor]]) -> list[torch.Tensor]:
        """前処理済みバッチで推論を実行します (Transformers用)。"""
        if "model" not in self.components or self.components["model"] is None:
            raise RuntimeError("Transformer モデルがロードされていません。")
        model: Any = self.components["model"]
        outputs = []
        # generateメソッドの一般的な引数やモデルのforwardメソッドの引数を想定
        KNOWN_ARGS = {
            "input_ids",
            "pixel_values",
            "attention_mask",
            "token_type_ids",
            "position_ids",
            "labels",
        }

        with torch.no_grad():
            for processed_image in processed:
                # モデルに渡す引数をフィルタリング
                model_kwargs = {k: v for k, v in processed_image.items() if k in KNOWN_ARGS}

                if hasattr(model, "generate"):
                    # generateメソッドにmax_lengthを追加
                    model_kwargs["max_length"] = self.max_length
                    model_out = model.generate(**model_kwargs)
                else:
                    model_out = model(**model_kwargs)
                    if hasattr(model_out, "last_hidden_state"):
                        model_out = model_out.last_hidden_state
                    elif hasattr(model_out, "logits"):
                        model_out = model_out.logits
                outputs.append(model_out)
        return outputs

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
        """キャプション文字列を単一要素のリストに変換します。

        formatted_outputは文字列型であるため、単純にそれを含む
        リストを返します。文字列以外の型の場合は、エラーログを出力して
        空のリストを返します。
        """
        try:
            if isinstance(formatted_output, str):
                return [formatted_output]
        except Exception as e:
            self.logger.exception(f"タグ生成中にエラー発生: {e}")
            return []


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

        # model_format の取得と検証 (config_registry を使用)
        model_format_input = config_registry.get(self.model_name, "model_format", "h5")
        allowed_formats = ("h5", "saved_model", "pb")
        if model_format_input not in allowed_formats:
            raise ValueError(
                f"設定 '{self.model_name}' の 'model_format' が不正です: '{model_format_input}'. "
                f"許可される形式: {allowed_formats}"
            )
        self.model_format = model_format_input

    def __enter__(self) -> "TensorflowBaseAnnotator":
        """TensorFlow モデルコンポーネントをロードします。状態管理は ModelLoad に委譲します。"""
        self.logger.debug(f"Entering context for TensorFlow model '{self.model_name}'")
        try:
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
            self.logger.exception(f"TensorFlow モデル '{self.model_name}' のロード/準備中に予期せぬエラー: {e}")
            self.components = {}
            raise ModelLoadError(f"予期せぬロードエラー: {e}") from e
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """TensorFlow モデルのリソースを解放します。"""
        self.logger.debug(f"Exiting context for TensorFlow model '{self.model_name}' (exception: {exc_type})")
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
            self.logger.error(f"TensorFlow モデル '{self.model_name}' のコンテキスト内で例外発生: {exc_val}")

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

    def _extract_category_tags(self, attr_name: str, tags_with_probs: list[tuple[str, float]]) -> dict[str, float]:
        """カテゴリータグを抽出するヘルパー関数 (TF タガー用)。"""
        category_tags: dict[str, float] = {}
        # サブクラスで定義される属性 (e.g., self.general_indexes) を取得
        indexes = getattr(self, attr_name, [])
        all_tags_list = getattr(self, "all_tags", [])  # all_tags もサブクラスで設定される
        for i in indexes:
            if 0 <= i < len(tags_with_probs):
                tag_name, prob = tags_with_probs[i]
                category_tags[tag_name] = prob
            else:
                self.logger.warning(f"インデックス {i} が範囲外です (タグ総数: {len(all_tags_list)})。")
        return category_tags

    def _format_predictions_single(
        self,
        raw_output: np.ndarray[Any, np.dtype[Any]] | tf.Tensor,  # TFテンソルも受け入れる
    ) -> dict[str, dict[str, float]]:
        """単一の生出力をカテゴリ別にフォーマットします (TF タガー用)。"""
        result: dict[str, dict[str, float]] = {}
        all_tags_list = getattr(self, "all_tags", [])  # サブクラスで設定される all_tags を取得
        if not all_tags_list:
            self.logger.warning("タグ候補リスト (all_tags) がロードされていません。フォーマットできません。")
            return {"error": {}}  # エラーを示す辞書を返す

        # 生出力が NumPy 配列であることを確認し、適切な次元から予測値を取得
        if isinstance(raw_output, tf.Tensor):  # TFテンソルの場合 NumPy に変換
            try:
                predictions = raw_output.numpy().astype(float)
            except Exception as e:
                self.logger.exception(f"TF テンソルの NumPy 変換中にエラー: {e}")
                return {"error": {}}
        elif isinstance(raw_output, np.ndarray):
            predictions = raw_output.astype(float)

        # 予測値の次元をチェック
        if predictions.ndim == 2 and predictions.shape[0] == 1:
            predictions = predictions[0]
        elif predictions.ndim != 1:
            self.logger.error(f"予期しない予測値形状: {predictions.shape}")
            return {"error": {}}

        # タグ数と予測数が一致するか確認
        if len(all_tags_list) != len(predictions):
            self.logger.error(
                f"タグ候補リスト数 ({len(all_tags_list)}) と予測数 ({len(predictions)}) が一致しません。"
            )
            return {"error": {}}

        tags_with_probs = list(zip(all_tags_list, predictions, strict=True))

        # _category_attr_map はサブクラス (e.g., DeepDanbooruTagger) で定義される
        category_map = getattr(self, "_category_attr_map", None)
        if category_map is None:
            self.logger.warning(
                "_category_attr_map がサブクラスで定義されていません。カテゴリ分類なしでフォーマットします。"
            )
            result["general"] = {tag: float(prob) for tag, prob in tags_with_probs}
            return result

        # カテゴリごとにタグを抽出
        for category_key, attr_name in category_map.items():
            category_tags = self._extract_category_tags(attr_name, tags_with_probs)
            if category_tags:
                result[category_key] = category_tags

        # rating キーを ratings にリネーム (後方互換性のため)
        if "rating" in result and "ratings" not in result:
            result["ratings"] = result.pop("rating")

        return result

    def _generate_tags_single(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """フォーマットされた単一出力からタグリストを生成します (TF タガー用)。"""
        tags = []
        if not formatted_output or "error" in formatted_output:
            return []

        # tag_threshold はサブクラスで設定される想定
        threshold = getattr(self, "tag_threshold", 0.35)

        for category, tag_dict in formatted_output.items():
            if category == "error":  # エラーカテゴリはスキップ
                continue
            for tag, confidence in tag_dict.items():
                # 確信度の値が辞書型の場合 (古い形式への対応?)、confidence キーの値を取得
                conf_value = (
                    confidence["confidence"]
                    if isinstance(confidence, dict) and "confidence" in confidence
                    else confidence
                )

                # confidence が数値型であることを確認してから比較
                if isinstance(conf_value, (float)) and conf_value >= threshold:
                    tags.append((tag, float(conf_value)))  # 確信度も float に統一

        # タグ名で重複を除去し、最も高い確信度を採用
        unique_tags: dict[str, float] = {}
        for tag, conf in tags:
            if tag not in unique_tags or conf > unique_tags[tag]:
                unique_tags[tag] = conf

        # 確信度で降順ソートしてタグ名のみを返す
        return [tag for tag, _ in sorted(unique_tags.items(), key=lambda x: x[1], reverse=True)]


class ClipBaseAnnotator(BaseAnnotator):
    """CLIP モデルをベースとする Scorer 用の基底クラス。"""

    def __init__(self, model_name: str, **kwargs: Any):
        super().__init__(model_name=model_name)
        # base_model は必須設定でデフォルト値なし
        self.base_model = config_registry.get(self.model_name, "base_model")  # 型チェック後に代入
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
                activation_type=config_registry.get(self.model_name, "activation_type"),
                final_activation_type=config_registry.get(self.model_name, "final_activation_type"),
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

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """CLIP Scorer モデルをキャッシュします。"""
        self.logger.debug(f"Exiting context for CLIP Scorer model '{self.model_name}' (exception: {exc_type})")
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
        self.batch_size = config_registry.get(self.model_name, "batch_size", 8)
        self.task = config_registry.get(self.model_name, "task", "image-classification")

    def __enter__(self) -> "PipelineBaseAnnotator":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        loaded_components = ModelLoad.load_transformers_pipeline_components(
            self.task,
            self.model_name,
            self.model_path,
            self.device,
            self.batch_size,
        )
        if loaded_components:
            self.components = loaded_components
        self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """Pipeline モデルをキャッシュします。"""
        self.logger.debug(f"Exiting context for Pipeline model '{self.model_name}' (exception: {exc_type})")
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

    def _preprocess_images(self, images: list[Image.Image]) -> list[Image.Image]:
        """Pipeline は PIL Image を直接受け付けるため、前処理は不要。"""
        return images

    def _run_inference(self, processed: list[Image.Image]) -> list[list[dict[str, Any]]]:
        """Pipeline を使用して推論を実行します。"""
        try:
            raw_outputs = self.components["pipeline"](processed)
            return raw_outputs
        except Exception as e:
            logger.exception(f"Pipeline 推論中にエラーが発生: {e}")
            raise

    def _format_predictions(self, raw_outputs: list[list[dict[str, Any]]]) -> Any:
        """
        Pipeline の生出力は人間が読めるので不要
        """
        return raw_outputs


class ONNXBaseAnnotator(BaseAnnotator):
    """ONNX Runtime を使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.all_tags: list[str] = []
        self.target_size: tuple[int, int] | None = None
        self.is_nchw_expected: bool = False

    def __enter__(self) -> Self:
        """
        ModelLoad を使用して ONNX モデルコンポーネントをロードします。
        """
        try:
            self.logger.info(f"Loading/Restoring ONNX components: model='{self.model_path}'")
            self.components = ModelLoad.load_onnx_components(self.model_name, self.model_path, self.device)
            self._load_tags()
            self._analyze_model_input_format()

        except OutOfMemoryError as e:
            raise e
        except Exception as e:
            logger.exception(f"ONNXモデル {self.model_name} の準備中にエラーが発生: {e}")
            raise

        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """ONNX モデルのリソースを解放します。"""
        self.logger.debug(f"Exiting context for ONNX model '{self.model_name}' (exception: {exc_type})")
        if self.components:
            self.components = ModelLoad.release_model_components(self.model_name, self.components)
        if exc_type:
            self.logger.error(f"ONNX モデル '{self.model_name}' のコンテキスト内で例外発生: {exc_val}")

    @abstractmethod
    def _load_tags(self) -> None:
        """タグ情報 (語彙) をロードし、必要に応じてカテゴリインデックスを設定します (サブクラスで実装)。"""
        raise NotImplementedError("ONNX サブクラスは _load_tags を実装する必要があります。")

    def _extract_category_tags(self, attr_name: str, tags_with_probs: list[tuple[str, float]]) -> dict[str, float]:
        """カテゴリータグを抽出するヘルパー関数 (ONNX タガー用)。"""
        category_tags: dict[str, float] = {}
        indexes = getattr(self, attr_name, [])
        all_tags_list = getattr(self, "all_tags", [])
        for i in indexes:
            if 0 <= i < len(tags_with_probs):
                tag_name, prob = tags_with_probs[i]
                category_tags[tag_name] = prob
            else:
                self.logger.warning(f"インデックス {i} が範囲外です (タグ総数: {len(all_tags_list)})。")
        return category_tags

    def _format_predictions_single(self, raw_output: np.ndarray[Any, np.dtype[Any]]) -> dict[str, dict[str, float]]:
        """単一の生出力をカテゴリ別にフォーマットします (ONNX タガー用)。"""
        result: dict[str, dict[str, float]] = {}
        all_tags_list = getattr(self, "all_tags", [])
        if not all_tags_list:
            self.logger.warning("タグ候補リスト (all_tags) がロードされていません。フォーマットできません。")
            return {"error": {}}  # エラーを示す辞書を返す

        # 出力が NumPy 配列であることを確認
        if not isinstance(raw_output, np.ndarray):
            self.logger.error(f"予期しない生出力型: {type(raw_output)}")
            return {"error": {}}

        # 予測値の次元をチェックして調整
        if raw_output.ndim == 2 and raw_output.shape[0] == 1:
            predictions = raw_output[0].astype(float)
        elif raw_output.ndim == 1:
            predictions = raw_output.astype(float)
        else:
            self.logger.error(f"予期しない生出力形状: {raw_output.shape}")
            return {"error": {}}

        # タグ数と予測数が一致するか確認
        if len(all_tags_list) != len(predictions):
            self.logger.error(
                f"タグ候補リスト数 ({len(all_tags_list)}) と予測数 ({len(predictions)}) が一致しません。"
            )
            return {"error": {}}

        tags_with_probs = list(zip(all_tags_list, predictions, strict=True))

        # _category_attr_map はサブクラス (e.g., WDTagger) で定義される
        category_map = getattr(self, "_category_attr_map", None)
        if category_map is None:
            self.logger.warning(
                "_category_attr_map がサブクラスで定義されていません。カテゴリ分類なしでフォーマットします。"
            )
            result["general"] = {tag: float(prob) for tag, prob in tags_with_probs}
            return result

        # カテゴリごとにタグを抽出
        for category_key, attr_name in category_map.items():
            category_tags = self._extract_category_tags(attr_name, tags_with_probs)
            if category_tags:
                result[category_key] = category_tags

        # rating キーを ratings にリネーム (後方互換性のため)
        if "rating" in result and "ratings" not in result:
            result["ratings"] = result.pop("rating")

        return result

    def _generate_tags_single(self, formatted_output: dict[str, dict[str, float]]) -> list[str]:
        """フォーマットされた単一出力からタグリストを生成します (ONNX タガー用)。"""
        tags = []
        if not formatted_output or "error" in formatted_output:
            return []

        # tag_threshold はサブクラスで設定される想定
        threshold = getattr(self, "tag_threshold", 0.35)

        for category, tag_dict in formatted_output.items():
            if category == "error":
                continue
            for tag, confidence in tag_dict.items():
                conf_value = (
                    confidence["confidence"]
                    if isinstance(confidence, dict) and "confidence" in confidence
                    else confidence
                )

                if isinstance(conf_value, (float)) and conf_value >= threshold:
                    tags.append((tag, float(conf_value)))

        unique_tags: dict[str, float] = {}
        for tag, conf in tags:
            if tag not in unique_tags or conf > unique_tags[tag]:
                unique_tags[tag] = conf

        return [tag for tag, _ in sorted(unique_tags.items(), key=lambda x: x[1], reverse=True)]

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

    def _run_inference(self, processed: list[np.ndarray[Any, np.dtype[Any]]]) -> list[np.ndarray[Any, np.dtype[Any]]]:
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


class WebApiBaseAnnotator(BaseAnnotator):
    """Web API を使用するモデル用の基底クラス。"""

    def __init__(self, model_name: str):
        super().__init__(model_name)
        # 設定ファイルから読み込む共通パラメータ
        self.prompt_template = config_registry.get(self.model_name, "prompt_template", "Describe this image.")
        timeout_val = config_registry.get(self.model_name, "timeout", 60)
        try:
            self.timeout = int(timeout_val)
        except (ValueError, TypeError):
            self.logger.warning(f"timeout に不正な値 {timeout_val} が設定されました。デフォルトの 60 を使用します。")
            self.timeout = 60

        # レート制限とリトライの設定
        # retry_count を int にキャスト
        retry_count_val = config_registry.get(self.model_name, "retry_count", 3)
        try:
            self.retry_count = int(retry_count_val)
        except (ValueError, TypeError):
            self.logger.warning(
                f"retry_count に不正な値 {retry_count_val} が設定されました。デフォルトの 3 を使用します。"
            )
            self.retry_count = 3

        # retry_delay を float にキャスト
        retry_delay_val = config_registry.get(self.model_name, "retry_delay", 1.0)
        try:
            self.retry_delay = float(retry_delay_val)
        except (ValueError, TypeError):
            self.logger.warning(
                f"retry_delay に不正な値 {retry_delay_val} が設定されました。デフォルトの 1.0 を使用します。"
            )
            self.retry_delay = 1.0

        self.last_request_time = 0.0
        # min_request_interval を float にキャストして型を保証
        min_interval_val = config_registry.get(self.model_name, "min_request_interval", 1.0)
        try:
            self.min_request_interval = float(min_interval_val)
        except (ValueError, TypeError):
            self.logger.warning(
                f"min_request_interval に不正な値 {min_interval_val} が設定されました。デフォルトの 1.0 を使用します。"
            )
            self.min_request_interval = 1.0

        self.model_name_on_provider: str | None = config_registry.get(self.model_name, "model_name_on_provider")

        self.api_key = self._load_api_key()
        self.client: Any = None  # 追加: クライアントインスタンス用属性

    @abstractmethod
    def __enter__(self) -> Self:
        """サブクラスでAPIクライアントを初期化し、self.clientに設定します。"""
        raise NotImplementedError("Web API サブクラスは __enter__ を実装する必要があります。")

    def __exit__(self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: Any) -> None:
        """APIクライアントのリソースを解放 (Noneを設定) します。"""
        # provider_name 属性が存在するか確認
        provider_name = getattr(self, "provider_name", self.model_name)  # provider_name がなければ model_name を使用
        if self.client:
            self.logger.debug(f"APIクライアントの閉鎖/リリース{provider_name}")
            self.client = None

    @abstractmethod
    def _load_api_key(self) -> str:
        """環境変数から API キーをロードします。"""
        raise NotImplementedError("Web API サブクラスは _load_api_key を実装する必要があります。")

    def _preprocess_images(self, images: list[Image.Image]) -> list[str]:
        """画像リストを Base64 エンコードした文字列のリストに変換する"""
        import base64
        from io import BytesIO

        encoded_images = []
        for image in images:
            buffered = BytesIO()
            # 画像をWEBP形式でメモリに保存
            image.save(buffered, format="WEBP")
            # バイトデータを取得し、Base64エンコードしてUTF-8文字列にデコード
            encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
            encoded_images.append(encoded_image)

        return encoded_images

    @abstractmethod
    def _run_inference(self, processed: list[str]) -> Any:
        """Base64エンコードされた画像文字列リストをAPIに送信して推論結果を取得する"""
        raise NotImplementedError("Web API サブクラスは _run_inference を実装する必要があります。")

    def _wait_for_rate_limit(self) -> None:
        """レート制限に従ってリクエスト間隔を調整する"""
        import time

        elapsed_time = time.time() - self.last_request_time
        wait_time = self.min_request_interval - elapsed_time
        if wait_time > 0:
            self.logger.debug(f"レート制限のため {wait_time:.2f} 秒待機します。")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _handle_api_error(self, e: Exception) -> NoReturn:
        """API エラーを捕捉し、適切なカスタム例外を発生させます。

        Args:
            e: 発生した例外。

        Raises:
            ApiAuthenticationError: API認証に失敗した場合 (401)。
            InsufficientCreditsError: クレジット不足の場合 (402)。
            ApiRateLimitError: APIのレート制限に達した場合 (429)。
            ApiRequestError: リクエストの形式または内容に問題があった場合 (400)。
            ApiServerError: APIサーバーで5xx系のエラーが発生した場合。
            ApiTimeoutError: APIリクエストがタイムアウトした場合。
            WebApiError: その他のAPI関連エラーの場合。
            ConfigurationError: provider_name 属性が設定されていない場合。
        """
        error_message = str(e)
        self.logger.error(f"API エラーが発生しました: {error_message}")
        self.logger.debug(traceback.format_exc())

        # provider_name 属性の存在確認
        if not hasattr(self, "provider_name") or not self.provider_name:
            raise ConfigurationError(
                f"Annotatorクラス ({self.__class__.__name__}) に 'provider_name' 属性が設定されていません。"
            )
        provider_name = self.provider_name

        # HTTPステータスコードに基づくエラーハンドリング
        if hasattr(e, "status_code"):
            status_code = getattr(e, "status_code", 0)
            if status_code == 401:
                raise ApiAuthenticationError(provider_name=provider_name) from e
            elif status_code == 402:
                raise InsufficientCreditsError(provider_name=provider_name) from e
            elif status_code == 429:
                retry_after_str = getattr(e, "retry_after", "60")  # デフォルト60秒
                try:
                    retry_after = int(retry_after_str)
                except ValueError:
                    retry_after = 60  # パース失敗時もデフォルト値
                raise ApiRateLimitError(provider_name=provider_name, retry_after=retry_after) from e
            elif status_code == 400:
                raise ApiRequestError(error_message, provider_name=provider_name) from e
            elif 500 <= status_code < 600:
                raise ApiServerError(error_message, provider_name=provider_name, status_code=status_code) from e

        # タイムアウトエラーの判定を強化
        if isinstance(e, TimeoutError | asyncio.TimeoutError) or "timeout" in error_message.lower():
            raise ApiTimeoutError(provider_name=provider_name) from e

        # 上記のいずれにも当てはまらない場合、汎用のWebApiErrorを送出
        raise WebApiError(f"処理中に予期せぬエラーが発生しました: {error_message}", provider_name=provider_name) from e

    def _parse_common_json_response(self, text_content: str) -> WebApiAnnotationOutput:
        """共通のJSONレスポンス文字列を解析し、WebApiAnnotationOutputを生成するヘルパー。

        Args:
            text_content: APIから返されたテキストコンテンツ。

        Returns:
            解析結果を含むWebApiAnnotationOutput辞書。
            エラーが発生した場合は、errorフィールドにメッセージが含まれる。
        """
        self.logger.debug(f"_parse_common_json_response を開始: text='{text_content[:100]}...'")
        try:
            # JSON文字列を辞書にパース
            data = json.loads(text_content)

            # "Annotation" キー (Gemini) または ルートレベルの辞書 (OpenAI/Anthropic/OpenRouter) を想定
            annotation_data: dict[str, Any] | None = None
            if isinstance(data, dict):
                if "Annotation" in data and isinstance(data["Annotation"], dict):
                    annotation_data = data["Annotation"]
                    self.logger.debug("JSONに 'Annotation' キーが見つかりました。")
                # 'tags', 'caption', 'score' がルートレベルに存在するケースも考慮
                elif any(key in data for key in ("tags", "caption", "score")):
                    annotation_data = data
                    self.logger.debug("JSONのルートレベルに注釈キーが見つかりました。")
                else:
                    self.logger.warning("JSON内に 'Annotation' キーまたは期待されるキーが見つかりません。")
                    return WebApiAnnotationOutput(
                        annotation=None,
                        error="JSON内に期待されるキー (Annotation, tags, caption, score) が見つかりません。",
                    )
            else:
                self.logger.warning(f"JSONデータが予期しない型 ({type(data)}) です。")
                return WebApiAnnotationOutput(annotation=None, error=f"JSONデータが予期しない型 ({type(data)}) です。")

            if annotation_data:
                self.logger.debug(f"JSON解析成功。Annotation: {str(annotation_data)[:100]}...")
                return WebApiAnnotationOutput(annotation=annotation_data, error=None)
            else:
                return WebApiAnnotationOutput(
                    annotation=None, error="解析後、有効なAnnotationデータが見つかりませんでした。"
                )

        except json.JSONDecodeError as json_e:
            error_message = f"JSON解析エラー: {json_e!s}. テキスト内容: '{text_content[:100]}...'"  # 末尾の \" を削除
            self.logger.error(error_message)
            return WebApiAnnotationOutput(annotation=None, error=error_message)
        except Exception as e:
            error_message = f"JSON解析中に予期せぬエラー: {e!s}"  # 末尾の \" を削除
            self.logger.exception(error_message)  # スタックトレースも記録
            return WebApiAnnotationOutput(annotation=None, error=error_message)

    def _extract_tags_from_text(self, text: str) -> list[str]:
        """API レスポンス (テキスト形式) からタグリストを抽出する基本実装。

        JSON形式、またはカンマ区切りのタグリスト形式を試みます。

        Args:
            text: API から返されたテキスト応答。

        Returns:
            抽出されたタグのリスト。見つからない場合は空リスト。
        """
        self.logger.debug("_extract_tags_from_text を開始します。")
        tags: list[str] = []

        # 1. JSON 形式の解析を試みる
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                # "tags" キーが存在し、リストまたはカンマ区切り文字列の場合
                if "tags" in data:
                    tags_data = data["tags"]
                    if isinstance(tags_data, list):
                        tags = [str(tag).strip() for tag in tags_data]  # 文字列に変換
                        self.logger.debug(f"JSONから {len(tags)} 個のタグを抽出しました。")
                        return tags
                    elif isinstance(tags_data, str):
                        tags = [tag.strip() for tag in tags_data.split(",") if tag.strip()]
                        self.logger.debug(f"JSON内のカンマ区切り文字列から {len(tags)} 個のタグを抽出しました。")
                        return tags
                # "Annotation" -> "tags" のネスト構造も考慮 (Geminiの例)
                elif "Annotation" in data and isinstance(data["Annotation"], dict) and "tags" in data["Annotation"]:
                    tags_data = data["Annotation"]["tags"]
                    if isinstance(tags_data, list):
                        tags = [str(tag).strip() for tag in tags_data]
                        self.logger.debug(f"JSON (Annotation->tags) から {len(tags)} 個のタグを抽出しました。")
                        return tags
            # JSONがリスト形式で、要素が文字列の場合
            elif isinstance(data, list) and all(isinstance(item, str) for item in data):
                tags = [item.strip() for item in data if item.strip()]
                self.logger.debug(f"JSONリストから {len(tags)} 個のタグを抽出しました。")
                return tags

        except json.JSONDecodeError:
            self.logger.debug("テキストは有効なJSONではありません。次の抽出方法を試みます。")
        except Exception as e:
            self.logger.warning(f"JSON解析中に予期せぬエラー: {e}。次の抽出方法を試みます。", exc_info=True)

        # 2. カンマ区切りテキスト形式の解析を試みる
        # "tags:" のようなプレフィックスがある場合とない場合の両方を考慮
        # より具体的にタグらしきものを抽出する正規表現
        # 例: tags: tag1, tag2, tag3 / tags: "tag1", "tag2" / tag1, tag2, ...
        patterns = [
            r"tags:?\s*\[?\"?\'?(.*?)\'?\"?\]?$",  # tags: ["tag1", "tag2"] or tags: 'tag1', 'tag2' or tags: tag1, tag2
            r"^\[?\"?\'?(.*?)\'?\"?\]?$",  # ["tag1", "tag2"] or 'tag1', 'tag2' or tag1, tag2 (行頭から)
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                potential_tags_str = match.group(1).strip()
                # クォートや括弧が残っている可能性があるので除去
                potential_tags_str = re.sub(r'^["\'\[\]\s]+|["\'\[\]\s]+$', "", potential_tags_str)
                # カンマで分割
                tags = [tag.strip() for tag in potential_tags_str.split(",") if tag.strip()]
                if tags:
                    self.logger.debug(
                        f"正規表現 ({pattern}) でカンマ区切りテキストから {len(tags)} 個のタグを抽出しました。"
                    )
                    return tags

        self.logger.warning(f"どの形式でもタグを抽出できませんでした。テキスト: {text[:100]}...")
        return []

    # _generate_tags の共通実装を追加
    def _generate_tags(self, formatted_output: WebApiAnnotationOutput) -> list[str]:
        """フォーマット済み出力からタグを生成する (WebApiBaseAnnotator 共通実装)

        Args:
            formatted_output: _format_predictions で生成された WebApiAnnotationOutput 辞書。

        Returns:
            抽出されたタグのリスト。エラー発生時やタグが見つからない場合は空リスト。
        """
        if formatted_output.get("error") or formatted_output.get("annotation") is None:
            return []

        annotation = formatted_output["annotation"]
        if annotation is None:
            return []

        # annotation 辞書から tags を抽出
        tags_data = annotation.get("tags")

        if isinstance(tags_data, list):
            # リスト内の要素が文字列であることを確認して返す
            return [str(tag).strip() for tag in tags_data if isinstance(tag, str)]
        elif isinstance(tags_data, str):
            # 文字列の場合はカンマで分割
            return [tag.strip() for tag in tags_data.split(",") if tag.strip()]
        else:
            self.logger.warning(f"予期しない形式のタグデータ: {type(tags_data)}. タグを抽出できませんでした。")
            return []
