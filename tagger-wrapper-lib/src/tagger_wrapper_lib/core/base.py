import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, TypedDict

import numpy as np
import onnxruntime as ort
import polars as pl
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoProcessor

# OutOfMemoryError は model_factory からも送出されるためインポート
from ..exceptions.model_errors import OutOfMemoryError
from .model_factory import ModelLoad
from .utils import load_model_config

logger = logging.getLogger(__name__)


# model_factory によって生成されるモデルのコンポーネントを表す型定義
class ModelComponents(TypedDict):
    # TransformerModel
    model: Optional[torch.nn.Module]
    processor: Optional[AutoProcessor]

    # ONNXModel
    session: Optional[ort.InferenceSession]
    csv_path: Optional[str]


class TagConfidence(TypedDict):
    """タグの信頼度情報を表す型定義"""

    confidence: float  # 信頼度
    source: str  # タグの情報元（例: deepdanbooru）


class BaseTagger(ABC):
    # チャンクサイズのデフォルト値をクラス変数として定義
    # TODO; デフォルトのチャンクサイズをスペックに応じて動的に決める処理は気が向いたら
    # スペックを参照する処理はmodel_factory にあるのでそれらをまとめて別のモジュールに定義するかも
    DEFAULT_CHUNK_SIZE = 8

    def __init__(self, model_name: str):
        """BaseTagger を初期化します。

        Args:
            model_name (str): モデルの名前。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = load_model_config()[model_name]

        self.model_path = self.config["model_path"]
        self.device = self.config.get("device", "cuda")
        # 設定ファイルからチャンクサイズを読み込む (なければデフォルト値を使用)
        self.chunk_size = self.config.get("chunk_size", self.DEFAULT_CHUNK_SIZE)
        if not isinstance(self.chunk_size, int) or self.chunk_size <= 0:
            # logger.warning(f"設定された chunk_size '{self.chunk_size}' は無効です。デフォルト値 {self.DEFAULT_CHUNK_SIZE} を使用します。") # ログは任意
            self.chunk_size = self.DEFAULT_CHUNK_SIZE

        self.components: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def __enter__(self) -> "BaseTagger":
        pass

    @abstractmethod
    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        pass

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像リストからタグをチャンクに分割してバッチ予測します。"""
        if not images:
            return []

        all_results: list[dict[str, Any]] = []
        num_images = len(images)
        chunk_size = self.chunk_size  # インスタンス変数から取得

        self.logger.info(
            f"モデル '{self.model_name}' で {num_images} 枚の画像をチャンクサイズ {chunk_size} で処理します。"
        )

        # 画像リストをチャンクに分割してループ処理
        for i in range(0, num_images, chunk_size):
            chunk_images = images[i : i + chunk_size]
            current_chunk_size = len(chunk_images)  # 現在のチャンクの実際のサイズ
            self.logger.debug(
                f"チャンク {i // chunk_size + 1}/{(num_images + chunk_size - 1) // chunk_size} (サイズ: {current_chunk_size}) を処理中..."
            )

            try:
                # --- チャンク単位でバッチ処理 ---
                # 1. 前処理 (サブクラスのバッチ対応メソッドを呼び出す)
                processed_batch = self._preprocess_image(chunk_images)

                # 2. 推論 (サブクラスのバッチ対応メソッドを呼び出す)
                raw_outputs = self._run_inference(processed_batch)

                # 3. フォーマット (サブクラスのバッチ対応メソッドを呼び出す)
                formatted_outputs = self._format_predictions(raw_outputs)

                # 4. タグ生成 (サブクラスのバッチ対応メソッドを呼び出す)
                annotation_lists = self._generate_tags(formatted_outputs)

                # 5. 結果を組み立てて all_results に追加
                if isinstance(annotation_lists, list) and len(annotation_lists) == current_chunk_size:
                    # formatted_outputs の型と長さをチェック
                    if isinstance(formatted_outputs, list) and len(formatted_outputs) == current_chunk_size:
                        # ONNX (list[dict]) or Transformer (list[str])
                        if all(isinstance(item, dict) for item in formatted_outputs) or all(
                            isinstance(item, str) for item in formatted_outputs
                        ):
                            for j in range(current_chunk_size):
                                # _generate_tagsの結果は2次元配列なので、インデックスjの要素を取得
                                tags = annotation_lists[j]
                                all_results.append(self._generate_result(formatted_outputs[j], tags))
                        else:
                            self.logger.error(
                                f"チャンク {i // chunk_size + 1}: 予期しないフォーマット済み出力タイプ: {type(formatted_outputs[0]) if formatted_outputs else 'N/A'}"
                            )
                    else:
                        self.logger.error(
                            f"チャンク {i // chunk_size + 1}: フォーマット済み出力の長さが不正です。期待: {current_chunk_size}, 実際: {len(formatted_outputs)}"
                        )

                else:
                    self.logger.error(
                        f"チャンク {i // chunk_size + 1}: タグリストの形式または長さが不正です。期待: {current_chunk_size}, 実際: {len(annotation_lists) if isinstance(annotation_lists, list) else 'N/A'}"
                    )

            except OutOfMemoryError as e:
                self.logger.error(
                    f"チャンク {i // chunk_size + 1} (サイズ: {current_chunk_size}) の処理中にメモリ不足エラーが発生: {e}"
                )
                self.logger.error(
                    "メモリが不足しています。設定ファイルで chunk_size を小さくすることを検討してください。"
                )
                raise OutOfMemoryError(
                    f"チャンク処理中にメモリ不足 (チャンクサイズ: {current_chunk_size})"
                ) from e
            except ValueError as e:  # preprocess などで発生する可能性のある他のエラー
                self.logger.error(f"チャンク {i // chunk_size + 1} の処理中にエラーが発生: {e}")
                raise
            except Exception as e:
                self.logger.exception(f"チャンク {i // chunk_size + 1} の処理中に予期せぬエラーが発生: {e}")
                raise

        self.logger.info(
            f"モデル '{self.model_name}' の全チャンク処理が完了しました。合計 {len(all_results)} 件の結果を生成しました。"
        )
        return all_results

    @abstractmethod
    def _preprocess_image(self, images: list[Image.Image]) -> list[Any]:
        """画像バッチを前処理します。

        Args:
            images: 処理する画像のリスト

        Returns:
            list[Any]: 前処理された画像データのリスト
        """
        pass

    @abstractmethod
    def _run_inference(self, processed_batch: Any) -> Any:
        """モデル推論をバッチで実行します。"""
        pass

    @abstractmethod
    def _format_predictions(self, raw_outputs: Any) -> list[dict[str, dict[str, float]]] | list[str]:
        """モデルのバッチ生出力をフォーマットします。"""
        pass

    @abstractmethod
    def _generate_tags(self, formatted_outputs: Any) -> list[list[str]]:
        """フォーマットされたバッチ出力からタグリストの2次元配列を生成します。
        各画像に対するタグのリストを含むリストを返します。

        Returns:
            list[list[str]]: 各画像のタグリストを含む2次元配列
        """
        pass

    def _generate_result(self, model_output: Any, annotation_list: list[str]) -> dict[str, Any]:
        """標準化された結果の辞書を生成します。

        Args:
            model_name (str): モデルの名前。
            model_output: モデルの出力。
            annotation_list (list[str]): 生成されたタグのリスト。

        Returns:
            dict: モデル出力、モデル名、タグリストを含む辞書。
        """
        return {
            "model_name": self.model_name,
            "model_output": model_output,
            "annotation": annotation_list,
        }


class TransformersModel(BaseTagger):
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

    def __enter__(self) -> "TransformersModel":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        メモリ不足エラーをハンドリングし、VRAM使用量をログに出力
        """
        try:
            # --- モデルロード処理 ---
            logger.info(f"モデルコンポーネントのロード試行: {self.model_name} をデバイス {self.device} へ")
            loaded_model = ModelLoad.load_transformer_components(
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

        except OutOfMemoryError as e:  # ModelLoad から送出されたエラーをキャッチ
            try:
                # 可能であればメモリ状況を出力
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    logger.error(torch.cuda.memory_summary(device=self.device))
            except Exception as mem_e:
                logger.error(f"CUDAメモリサマリーの取得に失敗: {mem_e}")
            # エラーを再送出
            raise e
        except Exception as e:
            # その他のロード/復元時エラー
            logger.exception(f"モデル {self.model_name} のロード/復元に失敗: {e}")
            raise

        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

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

    def _run_inference(self, processed_images: list[dict[str, Any]]) -> list[torch.Tensor]:
        """モデル推論を実行します。バッチ内の各画像を個別に処理します。
        Args:
            processed_images (list[dict[str, Any]]): モデルへの入力データリスト

        Returns:
            list[torch.Tensor]: モデルからの出力リスト
        """
        results = []
        try:
            for processed_image in processed_images:
                model_out: torch.Tensor = self.components["model"].generate(
                    **processed_image, max_length=self.max_length
                )
                self.logger.debug(f"推論結果のデバイス: {model_out.device}, 形状: {model_out.shape}")
                results.append(model_out)
            return results
        except torch.OutOfMemoryError as e:
            # 推論中のメモリ不足エラーログ
            error_message = f"CUDAメモリ不足: モデル '{self.model_name}' の推論実行中"
            logger.error(error_message)
            logger.error(f"元のPyTorchエラー: {e}")
            try:
                if self.device.startswith("cuda") and torch.cuda.is_available():
                    logger.error(torch.cuda.memory_summary(device=self.device))
            except Exception as mem_e:
                logger.error(f"CUDAメモリサマリーの取得に失敗: {mem_e}")
            raise OutOfMemoryError(error_message) from e
        except Exception as e:
            logger.exception(f"モデル '{self.model_name}' の推論実行中にエラーが発生: {e}")
            raise

    def _format_predictions(self, token_ids_list: list[torch.Tensor]) -> list[str]:
        """モデルの出力をデコードしてテキストにします。

        Args:
            token_ids_list: 複数の出力テンソルのリスト

        Returns:
            list[str]: デコードされたテキストのリスト
        """
        all_results = []
        for token_ids in token_ids_list:
            annotations: list[str] = self.components["processor"].batch_decode(
                token_ids, skip_special_tokens=True
            )
            # batch_decodeは1つのテンソルから複数の結果を返す可能性があるため、すべて追加
            all_results.extend(annotations)
        return all_results

    def _generate_tags(self, formatted_output: list[str]) -> list[list[str]]:
        """
        キャプションのリストを2次元リストに変換します。
        各キャプションを単一要素のリストとして返します。
        """
        # 各キャプションを単一要素の内部リストに変換
        return [[caption] for caption in formatted_output]


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
        self.labels: list[str] = []
        self.rating_indexes: list[int] = []
        self.general_indexes: list[int] = []
        self.character_indexes: list[int] = []
        self.target_size: Optional[tuple[int, int]] = None
        self.is_nchw_expected = False

    def __enter__(self) -> "ONNXModel":
        """
        ModelLoad を使用して ONNX モデルコンポーネントをロードします。
        """
        try:
            self.components = ModelLoad.load_onnx_components(
                self.model_name,
                self.model_path,
                self.device,
            )
            self._load_labels()
            # モデル情報を一度だけ解析して保存
            self._analyze_model_input_format()

        except OutOfMemoryError as e:
            raise e
        except Exception as e:
            logger.exception(f"ONNXモデル {self.model_name} の準備中にエラーが発生: {e}")
            raise

        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        if hasattr(self, "components"):
            self.components = ModelLoad.release_model_components(self.model_name, self.components)

    def _load_labels(self) -> None:
        """ラベル情報をロードし、カテゴリごとのインデックスを設定します。"""
        # ラベルファイルをpolarsで読み込み
        tags_df = pl.read_csv(self.components["csv_path"])

        # ラベル名を取得
        self.labels = tags_df["name"].to_list()

        categories = tags_df["category"].to_list()
        self.rating_indexes = [i for i, cat in enumerate(categories) if cat == 9]
        self.general_indexes = [i for i, cat in enumerate(categories) if cat == 0]
        self.character_indexes = [i for i, cat in enumerate(categories) if cat == 4]

    def _analyze_model_input_format(self) -> None:
        """モデル入力形式を分析し、ターゲットサイズと次元形式を判定・保存する"""
        input_shape = self.components["session"].get_inputs()[0].shape

        # target_sizeの判定ロジック
        target_size: Optional[list[int] | tuple[int, ...]] = None
        if len(input_shape) == 4:
            try:
                channel_dim = input_shape[1]
                height_dim = input_shape[2]
                width_dim = input_shape[3]
                if (
                    isinstance(channel_dim, int)
                    and channel_dim == 3
                    and isinstance(height_dim, int)
                    and isinstance(width_dim, int)
                ):  # NCHW?
                    target_size = [height_dim, width_dim]
                    self.is_nchw_expected = True
                else:  # NHWC? または不明
                    channel_dim_last = input_shape[3]
                    height_dim_first = input_shape[1]
                    width_dim_second = input_shape[2]
                    if (
                        isinstance(channel_dim_last, int)
                        and channel_dim_last == 3
                        and isinstance(height_dim_first, int)
                        and isinstance(width_dim_second, int)
                    ):
                        target_size = [height_dim_first, width_dim_second]
                        self.is_nchw_expected = False
                    else:
                        if isinstance(height_dim_first, int) and isinstance(width_dim_second, int):
                            self.logger.warning(
                                f"モデル {self.model_name} の不明な入力形状フォーマット: {input_shape}。ターゲットサイズとしてNHWC (インデックス 1, 2) を想定します。"
                            )
                            target_size = [height_dim_first, width_dim_second]
                            self.is_nchw_expected = False
            except IndexError:
                pass

        if target_size is None:
            raise ValueError(f"入力形状 {input_shape} から有効なターゲットサイズ (H, W) を決定できません。")

        if not (
            isinstance(target_size, (list, tuple))
            and len(target_size) == 2
            and all(isinstance(dim, int) for dim in target_size)
        ):
            raise ValueError(
                f"モデル {self.model_name} の入力形状 {input_shape} から有効なターゲットサイズを決定できませんでした。"
            )

        self.target_size = (int(target_size[0]), int(target_size[1]))

        self.logger.debug(
            f"モデル {self.model_name} の入力形状: {input_shape}, ターゲットサイズ: {self.target_size}, NCHW形式: {self.is_nchw_expected}"
        )

    def _preprocess_image(self, images: list[Image.Image]) -> list[np.ndarray[Any, np.dtype[np.float32]]]:
        """画像バッチを前処理します。各画像を個別に処理して結果をリストで返します。"""
        results = []
        for image in images:
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

            # target_sizeがNoneでないことを確認
            if self.target_size is None:
                raise ValueError(f"モデル {self.model_name} のtarget_sizeが設定されていません。")

            # 事前計算済みのtarget_sizeを使用
            resized = padded.resize(self.target_size, Image.Resampling.LANCZOS)

            # BGRに変換してバッチ次元追加
            img_array = np.array(resized, dtype=np.float32)[:, :, ::-1]
            input_data = np.expand_dims(img_array, axis=0)

            # 事前計算済みのis_nchw_expectedを使用
            if self.is_nchw_expected:
                input_data = np.transpose(input_data, (0, 3, 1, 2))

            results.append(input_data.astype(np.float32))
        return results

    def _run_inference(
        self, processed_images: list[np.ndarray[Any, np.dtype[Any]]]
    ) -> list[np.ndarray[Any, np.dtype[Any]]]:
        """バッチの各画像に対してONNX推論を実行します。"""
        results = []
        for input_data in processed_images:
            try:
                input_name = self.components["session"].get_inputs()[0].name
                label_name = self.components["session"].get_outputs()[0].name
                raw_output = self.components["session"].run([label_name], {input_name: input_data})
                results.append(raw_output[0])
            except ort.capi.onnxruntime_pybind11_state.RuntimeException as e:
                if "Failed to allocate memory" in str(e):
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

    def _format_predictions(
        self, raw_outputs: list[np.ndarray[Any, np.dtype[Any]]]
    ) -> list[dict[str, dict[str, float]]]:
        """バッチ出力をフォーマットします。"""
        result_list = []
        for raw_output in raw_outputs:
            # 各出力に対して処理を適用
            ratings = self._extract_ratings(raw_output)
            general_tags = self._extract_general_tags(raw_output)
            character_tags = self._extract_character_tags(raw_output)

            result_list.append(
                {
                    "ratings": ratings,
                    "general": general_tags,
                    "character": character_tags,
                }
            )
        return result_list

    def _generate_tags(self, formatted_outputs: list[dict[str, dict[str, float]]]) -> list[list[str]]:
        """バッチ出力からタグリストを生成します。"""
        all_tags_list = []

        for formatted_output in formatted_outputs:
            # 各出力に対してタグを生成
            # 以下は既存の処理を各出力に適用
            general_tags = formatted_output["general"]
            general_probs = np.array(list(general_tags.values()))
            general_threshold = self._calculate_mcut_threshold(general_probs)
            general_threshold = max(0.35, general_threshold)

            character_tags = formatted_output["character"]
            character_probs = np.array(list(character_tags.values()))
            character_threshold = self._calculate_mcut_threshold(character_probs)
            character_threshold = max(0.85, character_threshold)

            selected_general = [tag for tag, prob in general_tags.items() if prob > general_threshold]
            selected_character = [tag for tag, prob in character_tags.items() if prob > character_threshold]

            all_selected_tags = selected_general + selected_character
            escaped_tags = [tag.replace("(", r"\(").replace(")", r"\)") for tag in all_selected_tags]

            all_tags_list.append(escaped_tags)

        return all_tags_list

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


class TensorflowModel(BaseTagger, ABC):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # Tensorflow共通の設定
        self.tf_config = tf.compat.v1.ConfigProto()
        self.tf_config.gpu_options.allow_growth = True
        self.model_format = "h5"

    def __enter__(self) -> "TensorflowModel":
        """共通のTensorflowモデルローディング処理"""
        try:
            # 直接ModelLoadを使用
            components = ModelLoad.load_tensorflow_components(
                self.model_name,
                self.model_path,
                self.device,
                self.model_format,
            )
            if components:
                self.components = components
                self._load_tags()
                self.logger.info(f"モデル '{self.model_name}' を正常にロードしました")
        except Exception as e:
            self.logger.error(f"モデル '{self.model_name}' のロードに失敗: {e}")
            raise
        return self

    def __exit__(
        self,
        exception_type: Optional[type[BaseException]],
        exception_value: Optional[BaseException],
        traceback: Any,
    ) -> None:
        if hasattr(self, "components"):
            self.components = ModelLoad.release_model_components(self.model_name, self.components)

    def _load_tags(self) -> None:
        pass

    def _load_tag_file(self, tags_path: Path) -> list[str]:
        """タグ情報をロードし"""
        with open(tags_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    def _run_inference(self, processed: list[np.ndarray]) -> tf.Tensor:
        """バッチ処理のための共通推論処理"""
        if all(processed_image.ndim == 3 for processed_image in processed):
            processed_batch = np.stack(processed)
        else:
            raise ValueError(f"予期しない入力形式: {[image.shape for image in processed]}")

        # サブクラスでオーバーライド可能
        return self._tf_model_predict(processed_batch)

    def _tf_model_predict(self, batch_input: np.ndarray) -> tf.Tensor:
        """
        Tensorflowモデルの予測処理
        """
        if "model" not in self.components:
            raise ValueError(f"モデル '{self.model_name}' のcomponentsに'model'が見つかりません")
        return self.components["model"](batch_input)
