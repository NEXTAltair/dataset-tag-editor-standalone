"""TensorFlow を使用する Tagger モデルの実装。"""

import logging
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from ..core.base import TensorflowBaseAnnotator
from ..core.config import config_registry

logger = logging.getLogger(__name__)


class DeepDanbooruTagger(TensorflowBaseAnnotator):
    """DeepDanbooru モデルを使用した画像タガー。

    TensorFlow SavedModel または H5 ファイル形式をサポートします。
    モデルバージョン (v1, v3, v4) に応じて入力サイズを自動調整します。
    """

    # model_name は BaseAnnotator の __init__ で設定される
    # config から読み込まれる属性
    # tags, tags_general, tags_character は _load_tags で components に設定される

    def __init__(self, model_name: str):  # kwargs は不要
        """DeepDanbooruTagger を初期化します。"""
        super().__init__(model_name=model_name)
        # tag_threshold の設定 (config_registry.get を使用、デフォルト値 0.35)
        self.tag_threshold = config_registry.get(self.model_name, "tag_threshold", 0.35)
        self.logger.info(f"Tag threshold set to: {self.tag_threshold}")
        # model_format は TensorflowBaseAnnotator で処理されるため、ここでは不要
        logger.debug(f"DeepDanbooruTagger '{model_name}' initialized with threshold: {self.tag_threshold}")

    def _load_tags(self) -> None:
        """DeepDanbooru固有のタグファイル (tags.txt, tags-character.txt, tags-general.txt) をロードします。"""
        if "model_dir" not in self.components or not self.components["model_dir"]:
            self.logger.error(
                "モデルディレクトリが components に設定されていません。タグをロードできません。"
            )
            # タグファイルがないと動作しないため、エラーを送出する方が良いかもしれない
            raise FileNotFoundError("モデルディレクトリが見つかりません。")
            # return # ここでは到達不能

        model_dir = self.components["model_dir"]
        try:
            # 基底クラスのヘルパーメソッドを使用
            all_tags = self._load_tag_file(model_dir / "tags.txt")
            tags_character = self._load_tag_file(model_dir / "tags-character.txt")
            tags_general = self._load_tag_file(model_dir / "tags-general.txt")

            if not all_tags:
                raise FileNotFoundError("tags.txt が見つからないか、空です。")

            # 確実にリストを代入する
            self.components["all_tags"] = all_tags
            # None の可能性を排除してからリストを代入
            self.components["tags_character"] = tags_character if tags_character is not None else []
            self.components["tags_general"] = tags_general if tags_general is not None else []

            # components に代入したリストの長さを取得 (None でないことが保証される)
            general_len = len(self.components["tags_general"])
            character_len = len(self.components["tags_character"])
            self.logger.info(
                f"タグファイルをロードしました: total={len(all_tags)}, general={general_len}, character={character_len}"
            )

        except FileNotFoundError as e:
            self.logger.error(f"必須タグファイルが見つかりません: {e}")
            # 必須ファイルがない場合はエラーとして扱う
            raise e
        except Exception as e:
            self.logger.exception(f"タグファイルの読み込み中に予期せぬエラーが発生しました: {e}")
            raise e  # エラーを再送出

    # メソッド名を _preprocess_images に変更、引数型を修正
    def _preprocess_images(self, images: list[Image.Image]) -> np.ndarray[Any, np.dtype[np.float32]]:
        """画像リストを前処理し、単一の NumPy 配列バッチを返します。"""
        processed_images = []
        # モデル名からバージョンを判断してターゲットサイズを決定
        # TODO: 設定ファイルで明示的に指定する方が良いかもしれない
        if "v1-" in self.model_name:
            target_size = (299, 299)
            self.logger.debug("Using target size 299x299 for v1 model.")
        else:
            target_size = (512, 512)  # v3, v4 and potentially others
            self.logger.debug("Using target size 512x512 for v3/v4 model.")

        for image in images:
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_resized = image.resize(target_size, Image.Resampling.LANCZOS)
            img_array = np.asarray(img_resized, dtype=np.float32) / 255.0
            processed_images.append(img_array)

        # リストを NumPy 配列に変換してバッチを作成
        batch_array = np.stack(processed_images, axis=0)
        self.logger.debug(f"Preprocessed batch shape: {batch_array.shape}")
        return batch_array.astype(np.float32)

    # _run_inference は TensorflowBaseAnnotator の実装を使用

    # メソッド名を _format_predictions に変更
    # 戻り値の型を list[dict[str, dict[str, float]]] に変更 (BaseAnnotator._generate_results が解釈できる形式)
    def _format_predictions(self, raw_output: tf.Tensor) -> list[dict[str, dict[str, float]]]:
        """モデルの生出力バッチをフォーマットし、各画像のカテゴリ別タグ辞書のリストを返します。"""
        batch_results = []
        # TensorFlow のテンソルを NumPy 配列に変換 (CPU にコピー)
        try:
            predictions_np = raw_output.numpy()
        except Exception as e:
            self.logger.exception(f"TensorFlow テンソルの NumPy 変換中にエラー: {e}")
            # エラーが発生した場合、空の結果リストを返すか、例外を再送出するか検討
            # BaseAnnotator.predict でエラーハンドリングされるため、例外を送出する
            raise ValueError("TensorFlow テンソルの NumPy 変換に失敗しました。") from e

        batch_size = predictions_np.shape[0]
        self.logger.debug(f"Formatting predictions for batch size: {batch_size}")

        # components からタグリストを取得 (存在チェックと None の場合のデフォルト値設定)
        all_tags = self.components.get("all_tags")
        # _load_tags で None でないことを保証済みなので、デフォルト値は不要だが念のため
        general_tags = self.components.get("tags_general", [])
        character_tags = self.components.get("tags_character", [])

        if not all_tags:
            self.logger.error(
                "タグ候補リスト (all_tags) が components に存在しません。フォーマットできません。"
            )
            # エラーを示す結果を返すのではなく、例外を送出する
            raise ValueError("タグ候補リストが見つかりません。")

        if len(all_tags) != predictions_np.shape[1]:
            self.logger.error(
                f"タグ候補リストの長さ ({len(all_tags)}) とモデル出力次元 ({predictions_np.shape[1]}) が一致しません。"
            )
            raise ValueError("タグ候補リストとモデル出力の次元が一致しません。")

        for i in range(batch_size):
            single_preds = predictions_np[i]  # i番目の画像の予測結果 (NumPy 配列)
            tag_probs = {tag: float(prob) for tag, prob in zip(all_tags, single_preds, strict=True)}

            formatted: dict[str, dict[str, float]] = {"general": {}, "character": {}, "other": {}}
            classified = set()

            # General Tags
            for tag in general_tags:
                if tag in tag_probs:
                    formatted["general"][tag] = float(tag_probs[tag])
                    classified.add(tag)

            # Character Tags
            for tag in character_tags:
                if tag in tag_probs:
                    formatted["character"][tag] = float(tag_probs[tag])
                    classified.add(tag)

            # Other Tags
            for tag, prob in tag_probs.items():
                if tag not in classified:
                    formatted["other"][tag] = float(prob)

            # カテゴリ内で信頼度順にソート
            formatted["general"] = dict(
                sorted(formatted["general"].items(), key=lambda item: item[1], reverse=True)
            )
            formatted["character"] = dict(
                sorted(formatted["character"].items(), key=lambda item: item[1], reverse=True)
            )
            formatted["other"] = dict(
                sorted(formatted["other"].items(), key=lambda item: item[1], reverse=True)
            )

            batch_results.append(formatted)

        return batch_results

    # _generate_annotations_batch メソッドは削除 (BaseAnnotator._generate_results を使用)
    # _generate_results は BaseAnnotator の実装を使用
    # _generate_tags_single は BaseAnnotator の実装を使用 (閾値は self.tag_threshold を参照)


# モデルクラスの登録 (registry.py で自動検出されるため、ここでは不要)
# from ..core.registry import AnnotatorRegistry
# AnnotatorRegistry.register(DeepDanbooruTagger)
