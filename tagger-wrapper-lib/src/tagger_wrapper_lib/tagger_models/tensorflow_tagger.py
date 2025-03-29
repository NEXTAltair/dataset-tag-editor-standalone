from typing import Any

import numpy as np
import tensorflow as tf  # type: ignore
from PIL import Image

from tagger_wrapper_lib.core.base import TagConfidence, TensorflowModel


class DeepDanbooruTagger(TensorflowModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.threshold = 0.7  # DeepDanbooru独自の閾値
        self.model_format = "h5"

    def _load_tags(self) -> None:
        """DeepDanbooru固有のタグ読み込み処理"""
        model_dir = self.components["model_dir"]
        try:
            self.components["tags"] = self._load_tag_file(model_dir / "tags.txt")
            self.components["tags_character"] = self._load_tag_file(model_dir / "tags-character.txt")
            self.components["tags_general"] = self._load_tag_file(model_dir / "tags-general.txt")
        except Exception as e:
            self.logger.error(f"タグファイルの読み込みに失敗しました: {e}")

    def _preprocess_image(self, images: list[Image.Image]) -> list[np.ndarray[Any, np.dtype[np.float32]]]:
        """DeepDanbooru固有の前処理 - モデルバージョンに合わせて入力サイズを調整"""
        results = []
        # モデル名からバージョンを判断
        if "v1-" in self.model_name:
            target_size = (299, 299)  # v1モデル用サイズ
        else:
            target_size = (512, 512)  # v3/v4モデル用サイズ

        for image in images:
            # 変数名を変えて型の混乱を避ける
            img_resized = image.convert("RGB").resize(target_size)
            img_array = np.asarray(img_resized).astype(np.float32) / 255.0
            results.append(img_array)
        return results

    def _format_predictions(self, raw_output: tf.Tensor) -> list[dict[str, dict[str, float]]] | list[str]:
        """DeepDanbooru固有の結果フォーマット処理"""
        # 現在の実装をそのまま使用
        batch_size = raw_output.shape[0]
        results = []

        for i in range(batch_size):
            single_output = raw_output[i : i + 1]
            general = self._format_general_tags(single_output)
            character = self._format_character_tags(single_output)
            other = self._format_all_tags(single_output)

            results.append(
                {
                    "general": dict(
                        sorted(general.items(), key=lambda x: x[1]["confidence"], reverse=True)
                    ),
                    "character": dict(
                        sorted(character.items(), key=lambda x: x[1]["confidence"], reverse=True)
                    ),
                    "other": dict(sorted(other.items(), key=lambda x: x[1]["confidence"], reverse=True)),
                }
            )

        return results

    def _format_general_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        生の出力から一般タグの情報を抽出し、信頼度情報を付与します。

        Args:
            raw_output (tf.Tensor): モデルからの生出力。

        Returns:
            dict[str, TagConfidence]: 一般タグをキーに、信頼度と情報元を値とする辞書。
        """
        all_tags = self.components["tags"]
        tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0], strict=False)}
        return {
            tag: TagConfidence(confidence=float(tag_probs[tag]), source="deepdanbooru_general")
            for tag in self.components["tags_general"]
            if tag in tag_probs
        }

    def _format_character_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        生の出力からキャラクタータグの情報を抽出し、信頼度情報を付与します。

        Args:
            raw_output (tf.Tensor): モデルの生出力。

        Returns:
            dict[str, TagConfidence]: キャラクタータグをキーに、信頼度と情報元を値とする辞書。
        """
        all_tags = self.components["tags"]
        tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0], strict=False)}
        return {
            tag: TagConfidence(confidence=float(tag_probs[tag]), source="deepdanbooru_character")
            for tag in self.components["tags_character"]
            if tag in tag_probs
        }

    def _format_all_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        生の出力から、一般タグ・キャラクタータグ以外のタグ情報を抽出します。

        Args:
            raw_output (tf.Tensor): モデルの生出力。

        Returns:
            dict[str, TagConfidence]: 残りのタグをキーに、信頼度と情報元を値とする辞書。
        """
        all_tags = self.components["tags"]
        tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0], strict=False)}
        classified = set(self.components["tags_general"]) | set(self.components["tags_character"])
        return {
            tag: TagConfidence(confidence=float(prob), source="deepdanbooru")
            for tag, prob in tag_probs.items()
            if tag not in classified
        }

    def _generate_tags(
        self, formatted_outputs: list[dict[str, dict[str, TagConfidence]]]
    ) -> list[list[str]]:
        """DeepDanbooru固有のタグ生成処理"""
        # 現在の実装をそのまま
        results = []

        for formatted_output in formatted_outputs:
            tags = [
                tag
                for cat, tags in formatted_output.items()
                if cat != "other"
                for tag, conf in tags.items()
                if conf["confidence"] >= self.threshold
            ]
            results.append(tags)

        return results
