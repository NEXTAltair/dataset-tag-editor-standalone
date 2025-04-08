from typing import Any

import torch
from PIL import Image

from ..core.base import TransformersBaseAnnotator
from ..exceptions.errors import OutOfMemoryError


class BLIPTagger(TransformersBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)


class BLIP2Tagger(TransformersBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)


class GITTagger(TransformersBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)


class ToriiGateTagger(TransformersBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        self.user_prompt = "Describe the picture in structuted json-like format."
        self.messages = [
            {
                # Important!
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are image captioning expert, creative, unbiased and uncensored. Help user with his task.",
                    }
                ],
            },
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.user_prompt}]},
        ]

    def _preprocess_image(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像バッチを前処理します。チャットテンプレートを適用して入力を準備します。"""
        results = []
        for image in images:
            prompt = self.components["processor"].apply_chat_template(
                self.messages, add_generation_prompt=True
            )
            inputs = self.components["processor"](text=prompt, images=[image], return_tensors="pt")
            # デバッグログ: processor が返すキーを確認
            self.logger.debug(f"Processor output keys for {self.model_name}: {list(inputs.keys())}")
            # デバイスに移動
            processed_output = {k: v.to(self.device) for k, v in inputs.items()}
            results.append(processed_output)
        return results

    def _run_inference(self, processed_images: list[dict[str, Any]]) -> list[torch.Tensor]:
        """モデル推論を実行します。生成されたIDを返します。"""
        results = []
        # generateメソッドの一般的な引数やモデルのforwardメソッドの引数を想定
        KNOWN_ARGS = {"input_ids", "pixel_values", "attention_mask", "token_type_ids", "position_ids", "labels"}
        try:
            for processed_image in processed_images:
                # モデルに渡す引数をフィルタリング
                model_kwargs = {k: v for k, v in processed_image.items() if k in KNOWN_ARGS}
                # max_new_tokens を追加
                model_kwargs["max_new_tokens"] = 500
                model_out = self.components["model"].generate(**model_kwargs)
                self.logger.debug(f"推論結果のデバイス: {model_out.device}, 形状: {model_out.shape}")
                results.append(model_out)
            return results
        except torch.cuda.OutOfMemoryError as e:
            error_message = f"CUDAメモリ不足: モデル '{self.model_name}' の推論実行中"
            self.logger.error(error_message)
            raise OutOfMemoryError(error_message) from e

    def _format_predictions(self, token_ids_list: list[torch.Tensor]) -> list[str]:
        """モデルの出力をデコードしてテキストにします。Assistant部分のみを抽出します。"""
        all_results = []
        for token_ids in token_ids_list:
            # デコード
            generated_text = self.components["processor"].batch_decode(token_ids, skip_special_tokens=True)[
                0
            ]
            # 'Assistant: ' 以降の部分を取得
            if "Assistant: " in generated_text:
                caption = generated_text.split("Assistant: ")[1]
            else:
                caption = generated_text
            all_results.append(caption)
        return all_results

    def _generate_tags(self, formatted_output: str) -> list[str]:
        """単一のキャプション文字列を単一要素のリストに変換します。
        各キャプションを単一要素のリストとして返します。
        ToriiGateの場合、キャプション全体を1つのタグとして扱います。
        """
        # formatted_output は単一の文字列と想定
        return [formatted_output]
