# 新しいモデルの追加方法 (簡略版)

このガイドでは、`image-annotator-lib`に新しい画像アノテーションモデルを追加する手順を簡潔に説明します。

## 1. 適切な基底クラスを選択

モデルが使用するフレームワークに基づいて、継承するクラスを決定します。

- ONNX ベースのモデル → `ONNXBaseAnnotator`
- Transformers (Hugging Face) ベースのモデル → `TransformersBaseAnnotator`
- TensorFlow ベースのモデル → `TensorflowBaseAnnotator`
- CLIP ベースのモデル → `ClipBaseAnnotator`
- Pipelines (複合モデル) → `PipelineBaseAnnotator`

## 2. モデルクラスを作成

`src/image_annotator_lib/models/` ディレクトリに新しい Python クラスを作成します。

```python
# 例: src/image_annotator_lib/models/tagger_onnx.py に追加
from ..core.base import ONNXBaseAnnotator
from PIL import Image
from typing import Any, List

class MyNewONNXTagger(ONNXBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # モデル固有の初期化処理

    # 必要に応じて以下のメソッドをオーバーライド
    def _preprocess_images(self, images: List[Image.Image]) -> Any:
        # 画像の前処理 (必要な場合のみオーバーライド)
        return processed_data

    def _format_predictions(self, raw_outputs: Any) -> List[Any]:
        # モデル出力の整形
        return formatted_results

    def _generate_tags(self, formatted_output: Any) -> List[str]:
        # タグ生成ロジック
        return tags_list
```

## 3. 設定ファイルに追加

`config/annotator_config.toml` にモデルエントリを追加します。

```toml
[my-new-model-v1]  # モデルの一意な識別名
class = "MyNewONNXTagger"  # クラス名
model_path = "path/to/model.onnx"  # モデルへのパス/URL
estimated_size_gb = 0.5  # おおよそのサイズ (GB)
# device = "cuda"  # 必要に応じて指定
```

## 4. 動作確認

ライブラリを使用して新しいモデルが正しく動作するか確認します。

```python
from image_annotator_lib import annotate, list_available_annotators
from PIL import Image

# 利用可能なモデルを確認
models = list_available_annotators()
print(models)  # 'my-new-model-v1' が含まれているか確認

# テスト実行
img = Image.open("test.jpg")
results = annotate([img], ["my-new-model-v1"])
print(results)
```

## 注意点

- **クラス名の命名規則**: モデルの種類や特性を反映した名前をつけてください。
- **継承関係の遵守**: 適切な基底クラスを継承し、必要なメソッドのみをオーバーライドしてください。
- **共通処理の再利用**: `predict`メソッドなど共通処理はオーバーライドせず、基底クラスの実装を使用してください。
- **エラーハンドリング**: 内部処理でのエラーは適切に捕捉し、意味のあるエラーメッセージを提供してください。

詳細なガイドラインや実装例については、[完全版の新しいモデル追加ガイド](./add_new_model.md)を参照してください。
