# ハウツーガイド: 新しいモデルの追加方法

このガイドでは、`image-annotator-lib` に新しい画像アノテーションモデル (Tagger や Scorer) を追加する手順を説明します。

## 1. モデルクラスの実装

新しいモデルに対応する Python クラスを作成します。このクラスは、ライブラリのクラス階層に従う必要があります。

**クラス階層:**

1.  **`BaseAnnotator` (`core/base.py`)**: 全てのアノテーターの抽象基底クラス。
2.  **フレームワーク別基底クラス (`core/base.py`)**: 特定の ML フレームワーク (ONNX, Transformers, TensorFlow, CLIP など) に共通する処理を実装するクラス。
3.  **具象モデルクラス (`models/`)**: フレームワーク別基底クラスを継承し、個々のモデルに固有の処理のみを実装するクラス。

**実装手順:**

1.  **適切なフレームワーク別基底クラスを選択**: モデルが使用するフレームワークに基づいて選択します。

    - ONNX ベースのモデル → `ONNXBaseAnnotator`
    - Transformers ベースのモデル → `TransformersBaseAnnotator`
    - TensorFlow ベースのモデル → `TensorflowBaseAnnotator`
    - CLIP ベースのモデル → `ClipBaseAnnotator`

2.  **具象モデルクラスを作成**: `src/image_annotator_lib/models/` ディレクトリ内に新しい Python ファイルを作成します。

3.  **クラス定義**:
    - 選択したフレームワーク別基底クラスを継承します。
    - `__init__` メソッドを実装します。
      - `super().__init__(model_name)` を呼び出して親クラスを初期化します。
      - モデル固有の初期化処理 (例: 特定の閾値の設定) を追加します。
    - **必要な抽象メソッドをオーバーライド**: 最低限、以下のメソッドの実装またはオーバーライドが必要になることが多いです。
      - `_preprocess_images(self, images: list[Image.Image]) -> Any`: 画像リストをモデルが受け付ける形式に前処理します。
      - `_run_inference(self, processed: Any) -> Any`: 前処理済みデータで推論を実行します。
      - `_format_predictions(self, raw_outputs: Any) -> list[Any]`: モデルの生出力を整形済みリストに変換します。
      - `_generate_tags(self, formatted_output: Any) -> list[str]`: 整形済み出力から最終的なタグリスト (`list[str]`) を生成します。

**実装例 (ONNX Tagger の場合):**

```python
# src/image_annotator_lib/models/tagger_onnx.py

from ..core.base import ONNXBaseAnnotator
from PIL import Image
from typing import Any, List

class MyNewONNXTagger(ONNXBaseAnnotator):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        # モデル固有の初期化

    def _generate_tags(self, formatted_output: Any) -> List[str]:
        # formatted_output は _format_predictions の結果
        final_tags = []
        # カスタムタグ生成ロジック
        return final_tags
```

## 2. 設定ファイルへの追記

新しいモデルを利用可能にするために、設定ファイル (`config/annotator_config.toml`) にモデルのエントリを追加します。

**設定項目:**

- **セクション名 (`[model_unique_name]`)**: ライブラリ内でモデルを一意に識別するための名前。
- `class` (必須): 実装した具象モデルクラスの名前 (文字列)。
- `model_path` (必須): モデルファイルまたはリポジトリへのパスまたは URL。
- `estimated_size_gb` (推奨): モデルのおおよそのサイズ (GB)。
- `device` (任意): モデルを実行するデバイス (`"cuda"`, `"cpu"` など)。

**追記例:**

```toml
[my-new-onnx-tagger-v1]
class = "MyNewONNXTagger"
model_path = "path/to/your/model.onnx"
estimated_size_gb = 0.5
# device = "cuda" # 必要に応じて指定
```

## 3. 動作確認

上記の手順が完了したら、ライブラリを使用して新しいモデルが動作するか確認します。

```python
from image_annotator_lib import annotate, list_available_annotators
from PIL import Image

# 利用可能なモデル一覧を確認
print(list_available_annotators())  # 'my-new-onnx-tagger-v1' が含まれているか確認

# テスト実行
img = Image.open("path/to/test/image.jpg")
results = annotate([img], ["my-new-onnx-tagger-v1"]) # 新しいモデル名を指定
print(results)
```

これで、新しいモデルの追加は完了です。必要に応じて、テストコードや他のドキュメント (`REFERENCE/models.md` など) も更新してください。
