# クイックスタート

このチュートリアルでは、`image-annotator-lib`の基本的な使い方を簡潔に説明します。

## インストール

```bash
# ライブラリのインストール
pip install image-annotator-lib

# 開発版をソースからインストール
pip install -e .
```

## 基本的な使い方

### 1. ライブラリのインポート

```python
from PIL import Image
from image_annotator_lib import annotate, list_available_annotators
```

### 2. 利用可能なモデルの確認

```python
# 設定ファイルに定義されたモデル一覧を取得
available_models = list_available_annotators()
print("Available models:", available_models)
# 出力例: ['wd-v1-4-vit-tagger-v2', 'aesthetic-shadow-v2', 'blip-large-captioning', ...]
```

### 3. 単一画像・単一モデルでの処理

```python
# 画像を読み込み
img = Image.open("path/to/your/image.jpg")

# 単一モデルで処理（モデル名はlist_available_annotatorsの結果から選択）
model_name = "wd-v1-4-vit-tagger-v2"  # 例: WD Tagger
results = annotate([img], [model_name])

# 結果の確認
for phash, model_results in results.items():
    print(f"Image (pHash: {phash}):")
    result = model_results.get(model_name)
    if result and not result.get("error"):
        print(f"Tags: {result['tags']}")
    else:
        print(f"Error: {result.get('error')}")
```

### 4. 複数画像・複数モデルでの処理

```python
# 複数の画像を準備
img1 = Image.open("path/to/image1.jpg")
img2 = Image.open("path/to/image2.png")
images = [img1, img2]

# 複数のモデルを指定
models = ["wd-v1-4-vit-tagger-v2", "aesthetic-shadow-v2"]
results = annotate(images, models)

# 結果の処理
for phash, model_results in results.items():
    print(f"--- Image (pHash: {phash}) ---")

    # WD Taggerの結果
    wd_result = model_results.get("wd-v1-4-vit-tagger-v2")
    if wd_result and not wd_result.get("error"):
        print("Tags from WD Tagger:")
        print(wd_result["tags"][:5])  # 最初の5個のタグを表示

    # Aesthetic Shadowの結果
    aes_result = model_results.get("aesthetic-shadow-v2")
    if aes_result and not aes_result.get("error"):
        print("Aesthetic Score:")
        print(aes_result["formatted_output"])  # スコア値を表示

    print()
```

## 結果の構造

`annotate`関数の戻り値は以下のような構造を持ちます：

```python
{
    "画像1のpHash": {
        "モデル1の名前": {
            "tags": ["タグ1", "タグ2", ...],  # タグのリスト
            "formatted_output": {...},  # モデル固有の詳細出力
            "error": None  # エラーがない場合はNone
        },
        "モデル2の名前": {
            # モデル2の結果
        }
    },
    "画像2のpHash": {
        # 画像2の各モデル結果
    }
}
```

## エラー処理

モデル実行時のエラーは結果辞書の`error`フィールドにキャプチャされます。

```python
# エラーチェックの例
results = annotate([img], ["non-existent-model"])
for phash, model_results in results.items():
    for model_name, result in model_results.items():
        if result.get("error"):
            print(f"Error in {model_name}: {result['error']}")
```

## 次のステップ

- [詳細なチュートリアル](./basic_usage.md) - より詳しい使用方法
- [複数モデルでの評価](./annotate_multiple_models.md) - 複数モデル使用の詳細
- [新しいモデルの追加方法](../HOW_TO_GUIDES/simplified_add_model.md) - カスタムモデルの追加
