# チュートリアル: 基本的な使い方

このチュートリアルでは、`image-annotator-lib` を使って単一のモデルで画像を評価する基本的な手順を説明します。

## 1. 必要なライブラリのインポート

```python
from PIL import Image
from image_annotator_lib import annotate, list_available_annotators
```

## 2. 利用可能なモデルの確認

```python
available_models = list_available_annotators()
print("Available models:", available_models)
# 出力例: ['wd-v1-4-vit-tagger-v2', 'aesthetic-shadow-v2', 'blip-large-captioning', ...]
```

これにより、設定ファイル (`annotator_config.toml`) で定義され、利用可能なモデル名の一覧が表示されます。

## 3. 画像の準備

```python
try:
    image_path = "path/to/your/image.jpg"
    img = Image.open(image_path)
    images_to_process = [img]
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()
except Exception as e:
    print(f"Error opening image: {e}")
    exit()
```

## 4. モデルの選択

```python
model_name = "wd-v1-4-vit-tagger-v2" # 利用可能なモデル名から選択
models_to_use = [model_name]

# 選択したモデルが利用可能か確認 (任意)
if model_name not in available_models:
    print(f"Warning: Selected model '{model_name}' is not available.")
    exit()
```

## 5. 評価の実行

```python
results = annotate(images_to_process, models_to_use)
```

## 6. 結果の確認

```python
for phash, model_results in results.items():
    print(f"--- Image (pHash: {phash}) ---")
    if not model_results:
        print("  No results for this image.")
        continue

    # 今回はモデルを1つだけ指定したので、その結果を取得
    result_for_model = model_results.get(model_name)

    if result_for_model:
        if result_for_model.get("error"):
            print(f"  Error: {result_for_model['error']}")
        else:
            tags = result_for_model.get('tags', [])
            formatted_output = result_for_model.get('formatted_output')
            print(f"  Tags: {tags}")
            # print(f"  Formatted Output: {formatted_output}") # 必要に応じて確認
    else:
        print(f"  No result found for model: {model_name}")
```

これで、基本的な画像の評価が完了しました。
複数のモデルを同時に使って評価する方法については、[複数モデルでの評価](./annotate_multiple_models.md)を参照してください。
