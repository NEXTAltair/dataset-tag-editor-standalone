# チュートリアル: 複数モデルでの評価

このチュートリアルでは、`image-annotator-lib` を使って複数のモデルで画像を同時に評価する手順を説明します。

## 1. 準備

基本的なライブラリのインポートと画像の準備は [基本的な使い方](./basic_usage.md) と同様です。

```python
from PIL import Image
from image_annotator_lib import annotate, list_available_annotators

# 利用可能なモデルを確認
available_models = list_available_annotators()

# 評価したい画像を準備 (例として2枚)
try:
    image_path1 = "path/to/your/image1.jpg"
    image_path2 = "path/to/your/image2.png"
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    images_to_process = [img1, img2]
except FileNotFoundError as e:
    print(f"Error: Image file not found: {e}")
    exit()
except Exception as e:
    print(f"Error opening image: {e}")
    exit()
```

## 2. 複数モデルの選択

```python
# 例: WD Tagger と Aesthetic Scorer を使用
models_to_use = ["wd-v1-4-vit-tagger-v2", "aesthetic-shadow-v2"]

# 選択したモデルが利用可能か確認 (任意)
unavailable_models = [m for m in models_to_use if m not in available_models]
if unavailable_models:
    print(f"Warning: The following models are not available: {', '.join(unavailable_models)}")
    # 利用可能なモデルのみで続行
    models_to_use = [m for m in models_to_use if m in available_models]
    if not models_to_use:
        print("Error: No available models selected.")
        exit()
```

## 3. 評価の実行

```python
results = annotate(images_to_process, models_to_use)
```

## 4. 結果の確認

複数モデルで評価した場合、`annotate` 関数の戻り値は、**画像の pHash** をキーとする辞書になります。各 pHash の値は、モデル名をキーとする辞書で、そのモデルによる当該画像のアノテーション結果が格納されます。

```python
for phash, model_results in results.items():
    print(f"--- Image (pHash: {phash}) ---")

    # WD Taggerの結果
    wd_result = model_results.get("wd-v1-4-vit-tagger-v2")
    if wd_result and not wd_result.get("error"):
        # タグのリストを取得
        tags = wd_result.get('tags', [])
        print(f"Top 5 Tags from WD Tagger: {tags[:5]}")
    elif wd_result:
        print(f"Error in WD Tagger: {wd_result.get('error')}")

    # Aesthetic Scorerの結果
    aes_result = model_results.get("aesthetic-shadow-v2")
    if aes_result and not aes_result.get("error"):
        # スコア値を取得
        score = aes_result.get('formatted_output')
        print(f"Aesthetic Score: {score}")
    elif aes_result:
        print(f"Error in Aesthetic Scorer: {aes_result.get('error')}")

    print()  # 空行を挿入
```

## 戻り値の構造

結果データの構造は以下のようになっています：

```python
{
    "phash1": {  # 1枚目の画像の pHash
        "model1_name": {  # 1つ目のモデルの結果
            "tags": ["tagA", "tagB"],
            "formatted_output": {...},
            "error": None
        },
        "model2_name": {  # 2つ目のモデルの結果
            "tags": ["[SCORE]score_value"],
            "formatted_output": ...,
            "error": None
        }
    },
    "phash2": {  # 2枚目の画像の pHash
        "model1_name": { ... },
        "model2_name": { ... }
    }
}
```

この pHash ベースの構造により、同じ画像に対する複数のモデルの結果を簡単に比較・集約できます。
