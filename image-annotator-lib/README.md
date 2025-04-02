# Image Annotator Lib

`image-annotator-lib` は、様々な画像アノテーションモデル（タガー、スコアラーなど）を統一されたインターフェースで利用するための Python ライブラリです。`scorer_wrapper_lib` と `tagger_wrapper_lib` を統合し、コードの重複削減、API の統一、メンテナンス性の向上、機能拡張の容易化を目指しています。

## 主な機能

- 複数の画像タギングモデルと画像スコアリングモデルをサポート
- 統一された API (`annotate`) による複数モデル・複数画像の一括処理
- pHash に基づく画像と結果の紐付け
- 設定ファイル (`models.toml`) によるモデル選択と設定
- メモリ使用量に基づく効率的なモデルキャッシュ管理 (LRU, CPU 退避/CUDA 復元)

## インストール

本ライブラリは [uv](https://github.com/astral-sh/uv) を使用したパッケージ管理を推奨しています。

1.  **仮想環境の作成:**

    ```bash
    uv venv
    ```

    (Windows の場合: `.venv\Scripts\activate`, Linux/macOS の場合: `source .venv/bin/activate`)

2.  **依存関係のインストール:**
    プロジェクトルートにある `pyproject.toml` を使用してインストールします。
    ```bash
    uv pip install -e .
    ```
    (`-e` オプションは開発モードでのインストールです。通常のライブラリ利用の場合は不要な場合があります。)

## 基本的な使い方

```python
from PIL import Image
from image_annotator_lib import annotate, list_available_annotators

# 利用可能なモデル名を確認
available_models = list_available_annotators()
print("Available models:", available_models)

# 評価したい画像を開く (PIL.Image オブジェクトのリスト)
try:
    image1 = Image.open("path/to/your/image1.jpg")
    image2 = Image.open("path/to/your/image2.png")
    images_to_process = [image1, image2]
except FileNotFoundError:
    print("Error: Image file not found. Please provide valid image paths.")
    exit()

# 評価したいモデル名をリストで指定
# 例: 利用可能なモデルから WD Tagger と Aesthetic Scorer を選択
model_names_to_use = ["wd-v1-4-vit-tagger-v2", "aesthetic-shadow-v2"] # 実際のモデル名に合わせてください
# 使用するモデルが利用可能か確認 (任意)
for model_name in model_names_to_use:
    if model_name not in available_models:
        print(f"Warning: Model '{model_name}' is not available in the current configuration.")
        # 必要に応じてエラー処理やモデルリストからの除外を行う

# 評価を実行 (位置引数で渡す)
# 戻り値は pHash をキーとした辞書形式
# {phash: {model_name: {"tags": [...], "formatted_output": ..., "error": ...}}}
results = annotate(
    images_to_process,
    model_names_to_use
)

# 結果の処理例
for phash, model_results in results.items():
    print(f"--- Image (pHash: {phash}) ---")
    if not model_results:
        print("  No results for this image.")
        continue
    for model_name, result in model_results.items():
        print(f"  Model: {model_name}")
        if result.get("error"):
            print(f"    Error: {result['error']}")
        else:
            tags = result.get('tags', [])
            formatted_output = result.get('formatted_output')
            print(f"    Tags: {tags}")
            # print(f"    Formatted Output: {formatted_output}") # 必要に応じて出力
    print("-" * (len(str(phash)) + 18)) # 区切り線 (phashがNoneの場合も考慮)

```

## ドキュメント

より詳細な情報については、以下のドキュメントを参照してください。

- **チュートリアル:**
  - [基本的な使い方](./docs/TUTORIALS/basic_usage.md)
  - [複数モデルでの評価](./docs/TUTORIALS/annotate_multiple_models.md)
- **ハウツーガイド:**
  - [新しいモデルの追加方法](./docs/HOW_TO_GUIDES/add_new_model.md)
  - [ロギングの設定](./docs/HOW_TO_GUIDES/configure_logging.md)
  - [テストの実行方法](./docs/HOW_TO_GUIDES/run_tests.md)
- **リファレンス:**
  - [API リファレンス](./docs/REFERENCE/api.md)
  - [設定ファイル仕様](./docs/REFERENCE/configuration.md)
  - [サポートモデル詳細](./docs/REFERENCE/models.md)
- **解説:**
  - [アーキテクチャ](./docs/EXPLANATION/architecture.md)
  - [設計決定の背景](./docs/EXPLANATION/design_decisions.md)
  - [リファクタリング](./docs/EXPLANATION/refactoring.md)
  - [旧ライブラリ情報](./docs/EXPLANATION/legacy_info.md)
