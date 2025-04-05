# リファレンス: 設定ファイル仕様

このドキュメントでは、`image-annotator-lib` が使用する設定ファイルのフォーマットと設定可能な項目について説明します。

## 概要

`image-annotator-lib` は、利用可能なアノテーションモデルとその設定を TOML ファイルから読み込みます。このファイルによって、どのモデルをライブラリに認識させ、どのように動作させるかを定義します。

## ファイルの場所

設定ファイルは、通常、ライブラリを使用するプロジェクトのルートディレクトリ直下にある `config` ディレクトリ内に配置されます。

*   **デフォルトパス:** `[プロジェクトルート]/config/annotator_config.toml`

ライブラリは起動時にこのパスから設定を読み込もうとします。

## ファイルフォーマット

設定ファイルは TOML (Tom's Obvious, Minimal Language) 形式で記述されます。

*   ファイル全体が UTF-8 でエンコードされている必要があります。
*   各アノテーションモデルの設定は、トップレベルの **テーブル (セクション)** として定義されます。
*   テーブル名は、ライブラリ内でモデルを一意に識別するための **モデル名** となります。この名前は `list_available_annotators()` で返され、`annotate` 関数の `model_names` 引数で指定するものです。
    *   **重要:** モデル名にドット (`.`) が含まれる場合は、TOML の仕様に従い、モデル名をダブルクォーテーション (`"`) で囲む必要があります (例: `["my.model-v1.0"]`)。
*   各テーブル内には、そのモデルの設定をキーと値のペアで記述します。

## 設定項目

各モデルのテーブル (セクション) 内で以下のキーを設定できます。

*   **`class` (文字列, 必須)**
    *   このモデルに対応する Python クラスの名前。`src/image_annotator_lib/models/` 内に実装されている具象クラス (例: `WDTagger`, `AestheticShadowV2`) を指定します。
    *   `ModelRegistry` はこの名前を使って対応するクラスを探します。

*   **`model_path` (文字列, 必須)**
    *   モデルファイルまたはモデルリポジトリの場所を指定します。以下の形式がサポートされています。
        *   **Hugging Face Hub リポジトリ ID:** (例: `"Salesforce/blip-image-captioning-large"`) ライブラリが自動的に必要なファイルをダウンロードします。
        *   **直接ダウンロード URL:** (例: `"https://github.com/path/to/model.zip"`) 指定された URL からファイルをダウンロードします。ZIP ファイルの場合は自動的に解凍されます。
        *   **ローカルファイルパス:** (例: `"C:/path/to/my_model.onnx"` または `"./local_models/model.safetensors"`) ローカルにあるモデルファイルまたはディレクトリへのパス。絶対パスまたは相対パスで指定できます。

*   **`device` (文字列, 任意)**
    *   モデルの推論に使用するデバイスを指定します (例: `"cuda"`, `"cpu"`)。
    *   省略した場合、ライブラリは利用可能な最適なデバイス (通常は CUDA GPU があれば `"cuda"`) を選択しようとします。フレームワークによっては、そのフレームワークのデフォルト動作に従います。
    *   特定のモデルを CPU で実行したい場合などに指定します。

*   **その他のカスタム設定 (任意)**
    *   モデルクラスの実装によっては、上記以外のカスタム設定キーを受け付ける場合があります (例: `threshold`, `tag_file_path`, `onnx_providers`)。
    *   これらのキーは、モデルクラスの `__init__` メソッド内で `self.config.get("your_key", default_value)` のようにして読み込む必要があります。
    *   カスタム設定を追加する場合は、対応するモデルクラスのドキュメントやコードを確認してください。
    *   **推奨:** モデル固有のパラメータ（閾値など）は、可能な限りモデルクラスのインスタンス変数としてハードコーディングするか、デフォルト値を設定し、設定ファイルでの指定は任意とすることが望ましいです。

## 設定例

```toml
# config/annotator_config.toml

# --- Tagger Models ---

[wd-v1-4-vit-tagger-v2]
class = "WDTagger"
model_path = "SmilingWolf/wd-v1-4-vit-tagger-v2" # Hugging Face Repo ID
estimated_size_gb = 1.695
# device = "cuda" # 省略可能 (デフォルトで CUDA が試される)

["z3d-e621-tagger-v1"] # モデル名にハイフンがあるので引用符は不要
class = "Z3D_E621Tagger"
model_path = "zhonger/z3d-e621-tagger-v1"
estimated_size_gb = 1.695

[blip-large-captioning]
class = "BLIPTagger"
model_path = "Salesforce/blip-image-captioning-large"
estimated_size_gb = 2.1

["deepdanbooru-v3-20211112-sgd-e28"]
class = "DeepDanbooruTagger"
model_path = "https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip" # URL
estimated_size_gb = 0.723
device = "cpu" # このモデルは CPU で実行する指定

# --- Scorer Models ---

[aesthetic-shadow-v1]
class = "AestheticShadowV1"
model_path = "shadowlilac/aesthetic-shadow"
estimated_size_gb = 1.1 # 仮の値

[aesthetic-shadow-v2]
class = "AestheticShadowV2"
model_path = "NEXTAltair/cache_aestheic-shadow-v2" # ミラーリポジトリ
estimated_size_gb = 1.1 # 仮の値

[cafe-aesthetic]
class = "CafePredictor"
model_path = "cafeai/cafe_aesthetic"
estimated_size_gb = 0.086 # 仮の値

# --- Custom Model Example ---
# ["my.custom-tagger.v1.0"] # モデル名にドットが含まれる場合は引用符で囲む
# class = "MyCustomTagger"
# model_path = "./models/custom_tagger_v1" # ローカルパス
# estimated_size_gb = 0.2
# custom_threshold = 0.4 # モデル固有のカスタム設定

```