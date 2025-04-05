# リファレンス: サポートモデル詳細

このドキュメントでは、`image-annotator-lib` で利用可能な主要なアノテーションモデル（Tagger および Scorer）について、その特徴、仕様、設定例などを詳しく説明します。

**注意:** このドキュメントでは、ユーザーの指示に基づき、設定ファイル名を `annotator_config.toml` と記述していますが、現在の実際のコード実装では `models.toml` という名前が使用されている可能性があります。

## Scorer モデル (美的評価など)

主に画像の美的品質や特定の属性を評価するためのモデルです。

---

### 1. Aesthetic Shadow

アニメ画像の美的評価に特化したモデル。

#### 1.1. Aesthetic Shadow v1

*   **クラス名:** `AestheticShadowV1` (実装クラス: `src/image_annotator_lib/models/pipeline_scorers.py`)
*   **開発者:** shadowlilac
*   **ベース:** Vision Transformer (ViT-B/16)
*   **入力サイズ:** 1024x1024
*   **出力:** "hq" (高品質) と "lq" (低品質) のスコア (0.0-1.0)。通常 "hq" スコアを使用。
*   **特徴:** アニメ特有のスタイルを考慮。広く利用可能。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    [aesthetic-shadow-v1]
    class = "AestheticShadowV1"
    model_path = "shadowlilac/aesthetic-shadow"
    ```

#### 1.2. Aesthetic Shadow v2

*   **クラス名:** `AestheticShadowV2` (実装クラス: `src/image_annotator_lib/models/pipeline_scorers.py`)
*   **開発者:** shadowlilac
*   **ベース:** Vision Transformer (ViT-H/14)
*   **入力サイズ:** 1024x1024
*   **出力:** "hq" スコア (0.0-1.0)。スコアに基づき4段階評価 (very aesthetic/aesthetic/displeasing/very displeasing) が可能。
*   **特徴:** アニメのデフォルメ表現、陰影、色彩を評価。Stable Diffusion との連携に最適化。
*   **注意:** 公式リポジトリでの公開は停止。ミラーリポジトリを使用。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    [aesthetic-shadow-v2]
    class = "AestheticShadowV2"
    model_path = "NEXTAltair/cache_aestheic-shadow-v2" # ミラー
    ```

---

### 2. Cafe Aesthetic

汎用的な美的評価モデル。実写画像にも適用可能。

*   **クラス名:** `CafePredictor` (実装クラス: `src/image_annotator_lib/models/pipeline_scorers.py`)
*   **開発者:** cafeai
*   **ベース:** ViT-Base (microsoft/beit-base-patch16-384)
*   **入力サイズ:** 384x384
*   **出力:** "aesthetic" と "not_aesthetic" のスコア (0.0-1.0)。"aesthetic" スコアを 0-10 の整数スケールに変換して使用することが多い。
*   **特徴:** アニメ・実写両対応。マンガや低品質線画の識別。Waifu Diffusion のデータ選別に利用。バッチ処理が高速。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    [cafe-aesthetic]
    class = "CafePredictor"
    model_path = "cafeai/cafe_aesthetic"
    ```

---

### 3. CLIP Aesthetic Score Predictor (旧)

CLIP エンベディングとシンプルな MLP を使用したモデル。

*   **クラス名:** `ImprovedAestheticPredictor` (実装クラス: `src/image_annotator_lib/models/scorer_clip.py`)
*   **開発者:** Christopher Schuhmann
*   **ベース:** CLIP (OpenAI)
*   **出力:** 1-10 のスコア。
*   **特徴:** シンプルで高速。LAION 5B ベース。汎用性が高い。
*   **注意:** やや古いモデル。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    [clip-aesthetic-predictor-v2-vit-l-14] # モデル名は例
    class = "ImprovedAestheticPredictor"
    model_path = "openai/clip-vit-large-patch14" # CLIPモデルのパス
    # classifier_path = "path/to/aesthetic_classifier.pth" # 分類器のパス指定が必要な場合
    ```

---

### 4. Waifu-Diffusion Aesthetic Model (旧)

Waifu-Diffusion プロジェクト向けのアニメ画像専用モデル。

*   **クラス名:** `WaifuAestheticPredictor` (実装クラス: `src/image_annotator_lib/models/scorer_clip.py`)
*   **開発者:** Waifu Diffusion チーム
*   **ベース:** CLIP (openai/clip-vit-base-patch32)
*   **入力サイズ:** 224x224
*   **出力:** 0-10 の整数スコア。
*   **特徴:** アニメ調特化。シンプルな分類器 (3層 MLP)。
*   **注意:** やや古いモデル。分類器ファイル (`.pth`) の手動ダウンロードが必要な場合がある。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    [waifu-diffusion-aesthetic-v1] # モデル名は例
    class = "WaifuAestheticPredictor"
    model_path = "openai/clip-vit-base-patch32" # CLIPモデルのパス
    # classifier_path = "path/to/aes-B32-v0.pth" # 分類器のパス指定が必要
    ```

---

### 5. ImageReward

テキストプロンプトと生成画像の一致度および美的品質を評価するモデル。

**注意** NOTE: 実装が複雑なので現在未実装｡

*   **クラス名:** `ImageRewardScorer` (実装クラス: `src/image_annotator_lib/models/scorer_clip.py`)
*   **開発者:** THUDM (清華大学)
*   **ベース:** BLIP + MLP
*   **出力:** 標準正規分布に従うスコア。
*   **特徴:** 人間の好みを学習。プロンプト一致度と品質を同時評価。ReFL による生成モデル最適化機能。
*   **注意:** 高精度だが計算コストが高め。VRAM 16GB 以上推奨。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    [ImageReward-v1.0]
    class = "ImageRewardScorer"
    model_path = "THUDM/ImageReward" # または "Aesthetic" などサブタイプ指定
    ```

---

### 6. VisionReward (参考)

ImageReward の後継。多次元評価、チェックリスト評価、動画評価に対応。

*   **クラス名:** (現在ライブラリ未実装)
*   **開発者:** THUDM (清華大学)
*   **特徴:** より詳細で解釈可能な評価。動画対応。
*   **注意:** 最新モデル。ライブラリへの実装は今後の課題。

---

## Tagger モデル (タグ付け・キャプション生成)

画像の内容を表すタグやキャプションを生成するモデル。

---

### 1. WD Tagger (Waifu Diffusion Tagger)

アニメ/イラスト画像のタグ付けに特化した ONNX ベースのモデル。複数のバージョンが存在。

*   **クラス名:** `WDTagger` (実装クラス: `src/image_annotator_lib/models/tagger_onnx.py`)
*   **フレームワーク:** ONNX
*   **特徴:** 高速で比較的高精度。一般タグ、キャラクタータグなどを出力。閾値設定で出力タグ数を調整可能。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    [wd-v1-4-vit-tagger-v2]
    class = "WDTagger"
    model_path = "SmilingWolf/wd-v1-4-vit-tagger-v2"
    # threshold = 0.35 # 閾値 (モデルクラスのデフォルト値を使用)
    ```

---

### 2. Z3D Tagger

e621 などのデータセットで学習されたタグ付けモデル。ONNX ベース。

*   **クラス名:** `Z3D_E621Tagger` (実装クラス: `src/image_annotator_lib/models/tagger_onnx.py`)
*   **フレームワーク:** ONNX
*   **特徴:** 特定のデータセットに強いタグ付け能力。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    ["z3d-e621-tagger-v1"]
    class = "Z3D_E621Tagger"
    model_path = "zhonger/z3d-e621-tagger-v1"
    ```

---

### 3. BLIP / BLIP-2

画像キャプション生成モデル。画像の内容を自然言語で説明する。

*   **クラス名:**
    *   `BLIPTagger` (BLIP)
    *   `BLIP2Tagger` (BLIP-2)
    (実装クラス: `src/image_annotator_lib/models/tagger_transformers.py`)
*   **フレームワーク:** Transformers (PyTorch)
*   **特徴:** 画像全体の状況説明を生成。BLIP-2 はより高性能だが計算コストが高い。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    [blip-large-captioning]
    class = "BLIPTagger"
    model_path = "Salesforce/blip-image-captioning-large"

    ["blip2-opt-2.7b"]
    class = "BLIP2Tagger"
    model_path = "Salesforce/blip2-opt-2.7b"
    device = "cpu" # メモリ使用量が大きいため CPU 指定推奨の場合あり
    ```

---

### 4. GIT

画像キャプション生成モデル。

*   **クラス名:** `GITTagger` (実装クラス: `src/image_annotator_lib/models/tagger_transformers.py`)
*   **フレームワーク:** Transformers (PyTorch)
*   **特徴:** 比較的高速なキャプション生成。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    [GITLargeCaptioning]
    class = "GITTagger"
    model_path = "microsoft/git-large-coco"
    ```

---

### 5. Torii Gate Tagger

特殊な形式のタグを出力するモデル。

チャットボット形式で推論実行にはPromptが必要

*   **クラス名:** `ToriiGateTagger` (実装クラス: `src/image_annotator_lib/models/tagger_transformers.py`)
*   **フレームワーク:** Transformers (PyTorch)
*   **特徴:** 大規模モデル。特殊な前処理が必要。
*   **注意:** メモリ使用量が非常に大きい。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    ["ToriiGate-v0.3"]
    class = "ToriiGateTagger"
    model_path = "Minthy/ToriiGate-v0.3"
    device = "cpu" # CPU 指定推奨
    ```

---

### 6. DeepDanbooru

Danbooru タグを予測するモデル。TensorFlow ベース。

*   **クラス名:** `DeepDanbooruTagger` (実装クラス: `src/image_annotator_lib/models/tagger_tensorflow.py`)
*   **フレームワーク:** TensorFlow
*   **特徴:** Danbooru サイトのタグ体系に基づいた詳細なタグ付け。
*   **注意:** TensorFlow の環境設定が必要。
*   **設定例 (`config/annotator_config.toml`):**
    ```toml
    ["deepdanbooru-v3-20211112-sgd-e28"]
    class = "DeepDanbooruTagger"
    model_path = "https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip"
    # device = "cuda" # TensorFlow GPU 版が必要
    ```

---