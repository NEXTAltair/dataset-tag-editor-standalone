# Dataset Tag Editor Standalone フォーク版ドキュメント

## 1. コンテキスト

### 1.1 フォークの目的
このリポジトリは[Dataset Tag Editor Standalone](https://github.com/toshiaki1729/dataset-tag-editor-standalone)からフォークされ、以下の目的で再構築されています：

- タグ付け機能の外部提供可能化
- スコアリング機能の外部提供可能化
- 既存の実装はそのまま活用し、外部提供用のインターフェースを追加

### 1.2 スコープ
#### 変更対象
- 既存実装のラッパーインターフェース作成
- 外部提供用APIの設計
- 最小限の新規実装（ラッパーのみ）

#### 変更対象外
- 既存の実装コード
- GUIインターフェース
- データセット管理機能
- タグ編集機能
- フィルタリング機能

## 2. 既存の実装

### 2.1 タグ付け機能
#### サポートされているモデル
- BLIP
- BLIP2
- GIT Large
- DeepDanbooru
- WaifuDiffusion (v1, v2, v3)
- Z3D-E621

### 2.2 スコアリング機能
#### サポートされているモデル
- Aesthetic Shadow
- CafeAI Aesthetic Classifier
- Improved Aesthetic Predictor
- Waifu Aesthetic Classifier

## 3. ディレクトリ構成

```
lib/
├── tagger/
│   ├── __init__.py          # 外部提供用インターフェース
│   ├── wrapper.py           # 既存実装のラッパー
│   └── extensions/          # カスタムタガー用ディレクトリ
│       └── __init__.py
└── scorer/
    ├── __init__.py          # 外部提供用インターフェース
    ├── wrapper.py           # 既存実装のラッパー
    └── extensions/          # カスタムスコアラー用ディレクトリ
        └── __init__.py

# 既存の実装（変更なし）
scripts/
├── dataset_tag_editor/
│   ├── interrogators/       # 既存のタガー実装
│   │   ├── blip_large_captioning.py
│   │   ├── deep_danbooru_tagger.py
│   │   └── ...
│   └── ...
└── ...

userscripts/
└── taggers/                 # 既存のスコアラー実装
    ├── aesthetic_shadow.py
    ├── cafeai_aesthetic_classifier.py
    └── ...
```

## 4. 実装の詳細

### 4.1 ラッパーインターフェース
```python
# lib/tagger/wrapper.py
class TaggerWrapper:
    """既存タガー実装のラッパー"""
    def __init__(self, model_name: str):
        self.tagger = self._get_tagger(model_name)

    def predict(self, image: Image) -> list[str]:
        """
        画像からタグを予測

        Args:
            image: 入力画像
        Returns:
            予測されたタグのリスト
        """
        return self.tagger.predict(image)

    def batch_predict(self, images: list[Image]) -> list[list[str]]:
        """
        バッチ処理による予測

        Args:
            images: 入力画像のリスト
        Returns:
            各画像の予測タグのリスト
        """
        return self.tagger.batch_predict(images)

# lib/scorer/wrapper.py
class ScorerWrapper:
    """既存スコアラー実装のラッパー"""
    def __init__(self, model_name: str):
        self.scorer = self._get_scorer(model_name)

    def get_available_models() -> list[dict]:
        """
        登録済みスコアリングモデルのメタデータ一覧を返す

        Returns:
            [{
                "name": "表示用モデル名",
                "version": "モデルバージョン",
            }]
        """

    def predict(self, image: Image.Image) -> list[ImageScore]:
        """画像のスコアを予測します

        Args:
            image: 入力画像

        Returns:
            list[ImageScore]: 予測結果を含むリスト
            各ImageScoreは以下の構造:
            {
                "image_id": str,    # 画像の識別子
                "model_name": str,  # モデル名
                "score": float,     # スコア
            }
        """

    def predict_batch(
        self,
        images: list[Image.Image]
    ) -> Generator[list[ImageScore], None, None]:
        """複数の画像に対してバッチ処理でスコアを予測します

        Args:
            images: 入力画像のリスト

        Returns:
            Generator[list[ImageScore], None, None]:
            各画像の予測結果を含むジェネレータ。
            ImageScoreは以下の構造:
            {
                "image_id": str,    # 画像の識別子
                "model_name": str,  # モデル名
                "score": float,     # スコア
            }

        Raises:
            AttributeError: バッチ処理に対応していないスコアラーの場合
        """
```

### 4.2 使用例
```python
from lib.tagger import create_tagger
from lib.scorer import create_scorer

# タガーの使用
tagger = create_tagger("BLIP")
tags = tagger.predict(image)

# バッチ処理
batch_tags = tagger.batch_predict(images)

# スコアラーの使用
scorer = create_scorer("aesthetic")
score = scorer.evaluate(image)

# バッチ処理
batch_scores = scorer.batch_evaluate(images)
```

## 5. 拡張方法

### 5.1 カスタムタガーの追加
1. 既存の方法でタガーを実装（scripts/dataset_tag_editor/interrogators/）
2. 必要に応じてラッパーを拡張

### 5.2 カスタムスコアラーの追加
1. 既存の方法でスコアラーを実装（userscripts/taggers/）
2. 必要に応じてラッパーを拡張

## 6. 運用上の注意点

### 6.1 システム要件
- Python >= 3.9
- PyTorch >= 1.10.0（CUDA対応）
- 各種依存ライブラリ（requirements.txtを参照）

### 6.2 外部提供時の注意点
- ラッパーを通じて既存機能を利用
- 既存実装への変更は最小限に
- バージョン管理とドキュメント整備の重要性
- 上書き､追加に関する処理は利用側(LoRAIro)で実装