# API リファレンス

このドキュメントでは、`image-annotator-lib` が提供する公開 API について説明します。

## 主要関数

### `annotate(images: list[Image.Image], model_names: list[str]) -> dict[str | None, dict[str, dict[str, Any]]]`

指定された複数の画像に対して、指定された複数のアノテーションモデル（Tagger または Scorer）を実行し、結果を集約して返します。

**引数:**

- `images` (`list[Image.Image]`): 処理対象となる PIL Image オブジェクトのリスト。
- `model_names` (`list[str]`): 使用するアノテーションモデルの名前のリスト。利用可能なモデル名は `list_available_annotators()` で取得できます。

**戻り値:**

- `dict[str | None, dict[str, dict[str, Any]]]`:\
  画像の pHash (または代替キー) をトップレベルキーとする辞書。
  各 pHash の値は、モデル名をキーとし、そのモデルによるアノテーション結果 (`AnnotationResult` 相当の辞書) を値とする辞書です。

  **戻り値の構造例:**

  ```python
  {
      "bf9a2f...": { # 画像1のpHash
          "wd-v1-4-vit-tagger-v2": {
              "tags": ["1girl", "solo", "blue_hair", ...],
              "formatted_output": {"general": {"1girl": 0.99, ...}, ...},
              "error": None
          },
          "aesthetic-shadow-v2": {
              "tags": ["aesthetic"], # スコアタグ
              "formatted_output": 0.85, # 整形済みスコア
              "error": None
          }
      },
      "unknown_image_0": { # 画像2 (pHash計算失敗時の代替キー)
          "wd-v1-4-vit-tagger-v2": {
              "tags": [],
              "formatted_output": {},
              "error": "Model execution failed: OOM"
          },
          "aesthetic-shadow-v2": {
              "tags": [],
              "formatted_output": None,
              "error": "Model execution failed: OOM"
          }
      }
  }
  ```

  **結果辞書の詳細 (`results[phash][model_name]`):**

  - `tags` (`list[str]`): 抽出されたタグ、スコアタグ、キャプションなどを文字列リストとして集約したもの。
  - `formatted_output` (`Any`): モデル固有の整形済み出力。デバッグや詳細分析に利用できます。
  - `error` (`str | None`): そのモデルでの処理中にエラーが発生した場合のエラーメッセージ。成功した場合は `None`。

**内部処理の概要:**

1.  入力された `model_names` に対応するモデルクラスを `ModelRegistry` から取得します。
2.  各モデルについて、`ModelLoad` を通じてモデルのロード/キャッシュ管理を行います。
3.  各画像について pHash を計算します。
4.  各モデルの `predict` メソッド (実際には `BaseAnnotator` の共通 `predict`) を呼び出し、画像のリストを処理します。
    - 内部ではチャンク分割、前処理 (`_preprocess_images`)、推論 (`_run_inference`)、後処理 (`_format_predictions`, `_generate_tags`) が行われます。
    - 各画像・各モデルの結果 (`AnnotationResult` 相当) が生成されます。
5.  生成された結果を pHash とモデル名で集約し、最終的な辞書構造を構築して返します。
6.  エラーハンドリング: 個別のモデルや画像でエラーが発生しても、処理は継続され、エラー情報が結果辞書内の `error` フィールドに格納されます。

### `list_available_annotators() -> list[str]`

現在利用可能な（設定ファイル `config/annotator_config.toml` に定義され、クラスがロード可能な）アノテーションモデルの名前のリストを返します。

**引数:** なし

**戻り値:**

- `list[str]`: 利用可能なモデル名の文字列リスト。

**例:**

```python
from image_annotator_lib import list_available_annotators

available = list_available_annotators()
print(available)
# 出力例: ['wd-v1-4-vit-tagger-v2', 'aesthetic-shadow-v2', 'my-custom-tagger']
```

## データ構造

### `AnnotationResult` (TypedDict - 内部表現)

`BaseAnnotator.predict` メソッドが内部的に生成し、`annotate` 関数の最終的な戻り値の構成要素となるデータ構造です。

**注意:** これはライブラリ内部で使用される型定義であり、直接ユーザーがインスタンス化するものではありません。`annotate` 関数の戻り値に含まれる辞書がこの構造に準拠します。

```python
from typing import TypedDict, Any, List, Optional

class AnnotationResult(TypedDict, total=False):
    phash: Optional[str]       # 画像の知覚ハッシュ (計算失敗時は None)
    model_name: str            # 必須: モデル名
    tags: List[str]            # 必須: タグ、スコアタグ、キャプションなどを集約したリスト
    formatted_output: Any      # 必須: 整形済み出力 (モデル依存)
    error: Optional[str]       # エラー情報 (エラーがない場合は None)
```

- `total=False` のため、エラー発生時など状況によっては一部のキーが存在しない可能性がありますが、`annotate` 関数の戻り値としては `tags`, `formatted_output`, `error` は通常含まれます (`phash`, `model_name` は上位のキーとして使われます)。

## 例外クラス

ライブラリ内で発生する可能性のあるカスタム例外です。`image_annotator_lib.exceptions` からインポートして使用できます。

- `AnnotatorError`: ライブラリ関連の一般的なエラーの基底クラス。
- `ModelLoadError`: モデルのロードやキャッシュ管理に関するエラー。
- `ModelNotFoundError`: 指定されたモデル名が設定ファイルに存在しない、またはクラスが見つからない場合のエラー。
- `OutOfMemoryError`: モデルの実行中に GPU または CPU のメモリが不足した場合のエラー。`annotate` 関数の結果辞書の `error` フィールドにも記録されることがあります。

```python
from image_annotator_lib.exceptions import ModelNotFoundError, AnnotatorError

try:
    # ライブラリの関数呼び出し
    pass
except ModelNotFoundError as e:
    print(f"Model not found: {e}")
except AnnotatorError as e:
    print(f"An annotator error occurred: {e}")
```
