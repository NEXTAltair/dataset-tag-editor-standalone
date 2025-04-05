# アーキテクチャと設計思想

このドキュメントでは、`image-annotator-lib`の全体的なアーキテクチャと設計思想について解説します。

## 設計原則

本ライブラリは、`scorer_wrapper_lib`と`tagger_wrapper_lib`を統合するにあたり、以下の原則に基づいて設計されました。

- **コードの重複削減**: 共通する機能を基底クラスに集約し、重複する実装を排除します。
- **API の統一**: Tagger と Scorer を同じインターフェース（`annotate`関数）で扱えるようにします。
- **メンテナンス性の向上**: クラス構造を整理し、責務を明確にすることで、将来の修正や変更を容易にします。
- **機能拡張の容易化**: 新しいモデルやフレームワークを追加しやすい構造を目指します。
- **YAGNI 原則**: 現時点で明確に必要な機能のみを実装し、設計をシンプルに保ちます。

## ディレクトリ構造

ライブラリの主要なソースコードは`src/image_annotator_lib/`以下に配置されています。

```
src/
└── image_annotator_lib/   # ライブラリパッケージ
    ├── __init__.py        # パッケージ初期化、主要APIのエクスポート
    ├── core/              # Tagger/Scorer共通の基盤モジュール
    │   ├── base.py          # BaseAnnotator, フレームワーク別基底クラス, 型定義
    │   ├── model_factory.py # ModelLoad (モデルロード/キャッシュ管理)
    │   ├── registry.py      # モデル登録/取得関連
    │   └── utils.py         # 共通ユーティリティ (設定読み込みなど)
    ├── exceptions/        # カスタム例外クラス
    │   └── errors.py
    ├── models/            # 各モデルの実装 (具象クラス)
    │   ├── tagger_onnx.py
    │   ├── tagger_transformers.py
    │   ├── tagger_tensorflow.py
    │   ├── scorer_clip.py
    │   └── pipeline_scorers.py
    └── api.py             # ユーザー向けAPI関数 (annotate)
```

## クラス階層 (3 層構造)

コードの重複を避け、責務を明確にするため、以下の 3 層構造のクラス階層を採用しています。

### 1. `BaseAnnotator` (`core/base.py`)

全てのアノテーター（Tagger/Scorer）に共通する処理とインターフェースを提供します。

**責務**:

- 共通属性（`model_name`, `config`, `logger`など）の初期化
- **共通化された`predict`メソッド**: チャンク処理、pHash 計算、基本的なエラーハンドリング、標準的な結果の生成
- サブクラスが実装すべき抽象ヘルパーメソッド (`_preprocess_images`, `_run_inference`, `_format_predictions`, `_generate_tags`) の定義
- コンテキスト管理 (`__enter__`, `__exit__`) のインターフェース定義

### 2. フレームワーク別基底クラス (`core/base.py`)

`BaseAnnotator`を継承し、特定の ML フレームワーク（ONNX, Transformers など）に共通する処理を実装します。

**例**:

- `ONNXBaseAnnotator`
- `TransformersBaseAnnotator`
- `TensorflowBaseAnnotator`
- `ClipBaseAnnotator`
- `PipelineBaseAnnotator`

**責務**:

- フレームワーク固有のモデルロード/解放ロジック
- `BaseAnnotator`の抽象ヘルパーメソッドの実装

### 3. 具象モデルクラス (`models/`)

対応するフレームワーク別基底クラスを継承し、個々のモデルに固有の処理のみを実装します。

**例**:

- `WDTagger`
- `BLIPTagger`
- `AestheticShadowV1`
- `CafePredictor`

**責務**:

- モデル固有の初期化
- モデル固有のファイル読み込み
- 必要に応じてヘルパーメソッドのオーバーライド

## 主要クラス/モジュールの役割

### `BaseAnnotator` (`core/base.py`)

上記参照。

### `ModelLoad` (`core/model_factory.py`)

- モデルコンポーネントのロード、キャッシュ管理、リソース解放を担当
- フレームワークごとにロードメソッドを提供
- メモリ管理のための LRU キャッシュ戦略を実行

### モデル登録関連 (`core/registry.py`)

- 利用可能なアノテータークラス (`BaseAnnotator` のサブクラス) を設定ファイル (`annotator_config.toml`) と照合し、管理します。
- 設定名 (例: `"wd-v1-4-vit-tagger-v2"`) に対応するクラスオブジェクトを取得する機能を提供します (例: `get_annotator_class(model_name)`)。
- `list_available_annotators()` 関数の実装を提供し、利用可能なモデル名の一覧を返します。

### `core/utils.py`

- 設定ファイルの読み込み
- モデルサイズの保存
- ファイルダウンロード、キャッシュ管理ユーティリティ
- ロガー設定

### `exceptions/errors.py`

- ライブラリ固有の例外クラスを定義

### `api.py`

- ユーザー向けの主要なエントリーポイント（`annotate`関数）を提供

## 処理フロー

1. **初期化とモデル検出**:

   - `list_available_annotators()`関数 (内部で `core/registry.py` の機能を使用) が設定ファイル`annotator_config.toml`を読み込み、利用可能なモデルを検出
   - 対応するモデルクラスを `core/registry.py` の機能を使って検索

2. **アノテーション実行**:
   - ユーザーが`annotate(images, model_names)`を呼び出し
   - 各モデル名に対して、`core/registry.py` の機能を使って対応するクラスを取得しインスタンス化
   - `ModelLoad`を使用してモデルをロード（キャッシュも考慮）
   - 各モデルの`predict`メソッド（共通実装）を呼び出し
   - 結果を集約して最終的な辞書構造を構築して返却

## 設計決定

### 統一 API の採用

Tagger モデルと Scorer モデルを共通のインターフェース(`annotate`関数)で扱えるようにしました。これにより、ユーザーはモデルの種類を意識せずに一貫した方法で異なるモデルを使用できます。

### 3 層クラス階層

3 層構造のクラス階層により、コードの重複を劇的に減らし、新しいモデル追加のハードルを下げることができました。新しいモデルを追加する場合、共通処理は`BaseAnnotator`と「フレームワーク別基底クラス」が提供するため、モデル固有の処理のみを実装すれば良くなりました。

### pHash ベースの結果管理

画像の知覚ハッシュ(pHash)を結果のキーとして使用することで、入力順序や処理順序に依存せずに結果を一意に識別できるようになりました。これにより、複数モデルの結果を効率的に集約できます。

### シンプルなユーザーインターフェース

ライブラリの使用を簡素化するため、最小限の API を公開しています。大部分の処理は内部で自動的に行われ、ユーザーは`annotate`関数と`list_available_annotators`関数だけを覚えれば基本的な使用が可能です。

### モデルサイズに基づくメモリ管理

各モデルの推定サイズ情報を活用し、利用可能な GPU メモリを効率的に使用する LRU キャッシュ戦略を実装しています。これにより、限られたリソースでも複数の大規模モデルを切り替えて使用することが可能になりました。

## リファレンス実装

このライブラリは、`scorer_wrapper_lib`と`tagger_wrapper_lib`の統合と改良によって作成されています。いくつかの主要な変更点は以下の通りです。

1. **クラス階層の最適化**: 以前は個別のモデルクラスでかなりの重複があったコードを、基底クラスに集約しました。
2. **結果形式の統一**: 異なる形式だった Tagger と Scorer の結果を統一されたフォーマットで返すようにしました。
3. **メモリ管理の改善**: より洗練されたモデルキャッシュ戦略を実装しました。
4. **エラーハンドリングの強化**: より詳細で一貫したエラー報告を導入しました。
