# 解説: 旧ライブラリ情報 (統合前)

このドキュメントは、現在の `image-annotator-lib` に統合される前の、旧ライブラリ (`scorer_wrapper_lib` および `tagger_wrapper_lib`) に関する情報を記録として残すものです。現在のライブラリの設計背景を理解する一助となります。

## 概要

統合前は、画像スコアリング機能と画像タギング機能は、それぞれ独立したライブラリとして開発されていました。

- **`scorer_wrapper_lib`**: 各種美的評価モデル (Aesthetic Shadow, Cafe Aesthetic など) を統一インターフェースで操作するためのライブラリ。
- **`tagger_wrapper_lib`**: 各種画像タグ付け・キャプション生成モデル (WD Tagger, BLIP, DeepDanbooru など) を統一インターフェースで操作するためのライブラリ。

両ライブラリは、類似した目的とアーキテクチャ（設定ファイル駆動、基底クラスによる抽象化、共通 API 関数の提供など）を目指していましたが、実装の詳細や命名規則には差異がありました。

## 旧ライブラリの基本アーキテクチャ (共通点)

両ライブラリともに、以下のような共通の設計要素を持っていました。

- **基底クラス**:
  - `scorer_wrapper_lib`: `BaseScorer`
  - `tagger_wrapper_lib`: `BaseTagger`
  - これらの基底クラスが、モデルの初期化、コンテキスト管理 (`__enter__`/`__exit__`)、基本的な予測インターフェース (`predict`) などの共通機能を提供していました。
- **モデルタイプ別の中間クラス**: 特定のフレームワーク (Transformers, ONNX, TensorFlow など) に共通する処理をまとめる中間クラスが存在しました。
- **具象モデルクラス**: 各モデル固有の実装を行うクラス。
- **設定ファイル**:
  - モデルの種類、パス、設定などを TOML ファイル (例: `models.toml`, `taggers.toml`) で管理していました。
- **レジストリ**: 設定ファイルとクラス実装を紐付け、利用可能なモデルを管理する仕組み (`registry.py`) がありました。
- **主要 API 関数**:
  - `list_available_models()` / `list_available_taggers()`: 利用可能なモデル名をリストで返す関数。
  - `evaluate()`: 画像リストとモデル名リストを受け取り、評価/タグ付けを実行して結果を返す関数。
    - **注意:** 旧 `evaluate` 関数の戻り値形式は、現在の pHash ベースの形式とは異なり、モデル名をキーとするリスト形式 (`dict[str, list[dict[str, Any]]]`) でした。
- **モデル管理**: モデルインスタンスのキャッシュ、GPU/CPU 間のメモリ移動などの機能を持っていました (実装の詳細は `ModelLoad` / `model_factory.py` に引き継がれています)。

## 旧ライブラリのディレクトリ構造 (例)

参考として、旧ライブラリのおおよそのディレクトリ構造を示します。

**`scorer-wrapper-lib` (例):**

```
scorer-wrapper-lib/
    ├── src/
    │   └── scorer_wrapper_lib/
    │       ├── __init__.py
    │       ├── scorer.py           # evaluate 関数など
    │       ├── scorer_registry.py
    │       ├── core/
    │       │   ├── base.py         # BaseScorer など
    │       │   ├── utils.py
    │       │   └── model_factory.py
    │       ├── score_models/       # 具象モデルクラス
    │       └── exceptions/
    ├── config/
    │   └── models.toml
    └── tests/
```

**`tagger-wrapper-lib` (例):**

```
tagger-wrapper-lib/
    ├── src/
    │   └── tagger_wrapper_lib/
    │       ├── __init__.py
    │       ├── tagger.py           # evaluate 関数など
    │       ├── registry.py
    │       ├── core/
    │       │   ├── base.py         # BaseTagger など
    │       │   ├── utils.py
    │       │   └── model_factory.py
    │       ├── tagger_models/      # 具象モデルクラス
    │       └── exceptions/
    ├── config/
    │   └── taggers.toml
    └── tests/
```

## 統合への経緯

これらのライブラリは機能的に重複する部分が多く、特にモデル管理 (`model_factory.py`) やコアな基底クラス (`BaseScorer`/`BaseTagger`) の設計において共通化のメリットが大きいと判断されました。また、API を統一することで利用者が両機能をシームレスに扱えるようにするため、単一の `image-annotator-lib` への統合が行われました。

統合プロセスにおける設計上の決定事項については、[設計決定の背景](./design_decisions.md) を参照してください。
