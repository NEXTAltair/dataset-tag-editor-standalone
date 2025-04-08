# memory-bank/decisionLog.md

## 設計に関する決定事項

### 2025/04/03: pHash ベースの画像識別子システムの採用

- **決定**: 画像の識別にインデックスではなく pHash(知覚ハッシュ)を使用し、`annotate`関数内で一度だけ pHash を計算する設計に変更。
- **理由**:
  - 複数のモデルで同じ画像を処理する際に pHash 計算を 1 回だけ行うことで効率化できる。
  - 同じ画像には常に同じ pHash が割り当てられ、結果の一貫性が保証される。
  - 画像が分かりやすい識別子を持つことで、結果の整理・検索・参照が容易になる。
  - モデル間の結果比較が簡単になる。
- **実装方法**:
  - `annotate`関数内で最初に全画像の pHash を計算し、リストとマッピング辞書で保持。
  - `BaseAnnotator.predict`メソッドを拡張し、事前計算された pHash リストを受け取れるようにした。
  - `_calculate_phash`関数をモジュールレベルのユーティリティ関数として実装し、`core.utils`に配置。

### 2025/04/01: 3 層クラス階層の採用

- **決定**: アノテータークラスを以下の 3 層構造とする。
  1. `BaseAnnotator`: フレームワーク非依存の共通処理(チャンク処理、基本エラーハンドリング、結果構造生成など)とインターフェース定義。
  2. フレームワーク別基底クラス (`ONNXBaseAnnotator` など): フレームワーク固有の共通処理(モデルロード/解放、基本推論呼び出しなど)。
  3. 具象モデルクラス (`WDTagger` など): モデル固有の処理(特殊な前処理/後処理、タグファイル読み込みなど)。
- **理由**:
  - コードの重複を最大限削減し、共通処理を基底クラスに集約するため。
  - 各クラスの責務を明確化し、メンテナンス性と拡張性を向上させるため。
  - `predict` メソッドを `BaseAnnotator` で共通化し、一貫した処理フローを保証するため。
- **代替案**:
  - 2 層構造(Base + 具象): フレームワーク共通処理の重複が発生する可能性。
  - Mixin クラスの利用: 複雑化する可能性。

### 2025/04/01: `ModelLoad` と `ModelRegistry` の共通化

- **決定**: モデルのロード、キャッシュ管理、リソース解放を行う `ModelLoad` クラスと、モデルクラスの登録・取得を行う `ModelRegistry` クラスを `core` モジュールに配置し、Tagger と Scorer で共通利用する。
- **理由**:
  - 両ライブラリで類似していたモデル管理ロジックを統合し、重複を排除するため。
  - メモリ管理(CPU 退避、CUDA 復元、OOM ハンドリング)を一元化し、効率と安定性を向上させるため。
  - モデルの追加・管理を容易にするため。

### 2025/04/01: 設定ファイル形式として TOML を採用

- **決定**: モデル設定ファイル (`config/annotator_config.toml`) の形式として TOML を採用する。
- **理由**:
  - 人間が読み書きしやすく、構造化された設定を記述できるため。
  - Python 標準ライブラリ (`tomllib`, Python 3.11+) または外部ライブラリ (`toml`) で容易にパースできるため。
  - 既存の `tagger-wrapper-lib` および `scorer-wrapper-lib` で TOML が使用されていたため、移行コストが低い。
- **代替案**:
  - JSON: コメントが書けない。
  - YAML: 依存ライブラリが必要、インデントに厳格。

### 2025/04/01: コーディング規約の採用

- **決定**: Ruff (フォーマット、リント)、Mypy (型チェック)、Google スタイル Docstring (日本語) を採用する。
- **理由**:
  - コードの一貫性と品質を保つため。
  - 静的解析により早期にエラーを発見するため。
  - ドキュメントの可読性と保守性を向上させるため。

## ModelLoad クラスの設計変更(2024/04/02)

### 決定事項

1. 元の実装を維持

   - 理由:既存の安定した実装を優先
   - 影響:新機能の追加は見送り

2. 最小限の統合
   - クラス変数による状態管理を継続
   - エラーハンドリングの基本実装を維持
   - メソッド名と構造を保持

### 影響範囲

- `model_factory.py`の実装を元に戻す
- 不要な拡張機能を削除

### フォローアップ

- 動作確認の実施
- 既存のテストケースの実行
- ドキュメントの更新

## 2024-04-02: ModelLoad の階層構造の改善

### 背景と課題

- 現在の ModelLoad の実装では、各モデルタイプ(Transformers、ONNX、CLIP 等)のロード処理が単一クラスに集中している
- 特に CLIP モデルのロードが特別な扱いとなっており、他のモデルタイプとの一貫性が欠如
- コンポーネントの構造が各モデルタイプで異なるため、統一的なインターフェースでの抽象化が困難

### 決定事項

二階層のローダー構造を採用:

1. **基底ローダー層**

   - `BaseModelLoader`: 全モデルタイプ共通の基底クラス
   - メモリ管理、キャッシュ制御の共通機能を提供
   - 必要なコンポーネントの定義を強制

2. **具象ローダー層**
   - 各モデルタイプ固有のローダー実装
   - 例: `TransformersLoader`, `ONNXLoader`, `CLIPLoader`
   - モデルタイプ固有の初期化とロードロジックを実装

### 設計の詳細

```python
class BaseModelLoader:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def load(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def get_required_components(self) -> list[str]:
        pass

class TransformersLoader(BaseModelLoader):
    def get_required_components(self) -> list[str]:
        return ["model", "processor"]

class ONNXLoader(BaseModelLoader):
    def get_required_components(self) -> list[str]:
        return ["session", "csv_path"]

class CLIPLoader(BaseModelLoader):
    def get_required_components(self) -> list[str]:
        return ["model", "processor", "clip_model"]
```

### 利点

1. 責任の明確な分離
2. モデルタイプごとの要件を明示的に定義可能
3. 新しいモデルタイプの追加が容易
4. テストの構造化が改善

### 検討された代替案

1. ~~単一インターフェースでの統一~~

   - コンポーネントの違いを隠蔽しようとすると、かえって複雑化
   - 却下理由: モデルタイプごとの特性を活かせない

2. ~~三階層以上の構造~~
   - フレームワーク層を追加する案
   - 却下理由: 現状の要件には過剰な複雑さ

### 影響範囲

1. **テストへの影響**

   - BDD テストのシナリオ構造の変更が必要
   - モデルタイプごとの明確なテストケース定義が可能に

2. **既存コードへの影響**
   - `ModelLoad`クラスの大幅なリファクタリングが必要
   - 移行期間中の互換性維持が必要

### 次のステップ

1. 基底ローダークラスの実装
2. 各モデルタイプのローダー実装
3. テストケースの更新
4. 既存コードの段階的移行

## 2025/04/05: ドキュメント整理とルールの明確化

- **決定**: プロジェクト全体のドキュメント (`docs_image-annotator-lib`) および開発ルール (`.cursor/rules`) を整理・更新し、開発の指針を明確化する。
- **理由**:
  - 開発効率とコード品質の一貫性を向上させるため。
  - 新規参画者や将来のメンテナンス担当者がプロジェクトを理解しやすくするため。
  - AI (Roo) が遵守すべきルールを明確にし、より適切なコード生成・編集を促すため。
- **実施内容**:
  - `docs_image-annotator-lib` を Diátaxis フレームワーク (Getting Started, Developer Guide, Explanation, Reference) に基づいて再構成。
  - `.cursor/rules` 内の各ルールファイル (`coding-rules.mdc`, `directory-structure-rules.mdc`, `encapsulation-rules.mdc`, `project-documentation-rules.mdc`, `image-annotator-lib.mdc`) の内容をレビューし、最新の状況に合わせて更新・明確化。
- **明確化された主要ルール**:
  - **コーディング規約**: モダンな型ヒント (`list`, `dict` 等) の使用、`Any` の回避、エラー抑制コメント (`# type: ignore`, `# noqa`) の禁止、半角文字の使用徹底。
  - **カプセル化**: 他クラス内部変数 (`_` プレフィックス) への直接アクセス禁止、Tell Don't Ask 原則の遵守、ゲッター/セッターの原則禁止。
  - **ドキュメンテーション**: Google スタイル Docstring の記述、モジュールコメントの追加、Todo Tree 拡張タグ (`TODO`, `FIXME` 等) の活用、関連ドキュメントの同期更新。
  - **`image-annotator-lib` 固有ルール**: `BaseAnnotator.predict()` メソッドのオーバーライド禁止、`AnnotationResult` 形式の遵守。
  - **AI 連携**: 原則違反時のユーザーへの通知義務、問題解決プロセスの定義 (3 回試行ルール、エスカレーション)。
- **影響**:
  - 全ての開発者 (人間および AI) は、整理されたドキュメントと明確化されたルールを参照し、遵守する必要がある。
  - メモリーバンク (`activeContext.md`, `productContext.md`, `progress.md`) に関連情報を反映済み。

## 2024-04-06: モジュール構造のリファクタリング (循環参照解消と責務分離)

- **背景:** `core/utils.py` に設定読み込み、ロギング、定数定義、ファイル I/O など多くの責務が混在し、モジュール間の依存関係が複雑化していた。特に、共通定数 `DEFAULT_PATHS` とロガー設定 `setup_logger` が相互に依存しうる構造になっており、潜在的な循環参照のリスクがあった。
- **変更点:**
  - `core/config.py` に設定関連の機能を集約:
    - `utils.py` から共通定数 (`DEFAULT_PATHS`, `DEFAULT_TIMEOUT` など) を `config.py` の先頭に移動。
    - `utils.py` から設定ファイル操作関数 (`load_model_config`, `save_model_size`) を `config.py` に移動。
    - `ModelConfigRegistry` も `config.py` に配置。
  - `core/utils.py` の責務を整理:
    - ロギング設定関数 (`setup_logger`) は `utils.py` に残す。
    - ファイル I/O、ネットワーク処理、画像処理などの汎用ユーティリティ関数のみを `utils.py` に残す。
  - 依存関係の整理:
    - `utils.py` は `config.py` から定数をインポートする一方向の依存関係に変更。
    - `config.py` は `utils.py` に依存しないようにした。
  - 各モジュールのインポート文を修正し、循環参照を解消。
- **理由:**
  - 単一責任の原則に基づき、`utils.py` の責務を分割・明確化するため。
  - 循環参照のリスクを根本的に解消するため。
  - ファイル数を過度に増やさずに構造を改善するため (当初検討した `constants.py` や `logger.py` の新規作成は見送った)。
- **影響:**
  - `load_model_config` などの関数のインポート元が変更された。 (`core.utils` -> `core.config`)
  - モジュール間の依存関係が整理され、可読性と保守性が向上した。

## 2024-04-07: 設定管理クラス (`ModelConfigRegistry`) の役割変更

- **背景:** 当初 `ModelConfigRegistry` は各モデルのコード上のデフォルト値を登録・管理する役割だったが、設定値の取得ロジックが複雑化していた。また、クラスの責務として、単なるデフォルト値保持よりも、設定ファイル全体を管理する方が適切であると判断された。
- **変更点:**
  - `ModelConfigRegistry` クラスを再設計:
    - 設定ファイル (`annotator_config.toml`) から読み込んだ**全ての設定データ**をインスタンス変数 (`_config_data`) に保持するように変更。
    - 設定値を取得するための `get(model_name, key, default)` メソッドを実装。このメソッドは `_config_data` から値を探し、なければ指定されたデフォルト値を返す。
    - 各モデルファイルでのデフォルト値登録 (`register` メソッドと呼び出し) を**廃止**。
  - インスタンス共有:
    - `config.py` モジュールレベルで `ModelConfigRegistry` の共有インスタンス (`config_registry`) を作成し、初期化時に設定ファイルをロードするように変更。
  - 利用箇所の変更:
    - `BaseAnnotator` およびサブクラスの `__init__` で、`self.config` 属性の使用を廃止。
    - 代わりに、インポートした共有インスタンス `config_registry` の `get` メソッドを使用して、必要な設定値(`model_path`, `device`, `chunk_size`, `tag_threshold` など)を取得するように統一。
- **理由:**
  - 設定の読み込みとアクセスを一元管理するため (`ModelConfigRegistry` インスタンス)。
  - 各モデルクラスでのデフォルト値登録処理を不要にし、コードをシンプルにするため。
  - `BaseAnnotator` とサブクラスでの設定値取得方法を統一するため。
  - `ModelConfigRegistry` の責務をより明確にするため (単なるデフォルト値保持 → 設定全体の管理とアクセス提供)。
- **影響:**
  - `ModelConfigRegistry` のインターフェースが変更された (`register` 廃止、`get` の役割変更)。
  - `BaseAnnotator` およびサブクラスでの設定値の取得方法が `config_registry.get()` に統一された。
  - 各モデルファイル末尾の `register` 呼び出しが不要になった。
