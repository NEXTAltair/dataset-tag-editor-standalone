# pytest-bdd メモ

## 概要
- バージョン: 8.1.0 (2024年12月6日リリース)
- Gherkin言語のサブセットを実装したBDDテストフレームワーク
- pytestの機能と統合されており、別途ランナーを必要としない

## 主な特徴
1. pytestとの統合
   - pytestの機能と柔軟性を活用
   - ユニットテストと機能テストの統合が可能
   - CI設定の簡素化

2. fixture活用
   - pytestのfixtureをBDDのステップで再利用可能
   - 依存性注入によるテストセットアップの再利用
   - コンテキストオブジェクトの管理が不要

3. ブラウザテスト対応
   - pytest-splinterとの連携でブラウザテスト可能

## pytest-bdd の利点
- **可読性の高いテストシナリオ**: Featureファイルに自然言語に近いGherkin形式でテストシナリオを記述できるため、非技術者にも理解しやすいテスト仕様書となる。
- **BDDプラクティスの導入**: 行動駆動開発 (BDD) の考え方に基づいたテスト設計・実装が容易になり、開発チーム全体の共通認識を醸成しやすくなる。
- **ドキュメントとしての側面**: Featureファイル自体がテスト仕様書として機能するため、別途ドキュメントを作成する手間を削減できる。常に最新の状態に保たれたドキュメントとして活用可能。
- **コミュニケーションの円滑化**: Featureファイルを共有することで、開発者、テスター、プロダクトオーナーなど、関係者間のコミュニケーションが円滑になり、認識の齟齬を減らすことができる。

## インストール方法
```bash
pip install pytest-bdd
```

## 基本概念

### 1. ステップリファレンス (Featureファイルに記述)
BDDのシナリオで使用される基本的なステップの種類です。**Featureファイル(`.feature`)にGherkin言語で記述**します。

1. Given(前提条件)
   - テストの初期状態を設定
   - 例:`Given サポートしていない画像フォーマット(例: 非画像ファイル)が入力される`
   - システムの状態、データの準備などを定義

2. When(アクション)
   - テスト対象の操作を実行
   - 例:`When 画像評価機能が実行される`
   - ユーザーアクション、システム操作などを定義

3. Then(期待結果)
   - テスト結果の検証
   - 例:`Then システムは TypeError など、適切な例外を発生させる`
   - 期待される結果、状態変化などを検証

4. And/But(追加条件)
   - 既存のステップに条件を追加
   - 例:`And ...`
   - Given、When、Thenの補足として使用

### 2. ステップ定義 (テストコードに記述)
ステップリファレンスで定義した各ステップの具体的な処理を**Pythonのテストコード(`.py`)に記述**します。

- **`@given`, `@when`, `@then` デコレータ** を使用して、Featureファイルのステップリファレンスとテストコード内の関数を紐付けます。
- デコレータの引数には、Featureファイルに記述したステップリファレンスのテキスト(または正規表現)を指定します。

### 3. パラメータ化されたシナリオ
```gherkin
Scenario Outline: パラメータ化されたシナリオ
  Given <入力>が用意されている
  When 処理を実行する
  Then 結果は<出力>となる

  Examples:
    | 入力 | 出力 |
    | 値1  | 結果1 |
    | 値2  | 結果2 |
```

```python
@given(parsers.parse('{input}が用意されている'))
def step_given_input(input):
    return setup_input(input)
```

## 実装手順

### 1. Featureファイルの作成

1.  `tests/features/` ディレクトリに `.feature` ファイルを作成
2.  Feature名とシナリオを定義
3.  Given/When/Thenステップを記述

#### Featureファイルの構成

*   1つの`.feature`ファイルには、1つのFeatureを記述することが推奨されます。
    *   **理由**:
        *   **可読性**: 各ファイルが単一の機能に焦点を当てることで、内容が理解しやすくなります。
        *   **保守性**: 機能ごとにファイルが分割されているため、変更や修正の影響範囲を局所化できます。
        *   **明確性**: Featureファイルが、システムの特定の振る舞いを表すドキュメントとしての役割を明確に果たします。
*   Gherkinの仕様では、複数のFeatureを記述することも可能ですが、推奨されません。ファイルが肥大化し、理解しにくくなる可能性があります。
*   **フィーチャーファイルの分け方の基準**:
    *   関連するシナリオをまとめて1つのファイルにします。
    *   機能やテストの種類ごとにファイルを分けます。
    *   `scorer-wrapper-lib`での例:
        *   `model_loading.feature`: モデルのロードに関するシナリオ
        *   `image_scoring.feature`: 画像のスコアリングに関するシナリオ
        *   `integration.feature`: 統合テストのシナリオ
        *   `error_handling.feature`: エラー処理に関するシナリオ
*   **Featureの分け方の基準**:
    *   Featureは、ひとまとまりの機能、またはユーザーから見たシステムの振る舞いを記述します。

例:
```gherkin
# tests/features/integration.feature
Feature: スコアラー統合テスト
  BaseScorerを継承した各評価モデルの基本機能を検証する

  Scenario Outline: モデルロードから画像評価までの統合テスト
    Given <スコアラー>の有効な設定ファイルが存在し、適切に前処理された画像が用意されている
    When システムがモデルをロードし、画像の前処理および評価処理を連続して実行する
    Then 出力として、以下の情報が返される:
      | 生のスコア            | 整形済みスコア                  | 評価カテゴリ             |
      | 0.0 ~ 1.0 の float値  | 各モデル固有のスケールで変換された値  | <評価プレフィックス>_で始まる評価ラベル |

    Examples:
      | スコアラー | 評価プレフィックス |
      | AestheticShadow | v1 |
      | AestheticShadowV2 | v2 |
```

### 2. テストコードの作成
```python
# tests/test_aesthetic_shadow.py
from pytest_bdd import scenario, given, when, then, parsers
from scorer_wrapper_lib.score_models.aesthetic_shadow import AestheticShadow

@scenario("integration.feature", "モデルロードから画像評価までの統合テスト")
def test_integration():
    pass

@given("<スコアラー>の有効な設定ファイルが存在し、適切に前処理された画像が用意されている")
def setup_environment(aesthetic_shadow, mock_model, test_image_path):
    with patch.object(aesthetic_shadow, "model", mock_model):
        aesthetic_shadow.load_model()
    assert test_image_path.exists()
    return aesthetic_shadow

@when("システムがモデルをロードし、画像の前処理および評価処理を連続して実行する")
def execute_evaluation(setup_environment, test_image):
    result = setup_environment.predict(test_image)
    return result

@then(parsers.parse("出力として、以下の情報が返される:\n{expected_table}"))
def verify_output(execute_evaluation, expected_table):
    result = execute_evaluation
    assert "raw_scores" in result
    assert "formatted_score" in result
    assert "evaluation" in result
```

### 3. フィクスチャの定義
```python
# tests/conftest.py
@pytest.fixture
def test_image_path() -> Path:
    return Path(__file__).parent / "resources" / "img" / "test.jpg"

@pytest.fixture
def mock_model():
    mock = MagicMock()
    mock.return_value = [{"label": "hq", "score": 0.8}]
    return mock
```

## VSCode拡張機能の活用

### インストール方法
1. VSCodeの拡張機能マーケットプレイスを開く (Ctrl+Shift+X)
2. 検索バーに `vtenentes.bdd` と入力
3. "BDD - Cucumber/Gherkin Full Support" をインストール
4. VSCodeを再起動

### 主なコマンド

#### 1. Create Step
未実装のステップから、対応するPythonコードの雛形を自動生成します。

使い方:
1. Featureファイルで未実装のステップにカーソルを置く
2. コマンドパレットを開く (Ctrl+Shift+P)
3. "BDD: Create Step" を選択
4. 生成されたコードをテストファイルに貼り付け

生成されるコード例:
```python
@given("有効な設定ファイルが存在する")
def step_impl():
    """
    有効な設定ファイルが存在する
    """
    raise NotImplementedError
```

#### 2. Debug Scenario
シナリオ単位でデバッグ実行を行います。

使い方:
1. デバッグしたいシナリオにカーソルを置く
2. コマンドパレット → "BDD: Debug Scenario"
3. デバッグビューが開き、ブレークポイントで停止

デバッグ時の機能:
- 変数の値の確認
- ステップ実行
- 条件式の評価
- コールスタックの確認

#### 3. Find References
ステップの定義と使用箇所を素早く見つけます。

使い方:
1. Featureファイルのステップにカーソルを置く
2. コマンドパレット → "BDD: Find References"
3. 参照パネルに使用箇所が表示

表示される情報:
- ステップの定義場所
- 使用されているFeatureファイル
- 行番号とコンテキスト

#### 4. Run Scenario
特定のシナリオだけを実行します。

使い方:
1. 実行したいシナリオにカーソルを置く
2. コマンドパレット → "BDD: Run Scenario"
3. 統合ターミナルでテスト実行

オプション:
- `--gherkin-terminal-reporter`: Gherkin形式で結果表示
- `-v`: 詳細な実行ログを表示
- `--no-cov`: カバレッジ計測をスキップ

#### 5. Step Definition
ステップの実装箇所にジャンプします。

使い方:
1. Featureファイルのステップにカーソルを置く
2. コマンドパレット → "BDD: Step Definition"
   または F12 キー（Go to Definition）
3. 実装箇所にジャンプ

### キーボードショートカットの設定
`keybindings.json` に以下を追加することで、よく使う機能にショートカットを割り当てられます：

```json
{
    "key": "ctrl+shift+c",
    "command": "bdd.createStep",
    "when": "editorTextFocus && editorLangId == 'feature'"
},
{
    "key": "ctrl+shift+d",
    "command": "bdd.debugScenario",
    "when": "editorTextFocus && editorLangId == 'feature'"
},
{
    "key": "ctrl+shift+f",
    "command": "bdd.findReferences",
    "when": "editorTextFocus && editorLangId == 'feature'"
},
{
    "key": "ctrl+shift+r",
    "command": "bdd.runScenario",
    "when": "editorTextFocus && editorLangId == 'feature'"
}
```

### 便利な機能

#### コード補完
- Gherkinキーワードの自動補完
- 既存ステップの候補表示
- シナリオアウトラインのテーブル補完
- 変数名の補完

#### シンタックスハイライト
- Gherkinキーワードの色分け
- テーブルの罫線表示
- タグのハイライト
- エラー箇所の強調表示

#### スニペット
- シナリオの雛形
- シナリオアウトラインの雛形
- テーブル形式の雛形

#### その他の機能
- ステップの定義へのホバープレビュー
- 未実装ステップの警告表示
- ステップパラメータの型ヒント
- アウトラインの構造表示

## コマンド
- `pytest --gherkin-terminal-reporter`: Gherkin形式でテスト結果を表示
- `pytest -v`: 詳細なテスト結果を表示
- `pytest -k "シナリオ名"`: 特定のシナリオのみ実行

## ベストプラクティス
1. シナリオ名は具体的で理解しやすいものにする
2. ステップは再利用可能な粒度で作成する
3. フィクスチャを活用してテストのセットアップを簡潔にする
4. Docstringでステップの目的や引数を明確に記述する
5. エラーケースも含めてシナリオを作成する

## 注意点
- ステップ関数名は一意である必要がある
- パラメータ化されたステップでは型変換に注意
- 複雑なデータ構造はフィクスチャで提供する
- シナリオ間でステートを共有しない

## BDD (Behavior-Driven Development) とは

BDD（行動駆動開発）は、ソフトウェア開発において、システムの振る舞いを自然言語に近い形で記述し、開発者、テスター、ビジネス関係者間で共通の理解を形成するための手法です。

### BDDシナリオとは

BDDシナリオでは、ユーザーの行動やシステムの反応を「Given（前提）」「When（行動）」「Then（期待結果）」の形式で表現します。これにより、システムの振る舞いを明確にし、テストケースとして活用できます。

### BDDのメリット
- **要求仕様の明確化**: BDDは、システムの振る舞いを明確に記述することで、要求仕様の曖昧さを排除し、開発チームとビジネス関係者間の認識の齟齬を防ぎます。
- **テストの自動化**: BDDシナリオは、自動化されたテストケースとして実行できるため、リグレッションテストの効率化に貢献します。
- **ドキュメントの自動生成**: BDDシナリオは、システムの振る舞いを記述したドキュメントとしても機能します。
- **コミュニケーションの促進**: BDDは、共通の言語（Gherkinなど）を使用することで、開発チーム内のコミュニケーションを円滑にします。

### 具体的なファイルとディレクトリ構成例

```
scorer-wrapper-lib/
├── tests/
│   └── features/
│       ├── feature_overview.md      # 全体の機能概要と受け入れ基準
│       └── image_scoring.feature    # 画像評価に関するBDDシナリオ
```

#### 各ファイルの内容
- **feature_overview.md**
  - システム全体の目的、主要なユースケース、基本的な受け入れ基準を記述します。
  - 例: 「ユーザーがログインできる」「画像評価の結果が一定の範囲内で安定している」など。

- **image_scoring.feature**
  - 画像評価システムに対するシナリオを記述します。
  - 例:
    ```gherkin
    Feature: 画像評価
      画像に対して美的スコアを算出し、ユーザーにフィードバックを提供する

      Scenario: 単一画像の評価
        Given 適切に前処理された画像がある
        When ユーザーが画像評価機能を起動する
        Then 画像の評価スコアが0から10の範囲で返される

      Scenario: バッチ処理でのランキング
        Given 複数の画像がアップロードされている
        When システムが全画像を評価する
        Then スコアに基づく画像の順位付けが一貫して行われる
    ```
### BDDの進め方

1. **BDDシナリオの骨子作成**
   - まず、`feature_overview.md` のようなファイルを作成し、システム全体の機能と受け入れ基準をまとめます。

2. **Featureファイルの準備**
   - ユーザーの主要なユースケースごとに `.feature` ファイル（例：`image_scoring.feature`）を作成します。
   - 各ファイルには、具体的な「Given-When-Then」のシナリオを詳細に記述します。

3. **レビューとフィードバックの実施**
   - 作成したBDDシナリオを開発チームやビジネス側とレビューし、内容の正確性や網羅性を確認します。

4. **テストコードの作成**
    -  `.feature` ファイルに記述されたシナリオに対応するテストコードを、`pytest-bdd` を利用して記述します。

### まとめ

BDDシナリオの詳細化は、今後のTDD/BDD開発の基盤となるため、最初に取り組むべき重要な作業です。
上記のファイル群を用意し、各シナリオを細部まで記述することで、実装やテストの指針を明確にすることができます。