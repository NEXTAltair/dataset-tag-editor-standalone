# ハウツーガイド: テストの実行方法

このガイドでは、`image-annotator-lib` プロジェクトのテストを実行する方法について説明します。テストは `pytest` および `pytest-bdd` を使用して実装されています。

## 1. 準備

テストを実行する前に、開発用の依存関係がインストールされていることを確認してください。`pyproject.toml` があるプロジェクトルートで以下のコマンドを実行します。

```bash
# 仮想環境をアクティベートしていることを確認
# (例: source .venv/bin/activate または .venv\Scripts\activate)

# 開発用依存関係を含むライブラリをインストール
uv pip install -e .[dev]
```

## 2. テストの実行

テストは `pytest` コマンドを使用して実行します。プロジェクトのルートディレクトリで以下のコマンドを実行してください。

### 2.1. 全てのテストを実行

```bash
pytest
```

これにより、`tests/` ディレクトリ（または pytest が自動検出する他の場所）にある全てのテストが実行されます。

### 2.2. 詳細な出力で実行

`-v` オプションを付けると、各テスト関数の名前と結果が詳細に表示されます。

```bash
pytest -v
```

### 2.3. BDD (Gherkin) 形式での結果表示

`pytest-bdd` を使用しているため、BDD のフィーチャーファイルに基づいた結果を表示できます。

```bash
pytest --gherkin-terminal-reporter
```

これにより、どのシナリオのどのステップが成功/失敗したかが Gherkin 形式で表示され、可読性が向上します。

### 2.4. 特定のファイルまたはディレクトリのテストを実行

特定のファイルやディレクトリを指定してテストを実行することも可能です。

```bash
# 特定のファイルを実行
pytest tests/test_api.py

# 特定のディレクトリを実行
pytest tests/models/
```

### 2.5. 特定のテスト関数またはシナリオを実行

`-k` オプションを使用して、名前（関数名やシナリオ名の一部）にマッチするテストのみを実行できます。

```bash
# 関数名に 'annotate' を含むテストを実行
pytest -k annotate

# シナリオ名に 'model loading' を含む BDD テストを実行
pytest -k "model loading"
```

(注意: シナリオ名にスペースが含まれる場合は引用符で囲みます。)

### 2.6. テストカバレッジの計測

`pytest-cov` がインストールされていれば、テストカバレッジを計測できます。

```bash
pytest --cov=src/image_annotator_lib tests/
```

これにより、`src/image_annotator_lib` ディレクトリ内のコードがテストによってどれだけカバーされているかのレポートが出力されます。

## 3. VSCode でのテスト実行 (pytest-bdd 拡張機能)

`pytest-bdd.md` で説明されているように、VSCode 拡張機能 "BDD - Cucumber/Gherkin Full Support" (`vtenentes.bdd`) を使用すると、`.feature` ファイルから直接シナリオを実行したりデバッグしたりできます。

- **Run Scenario**: カーソルがあるシナリオを実行します (コマンドパレット: `BDD: Run Scenario`)。
- **Debug Scenario**: カーソルがあるシナリオをデバッグ実行します (コマンドパレット: `BDD: Debug Scenario`)。

詳細については、`docs/pytest-bdd.md` (または移植後の `EXPLANATION/testing_framework.md` など) を参照してください。
