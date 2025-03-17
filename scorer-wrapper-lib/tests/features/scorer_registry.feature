Feature: スコアラーレジストリの機能
    スコアラーレジストリの各機能の単体テストを行う

    Scenario: モジュールファイルのリスト取得
        Given テスト用のモジュールディレクトリが存在する
        When list_module_files関数を呼び出す
        Then 指定したディレクトリ内のPythonファイル一覧が取得できる

    Scenario: モジュールのインポート
        Given テスト用のモジュールファイルが存在する
        When import_module_from_file関数を呼び出す
        Then モジュールが正常にインポートされる

    Scenario: サブクラスの再帰的取得
        Given テスト用の基底クラスとサブクラスが存在する
        When recursive_subclasses関数を呼び出す
        Then 全てのサブクラスが取得できる

    Scenario: 利用可能なクラスの収集
        Given テスト用のスコアラークラスを含むディレクトリが存在する
        When gather_available_classes関数を呼び出す
        Then 利用可能なスコアラークラスの辞書が取得できる

    Scenario: コアクラスの収集
        Given コアモジュールが存在する
        When gather_core_classes関数を呼び出す
        Then コアクラスの辞書が取得できる

    Scenario: 設定からのモデル登録
        Given テスト用のモデル設定が存在する
        And 利用可能なクラスとコアクラスが準備されている
        When register_models_from_config関数を呼び出す
        Then モデルが正しく登録される

    Scenario: スコアラーの登録
        Given テスト用のスコアラー設定が存在する
        When register_scorers関数を呼び出す
        Then スコアラーが正しく登録される

    Scenario: レジストリの取得
        Given スコアラーが登録されている
        When get_registry関数を呼び出す
        Then 登録されたスコアラーの辞書が取得できる

    Scenario: 利用可能なスコアラーの一覧取得
        Given 複数のスコアラーが登録されている
        When list_available_scorers関数を呼び出す
        Then 登録されているスコアラー名のリストが取得できる

    Scenario: 無効なディレクトリからのモジュールファイルリスト取得
        Given 存在しないディレクトリが指定される
        When list_module_files関数を呼び出す
        Then 空のリストが返される

    Scenario: 無効なファイルからのモジュールインポート
        Given 存在しないモジュールファイルが指定される
        When import_module_from_file関数を呼び出す
        Then 適切なエラーが発生する