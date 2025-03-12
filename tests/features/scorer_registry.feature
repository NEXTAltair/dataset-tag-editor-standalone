Feature: スコアラーレジストリの機能
    スコアラーレジストリの主要機能のテスト

    Scenario: モジュールの検出とインポート
        Given モジュールディレクトリが存在する
        And Pythonモジュールファイルを検索した結果がある
        Then 指定したディレクトリ内のPythonファイル一覧が取得できる
        When モジュールをインポートする
        Then モジュールが正常にインポートされる
        When 存在しないモジュールのインポート
        Then 存在しないモジュールのインポートが失敗したことを検証

    Scenario: スコアラークラスの収集
        Given スコアラークラスを含むディレクトリが存在する
        And 利用可能なスコアラークラスが収集されている
        Then 利用可能なスコアラークラスの辞書が取得できる
        And 基本実装と派生クラスが含まれている

    Scenario: スコアラーの設定と登録
        Given スコアラー設定が存在する
        And 利用可能なスコアラークラスが収集されている
        When スコアラーを登録する
        Then スコアラーが正しく登録される
        When レジストリを取得する
        Then 登録されたスコアラーの辞書が取得できる
        And 登録されているスコアラー名のリストが取得できる