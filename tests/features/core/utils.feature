Feature: ユーティリティ関数のテスト
    scorer_wrapper_lib.core.utils モジュールの機能を検証するためのテスト

    # ローカルファイルとURL関連のテスト
    Scenario: ローカルファイルの取得
        Given ローカルに保存されたファイルがある
        When そのパスを指定してファイルにアクセスする
        Then 絶対パスのPathオブジェクトが返される

    Scenario: URLからの新規ファイルダウンロード
        Given 有効なURLがある
        When そのURLを指定してファイルにアクセスする
        Then ファイルがダウンロードされる
        And ダウンロードしたファイルはモジュールのmodelsディレクトリにキャッシュされる
        And ファイル名はURLのパス部分から抽出される
        And 絶対パスのPathオブジェクトが返される

    Scenario: 既にダウンロード済みのURLからのファイル取得
        Given 有効なURLがある
        And 以前にダウンロードしたファイルがキャッシュに存在する
        When そのURLを指定してファイルにアクセスする
        Then 新たなダウンロードは行われない
        And 絶対パスのPathオブジェクトが返される

    # エラー処理関連のテスト
    Scenario: 存在しないローカルファイルへのアクセス
        Given 存在しないパスがある
        When そのパスを指定してファイルにアクセスする
        Then ファイルアクセスに失敗する
        And 適切なエラーメッセージが表示される

    Scenario: 無効なURLからのファイルアクセス
        Given 無効なURLがある
        When そのURLを指定してファイルにアクセスする
        Then ファイルアクセスに失敗する
        And 適切なエラーメッセージが表示される

    Scenario: ファイルアクセスの総合的なエラーハンドリング
        Given 無効なパスまたはURLがある
        When そのパスを指定してファイルにアクセスする
        Then ファイルアクセスに失敗する
        And エラーメッセージには元のパスが含まれる

    # その他のユーティリティ関数テスト
    Scenario: ロガーのセットアップ
        Given アプリケーション名がある
        When ロガーをセットアップする
        Then 標準出力にログが記録される
        And ログファイルにもログが記録される

    Scenario: 設定ファイルからのモデル設定の取得
        Given モデル設定ファイル(TOML形式)が存在する
        When 設定ファイルから設定値にアクセスする
        Then 正しいモデル設定辞書が返される
        And 設定値はキャッシュされ再利用される