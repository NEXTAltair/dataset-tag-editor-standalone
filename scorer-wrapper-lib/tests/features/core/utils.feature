Feature: ユーティリティ関数のテスト
    scorer_wrapper_lib.core.utils モジュールの機能を検証するためのテスト

    Scenario: ローカルファイルのロード
        Given ローカルに存在するファイルがある
        When load_file関数でそのパスを指定する
        Then ファイルの絶対パスが返される

    Scenario: URLからのファイルダウンロード
        Given 有効なURLがある
        When load_file関数でそのURLを指定する
        Then ファイルがダウンロードされローカルパスが返される

    Scenario: HuggingFace Hubからのリポジトリダウンロード
        Given 有効なHuggingFace Hubのリポジトリ名がある
        When load_file関数でそのリポジトリ名を指定する
        Then リポジトリがダウンロードされローカルパスが返される

    Scenario: 無効なパスからのファイル取得
        Given 存在しないパスがある
        When load_file関数でそのパスを指定する
        Then 適切な例外が発生する

    Scenario: ロガーのセットアップ
        Given アプリケーション名がある
        When setup_logger関数でロガーを初期化する
        Then 標準出力とファイルにログが記録される

    Scenario: TOMLからのモデル設定の読み込み
        Given モデル設定ファイルが存在する
        When load_model_config関数を呼び出す
        Then 正しいモデル設定辞書が返される

    Scenario: コンソール出力のキャプチャ
        Given アプリケーションが標準出力に書き込む
        When ConsoleLogCaptureクラスでコンソール出力をキャプチャする
        Then 出力がログファイルに記録される