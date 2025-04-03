Feature: リソース管理機能
    ユーザーストーリー：
    データサイエンティストとして、
    様々なソースからモデルやファイルを簡単に利用できるようにしたい。
    それによって、効率的にモデルの実験や評価ができるようになる。

    Background:
        Given アプリケーションを起動している
        And ログレベルがDEBUGに設定されている

    # リソースアクセスのメインシナリオ
    Scenario Outline: リソースアクセスの基本パターン
        Given リソース<source_type>が<location>に存在する
        And キャッシュの状態が<cache_state>
        When システムがリソースパスを解決する
        Then <expected_action>が実行される
        And 保存されているリソースのパス<expected_path>を返す

        Examples:
            | source_type | location       | cache_state | expected_action | expected_path |
            | local_file  | local_image    | none        | read_local      | local_image   |
            | local_zip   | local_archive  | none        | extract_zip     | local_dir     |
            | remote_url  | remote_image   | empty       | download        | cache_image   |
            | remote_url  | remote_image   | exists      | use_cache       | cache_image   |
            | remote_zip  | remote_archive | empty       | download_zip    | cache_dir     |

    # エラー処理シナリオ
    Scenario Outline: リソースアクセスのエラー処理
        Given リソース<source_type>が<location>に存在する
        And アクセス状態が<condition>
        When システムがリソースパスを解決する
        Then <error_type>エラーが発生する

        Examples:
            | source_type | location     | condition | error_type     |
            | local_file  | invalid_path | missing   | file_not_found |
            | remote_url  | invalid_url  | invalid   | request_error  |

    # ユーティリティ機能シナリオ
    Scenario: システムログの記録
        When アプリケーションが重要な操作を実行する
        Then その操作の詳細が適切にログに記録される
        And ログエントリにはタイムスタンプと操作の種類が含まれる

    Scenario: 設定ファイルの管理
        Given 設定ファイルにパラメーターが定義されている
        When 設定ファイルの読み込みを要求される
        Then 設定ファイルから正しいパラメーターが読み込まれる

    Scenario: 画像のハッシュ計算
        Given ユーザーが画像ファイルを使用しようとする
        When システムが画像のpHashを計算する
        Then システムは有効なpHash文字列を返す

    # モデル管理シナリオ
    Scenario: ONNX Taggerモデルのダウンロード
        Given Hugging Face Hubにモデルリポジトリが存在する
        And そのリポジトリに必要なファイルが含まれる
        When システムがONNX Taggerモデルのダウンロードを試みる
        Then モデルファイルとラベルファイルのパスが返される
        And Hugging Face Hubのダウンロード関数が適切な引数で呼び出される

    Scenario: モデルサイズの保存
        Given モデルの設定が存在する
        And モデルのサイズが計算されている
        When システムがモデルのサイズを保存する
        Then モデルの推定サイズが保存される
