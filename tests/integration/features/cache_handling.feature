# language: en
Feature: モデルキャッシュ機能
    スコアリングライブラリは、モデルをキャッシュして効率的に再利用できるべきです

    Scenario: モデルのキャッシュと復元
        Given スコアリングライブラリが初期化されている
        And キャッシュディレクトリが存在する
        When 同じモデルを2回ロードする
        Then 2回目のロードは高速であるべき
        And キャッシュからモデルが取得されるべき

    Scenario: 複数モデルの初期化とキャッシュ
        Given スコアリングライブラリが初期化されている
        And 複数のモデル設定が存在する
        When 複数のモデルをロードする
        Then すべてのモデルが正常にキャッシュされるべき

    Scenario: キャッシュからの復元
        Given キャッシュされたモデルが存在する
        When モデルを解放して再度ロードする
        Then キャッシュから高速に復元されるべき

    Scenario: 無効なモデル名でのエラー処理
        Given スコアリングライブラリが初期化されている
        When 存在しないモデル名でロードを試みる
        Then 適切なエラーが発生するべき

    Scenario: キャッシュと解放の繰り返しによるメモリ管理
        Given スコアリングライブラリが初期化されている
        When モデルを繰り返しキャッシュと解放する
        Then システムリソースが適切に管理されるべき

    Scenario: キャッシュのクリア
        Given キャッシュされたモデルが存在する
        When キャッシュをクリアする
        Then キャッシュディレクトリは空になるべき
        And 次回のモデルロードは新規ロードになるべき