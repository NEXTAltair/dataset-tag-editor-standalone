Feature: キャッシュ処理の統合テスト
  スコアラーライブラリの複数モデル/インスタンス間のキャッシュ機能の統合テスト

  # 注: 基本的なキャッシュ機能のユニットテストはcore/base.featureで実施
  # このfeatureファイルでは複数モデルや実際のライブラリ関数を使った統合的な観点でテスト

  Scenario: 複数モデルの初期化とキャッシュ
    Given 有効なモデル設定が存在する
    When ライブラリが "ImprovedAesthetic" を使用してスコアラーを初期化する
    And ライブラリが "WaifuAesthetic" を使用して別のスコアラーを初期化する
    And 両方のスコアラーの load_or_restore_model メソッドを呼び出す
    And 両方のスコアラーの cache_to_main_memory メソッドを呼び出す
    Then 両方のスコアラーのモデルが正常にキャッシュされていることを確認する

  Scenario: 無効なモデル名でのエラー処理
    Given 有効なモデル設定が存在する
    When ライブラリが "存在しないモデル" を使用してスコアラーを初期化しようとする
    Then 適切なエラーが発生することを確認する

  Scenario: キャッシュと解放の繰り返しによるメモリ管理
    Given 有効なモデル設定が存在する
    When ライブラリが "ImprovedAesthetic" を使用してスコアラーを初期化する
    And スコアラーの load_or_restore_model メソッドを呼び出す
    And スコアラーの cache_to_main_memory メソッドを呼び出す
    And スコアラーの release_model メソッドを呼び出す
    And 再度ライブラリが "cafe_aesthetic" を使用して別のスコアラーを初期化する
    And 2つ目のスコアラーの load_or_restore_model メソッドを呼び出す
    Then システムリソースが適切に管理されていることを確認する