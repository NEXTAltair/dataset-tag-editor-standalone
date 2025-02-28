Feature: キャッシュ処理
  モデルのキャッシュと復元機能のテスト

  Scenario: モデルのキャッシュと復元
    Given 有効なモデル設定が存在する
    When ライブラリが "ImprovedAesthetic" を使用してスコアラーを初期化する
    And スコアラーの load_or_restore_model メソッドを呼び出す
    And スコアラーの cache_to_main_memory メソッドを呼び出す
    And スコアラーの restore_from_main_memory メソッドを呼び出す
    Then スコアラーのモデルが None でないことを確認する