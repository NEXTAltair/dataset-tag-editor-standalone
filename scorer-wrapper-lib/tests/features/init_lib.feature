Feature: ライブラリ初期化
  ライブラリの初期化処理とスコアラー登録の検証を行う

  Scenario Outline: 正常なモデルで初期化
    Given 有効なモデル設定が存在する
    When ライブラリが <model_name> を使用してスコアラーを初期化する
    Then 利用可能なスコアラー一覧に <model_name> が含まれていることを確認する

    Examples:
      | model_name          |
      | aesthetic_shadow_v1 |
      | aesthetic_shadow_v2 |
      | cafe_aesthetic      |
      | ImprovedAesthetic   |
      | WaifuAesthetic      |

  Scenario: 無効なモデルで初期化
    Given 無効なモデル名が指定される
    When ライブラリが "unknown_model" を使用してスコアラーを初期化する
    Then エラーが発生することを確認する
