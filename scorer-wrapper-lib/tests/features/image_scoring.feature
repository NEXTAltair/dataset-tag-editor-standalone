Feature: 画像評価システム
  画像に対して美的スコアが返されることを検証する

  Scenario Outline: 単一画像の評価
    Given 有効な画像が用意されている
    When ユーザーが画像評価機能を起動し、<model_name> を指定してその画像を入力する
    Then システムは以下の結果を返す: <model_name> <model_output> <score_tag>

    Examples:
      | model_output | model_name             | score_tag         |
      | list         | aesthetic_shadow_v1    | aesthetic         |
      | list         | aesthetic_shadow_v2    | aesthetic         |
      | list         | cafe_aesthetic         | [CAFE]score_6     |
      | list         | ImprovedAesthetic      | [IAP]score_5      |
      | list         | WaifuAesthetic         | [WAIFU]score_0    |
      | list         | ImageReward            | [IR]score_0       |


  Scenario Outline: 複数画像の評価
    Given 複数の有効な画像が用意されている
    When ユーザーが画像評価機能を起動し、<model_name> を指定してそれらの画像を入力する
    Then システムは <num_images> 件の評価結果を返す
    And 1番目の画像の評価結果は <model_name> <model_output> <score_tag1> である
    And 2番目の画像の評価結果は <model_name> <model_output> <score_tag2> である

    Examples:
      | model_name          | model_output | num_images | score_tag1     | score_tag2     |
      | aesthetic_shadow_v1 | list         | 2          | aesthetic      | aesthetic      |
      | aesthetic_shadow_v2 | list         | 2          | aesthetic      | aesthetic      |
      | cafe_aesthetic      | list         | 2          | [CAFE]score_6  | [CAFE]score_6  |
      | ImprovedAesthetic   | list         | 2          | [IAP]score_5   | [IAP]score_5   |
      | WaifuAesthetic      | list         | 2          | [WAIFU]score_0 | [WAIFU]score_0 |
      | ImageReward         | list         | 2          | [IR]score_0    | [IR]score_0    |

  Scenario Outline: 複数モデルでの単一画像の評価
    Given 有効な画像が用意されている
    When ユーザーが画像評価機能を起動し、以下のモデルリストを指定してその画像を入力する:
      | model1      | model2         |
      | <model1>    | <model2>       |
    Then システムは2件のモデル評価結果を返す
    And 1番目のモデルの評価結果は <model1> <output_type> <score_tag1> である
    And 2番目のモデルの評価結果は <model2> <output_type> <score_tag2> である

    Examples:
      | model1              | model2              | output_type | score_tag1    | score_tag2      |
      | aesthetic_shadow_v1 | cafe_aesthetic      | list        | aesthetic     | [CAFE]score_6   |
      | ImprovedAesthetic   | WaifuAesthetic      | list        | [IAP]score_5  | [WAIFU]score_0  |
      | aesthetic_shadow_v2 | ImprovedAesthetic   | list        | aesthetic     | [IAP]score_5    |

  Scenario Outline: 複数モデルでの複数画像の評価
    Given <num_images>枚の有効な画像が用意されている
    When ユーザーが画像評価機能を起動し、以下のモデルリストを指定してそれら画像を入力する:
      | model1      | model2         |
      | <model1>    | <model2>       |
    Then システムは<num_images>枚の画像に対して、各モデルの評価結果を返す
    And 1番目のモデルの1番目の画像の評価結果は <model1> <output_type> <score1_1> である
    And 1番目のモデルの2番目の画像の評価結果は <model1> <output_type> <score1_2> である
    And 2番目のモデルの1番目の画像の評価結果は <model2> <output_type> <score2_1> である
    And 2番目のモデルの2番目の画像の評価結果は <model2> <output_type> <score2_2> である

    Examples:
      | model1              | model2            | num_images | output_type | score1_1  | score1_2  | score2_1      | score2_2      |
      | aesthetic_shadow_v1 | cafe_aesthetic    | 2          | list        | aesthetic | aesthetic | [CAFE]score_6 | [CAFE]score_6 |
      | ImprovedAesthetic   | WaifuAesthetic    | 2          | list        | [IAP]score_5 | [IAP]score_5 | [WAIFU]score_0 | [WAIFU]score_0 |

  Scenario: 不正な入力画像の場合のエラーハンドリング
    Given サポートされていない画像フォーマットまたは破損した画像が提供される
    When ユーザーが画像評価機能を起動する
    Then システムは適切なエラーメッセージを表示するか、例外を発生させる
