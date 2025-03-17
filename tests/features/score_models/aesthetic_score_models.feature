Feature: 美的スコア評価モデル

    Background:
    特定のHuggingFaceモデルを活用して画像の美的価値を数値化するモデル群です。

    Scenario: Aesthetic Shadow V1 モデル
        Given "shadowlilac/aesthetic-shadow" モデルが利用可能である
        When 画像を評価すると
        Then 0-1の範囲のスコアが返される
        And スコアに基づいて適切なタグが付与される

    Scenario: Aesthetic Shadow V2 モデル
        Given "NEXTAltair/cache_aestheic-shadow-v2" モデルが利用可能である
        When 画像を評価すると
        Then 0-1の範囲のスコアが返される
        And スコアに基づいて適切なタグが付与される

    Scenario Outline: CAFE Aesthetic モデルのスコア変換
        Given "cafeai/cafe_aesthetic" モデルが利用可能である
        When 画像の評価結果が <スコア> である
        Then 変換されたタグは "[CAFE]score_<整数値>" となる

        Examples:
            | スコア | 整数値 |
            | 0.67   | 6      |
            | 0.25   | 2      |
            | 0.98   | 9      |

    Scenario Outline: スコアのしきい値による分類
        Given Aesthetic Shadow モデルが初期化されている
        When 画像の評価結果が <スコア> である
        Then 評価タグは <タグ> となる

        Examples:
            | スコア | タグ             |
            | 0.90   | very aesthetic   |
            | 0.71   | very aesthetic   |
            | 0.60   | aesthetic        |
            | 0.45   | aesthetic        |
            | 0.30   | displeasing      |
            | 0.27   | displeasing      |
            | 0.10   | very displeasing |