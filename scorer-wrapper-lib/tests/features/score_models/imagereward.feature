Feature: Image Reward Model
    BLIPをベースにした画像の美的評価モデル

    Scenario: ImageRewardモデルの初期化
        Given ImageRewardモデルの設定が利用可能である
        When ImageRewardScorerが初期化される
        Then モデルが正常に初期化される

    Scenario: 画像の評価
        Given 初期化されたImageRewardモデルが存在する
        When 画像が評価される
        Then 0-10の範囲のスコアが返される
        And スコアに基づいたタグが付与される

    Scenario Outline: スコアタグ生成
        Given 初期化されたImageRewardScorerインスタンスが存在する
        When スコア値 <score> がタグに変換される
        Then 変換されたタグは "[IR]score_<tag_value>" となる

        Examples:
            | score | tag_value |
            | 0.5   | 0         |
            | 3.2   | 3         |
            | 7.9   | 7         |
            | 10.0  | 10        |

    Scenario: バッチ画像の評価
        Given 初期化されたImageRewardモデルと複数の画像が存在する
        When 画像がバッチで評価される
        Then 各画像に対して評価結果が返される

    Scenario: プロンプト文との関連性評価
        Given 初期化されたImageRewardモデルと画像、プロンプト文が存在する
        When 画像がプロンプト文と共に評価される
        Then プロンプト文との関連性を考慮したスコアが返される

    Scenario: スコアの正規化処理
        Given 初期化されたImageRewardScorerインスタンスが存在する
        When モデルからの生のスコア値 <raw_score> が正規化される
        Then 正規化後のスコアは適切な範囲内に収まる

        Examples:
            | raw_score |
            | -2.0      |
            | 0.0       |
            | 1.5       |