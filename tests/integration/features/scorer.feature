Feature: スコアラー機能
    スコアラーモジュールの各機能の単体テストを行う

    Scenario: スコアラーインスタンスの生成
        Given 有効なクラス名とモデル名が指定される
        When _create_scorer_instance関数を呼び出す
        Then 指定したクラスのスコアラーインスタンスが生成される

    Scenario: 無効なクラス名でのスコアラーインスタンス生成
        Given 無効なクラス名とモデル名が指定される
        When _create_scorer_instance関数を呼び出す
        Then 適切なエラーが発生する

    Scenario: スコアラーの初期化
        Given 有効なモデル名が指定される
        When init_scorer関数を呼び出す
        Then スコアラーが正常に初期化され、キャッシュに保存される

    Scenario: すでに初期化済みのスコアラーの再初期化
        Given すでに初期化されたモデル名が指定される
        When init_scorer関数を呼び出す
        Then キャッシュから既存のスコアラーが返される

    Scenario: 無効なモデル名でのスコアラー初期化
        Given 存在しないモデル名が指定される
        When init_scorer関数を呼び出す
        Then 適切なエラーが発生する

    Scenario: 単一モデルでの評価
        Given 有効なスコアラーインスタンスと画像リストが存在する
        When _evaluate_model関数を呼び出す
        Then 画像ごとの評価結果リストが取得できる

    Scenario Outline: 複数モデルでの評価
        Given 画像リストが存在する
        And <model_list> のモデルリストが指定される
        When evaluate関数を呼び出す
        Then 各モデルの評価結果が辞書形式で取得できる

        Examples:
            | model_list                                     |
            | ["aesthetic_shadow_v1"]                        |
            | ["aesthetic_shadow_v1", "aesthetic_shadow_v2"] |
            | ["aesthetic_shadow_v1", "cafe_aesthetic"]      |
            | []                                             |

    Scenario: スコアラーの評価エラー処理
        Given エラーを発生させるモックスコアラーが存在する
        When _evaluate_model関数を呼び出す
        Then エラーが適切に処理される

    Scenario: キャッシュされていないモデルでの評価
        Given キャッシュされていないモデル名のリストが指定される
        When evaluate関数を呼び出す
        Then 自動的にモデルが初期化され評価結果が返される