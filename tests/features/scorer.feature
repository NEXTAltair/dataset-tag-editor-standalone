Feature: スコアラー機能
    スコアラーモジュールの機能テスト

    # ------ インスタンス生成と初期化 ------
    Scenario Outline: スコアラーインスタンスの生成
        Given モデル <model_name> が指定される
        When レジストリに登録された <model_name> に対応するクラスを取得する
        And <model_name> を引数にモデルクラスをインスタンス化する
        Then 対応したスコアラーインスタンスが生成される

        Examples:
            | model_name          |
            | ImprovedAesthetic   |
            | aesthetic_shadow_v1 |
            | WaifuAesthetic      |

    Scenario Outline: スコアラーインスタンスの一貫性確認
        Given モデル <model_name> が指定される
        When 同じモデルで2回スコアラーインスタンスを取得する
        Then 同一のインスタンスが返される

        Examples:
            | model_name          |
            | ImprovedAesthetic   |
            | aesthetic_shadow_v1 |
            | WaifuAesthetic      |

    Scenario Outline: 異なるモデルの独立性確認
        Given モデル <model_name> が指定される
        When 各モデルのスコアラーインスタンスを取得する
        Then 各インスタンスは対応するモデル名を持つ

        Examples:
            | model_name                        |
            | ImprovedAesthetic, WaifuAesthetic |

    # ------ キャッシュ処理 ------
    # このシナリオは統合テストスイートで実装されます
    # 理由：システムメモリとリソース管理の実際の挙動を検証する必要があるため
    @skip
    Scenario Outline: 複数モデルのキャッシュ管理
        Given <test_type> 用のモデル設定が存在する
        When <first_model> と <second_model> のスコアラーを順番に初期化する
        Then 両方のモデルがキャッシュされていることを確認する
        When <first_model> のモデルをアンロードする
        Then <first_model> のリソースが解放され <second_model> はキャッシュに残ることを確認する
        When <third_model> を新たに初期化する
        Then システムメモリが適切に管理されていることを確認する

        Examples:
            | test_type        | first_model         | second_model   | third_model         |
            | unit_test        | ImprovedAesthetic   | WaifuAesthetic | cafe_aesthetic      |
            | unit_test        | aesthetic_shadow_v1 | cafe_aesthetic | ImageReward         |
            | integration_test | ImprovedAesthetic   | WaifuAesthetic | aesthetic_shadow_v2 |

    @skip
    Scenario Outline: 単一画像を複数モデルで評価時のキャッシュ管理
        Given 単一の画像を用意する
        And モデル <model_name> が指定される
        When evaluate関数で初回評価を実行する
        Then 評価結果が辞書形式で返され各モデルの結果を含む
        And 使用された全モデルが_LOADED_SCORERSにキャッシュされている
        When 同じモデルリストで再度evaluate関数を実行する
        Then キャッシュからモデルが取得されること
        And 評価結果が初回と同一であること
        When <model_list>の最初のモデルをメモリから解放する
        Then そのモデルのリソースが解放されキャッシュから削除される
        And 残りのモデルはキャッシュに残っている

        Examples:
            | model_list                                   |
            | ["aesthetic_shadow_v1", "ImprovedAesthetic"] |
            | ["ImprovedAesthetic", "WaifuAesthetic"]      |
            | ["aesthetic_shadow_v1", "cafe_aesthetic"]    |

    # ------ 画像評価 ------
    Scenario Outline: 単一/複数画像の評価
        Given 単一の画像を用意する
        And 複数の画像を用意する
        And モデル <model_name> が指定される
        When 画像評価実行
        Then 辞書のキーが <model_name> の各モデル名と一致することを確認する

        Examples:
            | model_name          |
            | aesthetic_shadow_v1 |
            | WaifuAesthetic      |
            | cafe_aesthetic      |

    Scenario Outline: 複数モデルでの画像評価
        Given 単一の画像を用意する
        And 複数の画像を用意する
        And モデル <model_name> が指定される
        When 画像評価実行
        Then 辞書のキーが <model_name> の各モデル名と一致することを確認する
        And 各モデル名に対応する値が画像数と同じ長さのリストであることを確認する

        Examples:
            | model_name                                 |
            | "aesthetic_shadow_v1", "ImprovedAesthetic" |
            | "ImprovedAesthetic", "WaifuAesthetic"      |
            | "aesthetic_shadow_v1", "cafe_aesthetic"    |

    # ------ パフォーマンス ------
    @skip
    Scenario: パフォーマンス評価
        Given 複数の画像サンプル（10枚程度）が用意されている
        When "ImprovedAesthetic" と "WaifuAesthetic" の両方のモデルで評価を実行する
        Then 合計処理時間が許容範囲内であることを確認する
        And メモリ使用量が許容範囲内であることを確認する

    @skip
    Scenario Outline: スケーラビリティ検証
        Given <num_images>枚の画像サンプル
        When <model_name>で評価実行
        Then 処理時間が画像数に比例し許容範囲内である

        Examples:
            | model_name        | num_images |
            | ImprovedAesthetic | 5          |
            | ImprovedAesthetic | 20         |
            | WaifuAesthetic    | 5          |
            | WaifuAesthetic    | 20         |

    @skip
    Scenario: エラー状態の処理
        Given GPUメモリ不足をシミュレートする環境
        When 大きなモデルのロードを試みる
        Then 例外が適切に処理され代替方法が提案される
