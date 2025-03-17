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

Feature: パフォーマンスとスケーラビリティの統合テスト
  スコアラーライブラリの処理速度とリソース使用効率を検証する

  Scenario: 複数モデルの同時評価パフォーマンス
    Given 複数の画像サンプル（10枚程度）が用意されている
    When "ImprovedAesthetic" と "WaifuAesthetic" の両方のモデルで同時に評価を実行する
    Then 合計処理時間が個別に処理した時間の合計より短いか同等であることを確認する
    And メモリ使用量が許容範囲内であることを確認する

  Scenario Outline: 異なる画像数での処理速度の検証
    Given <num_images>枚の画像サンプルが用意されている
    When <model_name> モデルで評価を実行する
    Then 処理時間が画像数にほぼ線形比例することを確認する
    And 平均処理時間が許容範囲内であることを確認する

    Examples:
      | model_name        | num_images |
      | ImprovedAesthetic | 5          |
      | ImprovedAesthetic | 10         |
      | ImprovedAesthetic | 20         |
      | WaifuAesthetic    | 5          |
      | WaifuAesthetic    | 10         |
      | WaifuAesthetic    | 20         |

  Scenario: バッチサイズによる処理効率の検証
    Given 処理対象の画像セット（20枚程度）が用意されている
    When 異なるバッチサイズ（1, 4, 8）で処理を実行する
    Then 最適なバッチサイズと処理時間の関係が記録される
    And GPUメモリ使用量とバッチサイズの関係が記録される

Feature: 異常系の統合テスト
  実際の使用環境を想定した異常系の統合テスト

  Scenario: 無効な画像形式での処理
    Given 正常画像と無効画像（破損ファイル、サポート外形式）の混合セットが用意されている
    When 複数モデルを使用して画像評価を実行する
    Then 無効画像に対しては適切な例外が発生する
    And 処理可能な画像のみ正常に評価される
    And エラーログが適切に記録される

  Scenario: 大きすぎる画像サイズの処理
    Given 非常に解像度が高い画像（例:10000x10000ピクセル）が用意されている
    When 画像評価を実行する
    Then 適切なエラーメッセージと共に処理が制御される
    And システムクラッシュが発生しない

  Scenario: モデルロード中の例外処理
    Given モデルロードに失敗する状況をシミュレートする
    When モデル初期化を試みる
    Then ModelLoadError例外が適切に捕捉される
    And エラー情報が詳細にログに記録される
    And ユーザーフレンドリーなエラーメッセージが返される

  Scenario: GPUメモリ不足時の動作
    Given GPUメモリ不足をシミュレートする環境
    When 大きなモデルのロードを試みる
    Then OutOfMemoryError例外が適切に捕捉される
    And CPUモードへのフォールバックまたは適切なエラーメッセージが返される
