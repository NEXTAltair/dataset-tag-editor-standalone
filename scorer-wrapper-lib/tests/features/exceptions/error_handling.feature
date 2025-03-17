Feature: エラーハンドリング
  システムは不正な入力や内部エラーに対して適切な例外処理とエラーメッセージの提供を実施することを検証する

  Scenario: 無効な画像入力による例外処理
    Given サポートしていない画像フォーマット（例: 非画像ファイル）が入力される
    When 画像評価機能が実行される
    Then システムは TypeError など、適切な例外を発生させる

  Scenario: モデルロード時の不正なパスによるエラー処理
    Given 不正な、または存在しないモデルパスが提供される
    When モデルロード処理が実行される
    Then システムは FileNotFoundError もしくは適切なエラーメッセージを返す

  Scenario: 推論実行中の内部エラーのキャッチ
    Given 有効な入力画像が提供されているが、内部処理で予期せぬエラーが発生する状況をシミュレートする
    When 画像評価機能が実行される
    Then システムは例外をキャッチし、ユーザーに分かりやすいエラーメッセージを返す

  Scenario: モデル例外の発生テスト
    Given モデル例外クラスが利用可能である
    When ModelNotFoundErrorを発生させる
    Then 例外が正しく処理される

    When ModelLoadErrorを発生させる
    Then 例外が正しく処理される

    When InvalidModelConfigErrorを発生させる
    Then 例外が正しく処理される

    When UnsupportedModelErrorを発生させる
    Then 例外が正しく処理される

    When ModelExecutionErrorを発生させる
    Then 例外が正しく処理される

    When InvalidInputErrorを発生させる
    Then 例外が正しく処理される

    When InvalidOutputErrorを発生させる
    Then 例外が正しく処理される

  Scenario: メモリ不足による例外処理
    Given 大規模なモデルや入力データによりメモリ消費が制限を超える状況
    When 処理が実行される
    Then システムはOutOfMemoryErrorを適切に処理し、明確なエラーメッセージを返す

  Scenario: 処理タイムアウトの適切な処理
    Given 処理が長時間実行される状況
    When 設定されたタイムアウト時間を超過する
    Then システムはタイムアウトエラーを発生させ、処理を適切に終了する

  Scenario: 並行リクエスト処理中のエラー処理
    Given 複数のリクエストが同時に処理されている
    When あるリクエストでエラーが発生する
    Then 他のリクエスト処理に影響せず、エラーが発生したリクエストのみ適切に例外処理される

  Scenario: 外部APIやサービス連携時のエラー処理
    Given 外部サービスやAPIが応答しない状況
    When システムが外部サービスにアクセスする
    Then 適切なタイムアウトと再試行メカニズムが機能し、エラーメッセージを返す

  Scenario: データ形式変換時のエラー処理
    Given 不正な形式のデータが処理対象となる
    When データ変換処理が実行される
    Then 適切な例外を発生させ、具体的な問題箇所を示すエラーメッセージを返す

  Scenario: GPU環境依存エラーの処理
    Given GPUが必要なモデルがCPU環境で実行される
    When モデル推論が実行される
    Then 適切なエラーメッセージと代替処理方法の提案を返す

  Scenario: 一時的なエラーからの回復処理
    Given 一時的なエラーが発生する状況（ネットワーク遮断など）
    When エラーが発生した後、条件が復旧する
    Then システムは適切に回復し、処理を続行または再開する

  Scenario: エラー発生時のログ記録処理
    Given 何らかのエラーが発生する状況
    When エラーが発生する
    Then システムは詳細なエラー情報をログに記録し、トラブルシューティングに役立つ情報を残す