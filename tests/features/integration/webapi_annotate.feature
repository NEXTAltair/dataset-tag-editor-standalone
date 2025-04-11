Feature: Web API アノテーターによる画像アノテーション
    マルチモーダルなWeb APIをベースとしたアノテーターが
    画像に対して正しく動作し、適切なタグを生成できることを確認する。

    Background:
        Given APIキーが環境変数に設定されている
        And モデル設定ファイルが正しく構成されている
        And テスト用画像が用意されている

    # 各プロバイダーの正常系テスト
    Scenario: Google Gemini APIで画像を分析し正しいタグを取得できる
        Given Google Gemini APIが利用可能な状態になっている
        When 画像を指定してGeminiモデルでアノテーションを実行する
        Then アノテーション結果に期待通りの内容が含まれる
        And エラーは発生していない

    Scenario: OpenAI GPT-4oで画像を分析し正しいタグを取得できる
        Given OpenAI APIが利用可能な状態になっている
        When 画像を指定してGPT-4oモデルでアノテーションを実行する
        Then アノテーション結果に期待通りの内容が含まれる
        And エラーは発生していない

    Scenario: Anthropic Claude 3で画像を分析し正しいタグを取得できる
        Given Anthropic APIが利用可能な状態になっている
        When 画像を指定してClaude 3モデルでアノテーションを実行する
        Then アノテーション結果に期待通りの内容が含まれる
        And エラーは発生していない

    Scenario: OpenRouterで画像を分析し正しいタグを取得できる
        Given OpenRouter APIが利用可能な状態になっている
        When 画像を指定してOpenRouterモデルでアノテーションを実行する
        Then アノテーション結果に期待通りの内容が含まれる
        And エラーは発生していない

    # 異常系テスト
    Scenario: APIキーがない場合は適切なエラーを返す
        Given APIキーが未設定の状態になっている
        When 画像を指定してアノテーションを実行する
        Then APIキー未設定のエラーメッセージが返される
        And 結果のタグリストは空である

    Scenario: APIリクエストがタイムアウトした場合は適切なエラーを返す
        Given APIがタイムアウトするよう設定されている
        When 画像を指定してアノテーションを実行する
        Then タイムアウトエラーメッセージが返される
        And 結果のタグリストは空である

    Scenario: APIからエラーレスポンスが返された場合は適切に処理する
        Given APIからエラーレスポンスが返されるよう設定されている
        When 画像を指定してアノテーションを実行する
        Then APIエラーメッセージが結果に含まれる
        And 結果のタグリストは空である

    # パラメータバリエーション
    Scenario Outline: 異なるプロンプトテンプレートでも正しく動作する
        Given <provider>用のプロンプトテンプレートが「<prompt>」に設定されている
        When 画像を指定して<model_alias>モデルでアノテーションを実行する
        Then アノテーション結果にはプロンプトに対応した内容が含まれる
        And エラーは発生していない

        Examples:
            | provider  | model_alias | prompt                          |
            | google    | Gemini      | "gazou no setsumei wo shite kudasai" |
            | openai    | GPT-4o      | "Generate tags for this image." |
            | anthropic | Claude 3    | "What objects are in this image?" |
            | openrouter| OpenRouter  | "Describe what you see."        |

    Scenario: 複数の画像を一度にアノテーションできる
        Given 複数のテスト画像が用意されている
        When 複数画像を指定してアノテーションを実行する
        Then 全ての画像に対する結果が正しく返される
        And 全ての結果にエラーは含まれていない