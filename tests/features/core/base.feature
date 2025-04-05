# このfeatureファイルはsrc/image_annotator_lib/core/base.pyモジュールの
# BaseBaseAnnotator抽象クラスとその実装クラスの基本機能をテストします。
#
# 【モデルロードの仕組み】
# - BaseBaseAnnotator の初期化時（__init__）ではモデル構造や重みはロードされません
# - 実際のモデルロードは _load_model メソッドで行われ、内部で model_factory.py の
#   create_model 関数を呼び出します
# - 画像の埋め込み処理は image_embeddings 関数を使用します
# - モデルのロード状態は is_model_loaded フラグで管理されます

Feature: BaseBaseAnnotator基本機能のテスト
    BaseBaseAnnotator クラスとその派生クラスの基本機能をテストする

    Background:
        Given テスト用のモデル設定が存在する

    Scenario: モデルの直接ロード
        Given スコアラーインスタンスが初期化されている
        When _load_modelメソッドを直接呼び出す
        Then モデルが正しくロードされる
        And is_model_loadedフラグがTrueになる
        And config.tomlで指定された全てのモデルコンポーネントがロードされている

    Scenario: モデルのロードと復元
        Given スコアラーインスタンスが初期化されている
        When load_or_restore_modelメソッドを呼び出す
        Then モデルが正しくロードされる
        And is_model_loadedフラグがTrueになる
        When 再度load_or_restore_modelメソッドを呼び出す
        Then モデルが再ロードされずメモリから復元される

    Scenario: リソースの解放
        Given スコアラーインスタンスが初期化されていてモデルがロードされている
        When release_resourcesメソッドを呼び出す
        Then is_model_loadedフラグがFalseになる
        And メインメモリからモデルリソースが完全に解放される
        And 必要に応じてGPUキャッシュもクリアされる

    Scenario: メモリキャッシュと復元
        # モデルを保持したままGPUとCPU間で移動させる機能のテスト
        Given スコアラーインスタンスが初期化されていてモデルがロードされている
        When cache_to_main_memoryメソッドを呼び出す
        Then モデルがCPUメモリにキャッシュされる
        And モデル内の全コンポーネントがCPUに移動する
        And GPUメモリ(VRAM)が解放される
        When restore_from_main_memoryメソッドを呼び出す
        Then モデルが指定デバイスに復元される
        And モデル内の全コンポーネントが元のデバイスに移動する
        And is_model_loadedフラグはTrueのままである

    Scenario: cache_and_release_modelメソッドの動作
        # モデルを一時的にキャッシュしてから完全に解放する複合機能のテスト
        Given スコアラーインスタンスが初期化されていてモデルがロードされている
        When cache_and_release_modelメソッドを呼び出す
        Then モデルが一時的にCPUメモリにキャッシュされる
        And その後リソースが完全に解放される
        And is_model_loadedフラグがFalseになる

    Scenario: 各派生クラスのpredict機能
        Given <モデルタイプ>インスタンスが初期化されてモデルがロードされている
        When 画像を入力して予測を実行する
        Then 正しい形式の予測結果が返される
        And 結果には必要なキー（model_output、model_name、score_tag）が含まれている

        Examples:
            | モデルタイプ        |
            | PipelineModel       |
            | ClipMlpModel        |
            | ClipClassifierModel |

    Scenario: 画像の前処理機能
        Given スコアラーインスタンスが初期化されていてモデルがロードされている
        When preprocessメソッドに画像を入力する
        Then 適切な形式のテンソルが返される