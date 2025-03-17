Feature: Model Factory
    # model_factory.pyは、様々な種類の機械学習モデル（CLIP+MLP、CLIP+Classifier、BLIPなど）を
    # 設定に基づいて生成し、画像埋め込み抽出などの機能を提供するコンポーネントです。
    # このテストでは各モデル作成パターンや機能が正しく動作することを検証します。

    モデルファクトリーは異なる種類の機械学習モデルを生成して、スコアリングシステムで使用できるようにするコンポーネントです。

    Scenario: pipelineモデルの作成
        Given モデルタイプが "pipeline" の設定が与えられている
        When create_modelが呼び出される
        Then pipelineオブジェクトを含む辞書が返される

    Scenario: CLIP+MLPモデルの作成
        Given モデルタイプが "clip_mlp" の設定が与えられている
        When create_modelが呼び出される
        Then MLPモデル、CLIPプロセッサ、およびCLIPモデルを含む辞書が返される

    Scenario: CLIP+Classifierモデルの作成
        Given モデルタイプが "clip_classifier" の設定が与えられている
        When create_modelが呼び出される
        Then Classifierモデル、CLIPプロセッサ、およびCLIPモデルを含む辞書が返される

    # BLIPモデルは現在実装中のため、一般的なBLIPモデルのテストは除外しています

    Scenario: BLIP SFR Vision Language Researchモデルの作成
        Given モデルタイプが "blip_mlp" の設定が与えられている
        And クラスが "ImageRewardScorer" である
        When create_modelが呼び出される
        Then ImageRewardモデル関連のコンポーネントを含む辞書が返される

    Scenario: 画像埋め込みベクトルの生成
        Given 画像とCLIPモデルとプロセッサが準備されている
        When image_embeddings関数が呼び出される
        Then 正規化された画像の埋め込みベクトルが返される

    Scenario: MLPモデルの順伝播
        Given 入力テンソルとMLPモデルが準備されている
        When MLPモデルのforward関数が呼び出される
        Then 処理された出力テンソルが返される

    Scenario: Classifierモデルの順伝播
        Given 入力テンソルとClassifierモデルが準備されている
        When Classifierモデルのforward関数が呼び出される
        Then シグモイド活性化された出力テンソルが返される