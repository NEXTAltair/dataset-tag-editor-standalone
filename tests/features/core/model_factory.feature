Feature: ModelLoad機能
    # ModelLoadは画像評価システムで使用される機械学習モデルの
    # ロード、メモリ管理、キャッシュ制御を担当するコンポーネントです。
    # 二階層のローダー構造により、各モデルタイプの特性に応じた
    # 効率的なモデル管理を実現します。

    Background:
        Given システムが起動している
        And メモリに十分な空き容量がある

    @base_loader
    Scenario: 基底ローダーの共通機能
        Given モデル名と対象デバイスが設定されている
        When BaseModelLoaderを初期化
        Then 共通属性が正しく設定される
        And 必要なコンポーネントのリストが取得できる

    @transformers_loader
    Scenario: Transformersモデルのロード
        Given Transformersモデルのパスが設定されている
        When TransformersLoaderでロードを実行
        Then 以下のコンポーネントが生成される:
            | component | type      |
            | model     | AutoModel |
            | processor | Processor |

    @transformers_pipeline_loader
    Scenario: TransformersPipelineモデルのロード
        Given Pipelineモデルの設定:
            | parameter  | value         |
            | model_path | pipeline_path |
            | task       | image-to-text |
        When TransformersPipelineLoaderでロードを実行
        Then 以下のコンポーネントが生成される:
            | component | type          |
            | pipeline  | Pipeline      |
            | model     | PipelineModel |

    @onnx_loader
    Scenario: ONNXモデルのロード
        Given ONNXモデルのパスが設定されている
        When ONNXLoaderでロードを実行
        Then 以下のコンポーネントが生成される:
            | component | type             |
            | session   | InferenceSession |
            | csv_path  | Path             |

    @tensorflow_loader
    Scenario Outline: TensorFlowモデルのロード
        Given TensorFlowモデルのパス "<model_path>" とフォーマット "<format>" が設定されている
        # テーブル形式から直接引数を取る形式に変更
        #   | parameter  | value        |
        #   | model_path | <model_path> |
        #   | format     | <format>     |
        When TensorFlowLoaderでロードを実行
        Then 以下のコンポーネントが生成される:
            | component | type   |
            | model     | <type> |
            | model_dir | Path   |

        Examples:
            | format      | model_path      | type       |
            | h5          | model.h5        | KerasModel |
            | saved_model | saved_model_dir | SavedModel |
            | pb          | model.pb        | SavedModel |

    @clip_loader
    Scenario: CLIPモデルのロード
        Given CLIPモデルの設定:
            | parameter       | value     |
            | base_model      | clip_path |
            | model_path      | mlp_path  |
            | activation_type | ReLU      |
        When CLIPLoaderでロードを実行
        Then 以下のコンポーネントが生成される:
            | component  | type       |
            | model      | Classifier |
            | processor  | Processor  |
            | clip_model | CLIPModel  |

    @memory_management
    Scenario: メモリ使用量の計算と管理
        Given モデルサイズの推定値が設定されている
        When get_model_sizeを実行
        Then モデルの推定メモリ使用量がMB単位で返される
        And キャッシュにサイズが保存される

    @cache_control
    Scenario: キャッシュ制御
        Given 複数のモデルがメモリにロードされている
        When 新しいモデルのキャッシュが必要
        Then 最も古いモデルが解放される
        And 新しいモデルのための領域が確保される

    @cuda_management
    Scenario: CUDAメモリの管理
        Given モデルがCPUにキャッシュされている
        When restore_model_to_cudaを実行
        Then モデルがCUDAデバイスに復元される
        And モデルの状態が更新される

    @error_handling
    Scenario: メモリ不足の検出
        Given 大規模なモデルを選択
        When メモリ不足の状態でモデルをロード
        Then OutOfMemoryErrorが発生