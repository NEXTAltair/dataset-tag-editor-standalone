# Tagger Wrapper Library 開発計画

このドキュメントは、dataset-tag-editor が持つ AI アノテーション機能（tagger 機能）をライブラリ化するための開発計画と TODO リストを提供します。

## プロジェクト概要

tagger-wrapper-lib は、各種画像タグ付けモデル（BLIP、DeepDanbooru、WD Tagger など）に統一されたインターフェースを提供するライブラリです。scorer-wrapper-lib と同様のアーキテクチャを採用し、モジュール性と拡張性を重視します。

## TODO リスト

### フェーズ 1: 分析と設計 (優先度: 高)

- [x] 既存 tagger 実装のコード分析

  - [x] scripts/tagger.py の機能分析

    - `Tagger` クラスは、カスタムタグ付けクラスの基底クラスとして機能し、コンテキストマネージャとして使用可能です。
    - `start` と `stop` メソッドは、タグ付けプロセスの開始と終了を管理します。
    - `predict` メソッドは、画像に対してタグを予測します。

  - [x] scripts/dataset_tag_editor/taggers_builtin.py の機能分析

    - 複数のタグ付けクラスが定義され、各クラスは `Tagger` クラスを継承し、特定のインターロゲータを使用して画像からタグを生成します。

  - [x] scripts/dataset_tag_editor/interrogators/ 内の各モデル実装調査

    - `waifu_diffusion_tagger_timm.py` は、画像と変換を管理するデータセットクラスを含み、モデルのロード、アンロード、適用を行います。
    - `deepbooru_model.py` は、ResNet をベースにしたモデルで、画像からタグを予測します。
    - `waifu_diffusion_tagger.py` は、Hugging Face Hub からモデルをロードし、画像にタグを付けるためのクラスです。
      - `load` メソッドでモデルとラベルをロードし、`apply` メソッドで画像にタグを付けます。
    - `blip_large_captioning.py` は、BLIP 大規模キャプショニングモデルを使用して画像からキャプションを生成するクラスです。
      - `load` メソッドでモデルとプロセッサをロードし、`apply` メソッドで画像からキャプションを生成します。
    - `deep_danbooru_tagger.py` は、DeepDanbooru モデルを使用して画像にタグを付けるクラスです。
      - `load` メソッドでモデルをロードし、`apply` メソッドで画像にタグを付けます。
    - `git_large_captioning.py` は、GIT 大規模キャプショニングモデルを使用して画像からキャプションを生成するクラスです。
      - `load` メソッドでモデルとプロセッサをロードし、`apply` メソッドで画像からキャプションを生成します。
    - `blip2_captioning.py` は、BLIP2 キャプショニングモデルを使用して画像からキャプションを生成するクラスです。
      - `load` メソッドでモデルとプロセッサをロードし、`apply` メソッドで画像からキャプションを生成します。

  - [x] 依存関係と必要なライブラリのリスト化

    - 主要なライブラリとして `PIL`, `torch`, `torchvision`, `timm`, `numpy`, `tqdm` などが使用されています。

  - [x] 参考実装 stable-diffusion-webui-wd14-tagger\scripts\tagger.py の分析
    - `Image` オブジェクトの初期化や、トランケートされた画像の読み込みエラーを防ぐ設定が行われています。
    - `script_callbacks` を使用して、アプリケーションの開始時や UI タブの設定時に特定の関数を呼び出します。

- [ ] アーキテクチャ設計

  - [x] BaseTagger 抽象クラスの設計

    - `BaseTagger` クラスを設計し、タグ付けモデルとスコアリングモデルの共通インターフェースを提供します。
    - 主なメソッドとプロパティ:

      ```python
      class BaseTagger(ABC):
          def __init__(
              self,
              model_name: str,
          ):
              # 基本属性を設定
              self.model_name = model_name
              self.config: dict[str, Any] = load_model_config()[model_name]
              self.model_path = self.config["model_path"]
              self.device = self.config["device"]

              # モデルインスタンスと他の共通属性
              self.model: dict[str, Any] = {}
              self.logger = logging.getLogger(__name__)

          @abstractmethod
          def __enter__(self) -> "BaseTagger":
              """
              この処理はタイプ別クラスに任す
              """
              pass

          def __exit__(self, exception_type: type[Exception], exception_value: Exception, traceback: Any) -> None:
              self.model = ModelLoad.cache_to_main_memory(self.model_name, self.model)

          @abstractmethod
          def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
              """画像からタグを予測します。"""
              pass

          @abstractmethod
          def _generate_result(self, model_output: Any, tags: list[str]) -> dict[str, Any]:
              """標準化された結果の辞書を生成します。

              Args:
                  model_name (str): モデルの名前。
                  model_output: モデルの出力。
                  tags (list)-各サブクラスでスコアを変換したタグ

              Returns:
                  dict: モデル出力、モデル名、スコアタグを含む辞書。
              """
          return {
              "model_name": self.model_name,
              "model_output": model_output,
              "tags": tag_list,
          }

      ```

  - [x] モデルタイプ別の中間抽象クラスの設計

    - TransformerModelTagger クラス:

      ```python
      class TransformerModelTagger(BaseTagger):
          """Transformersライブラリを使用するモデル用の抽象クラス。
          BLIP、BLIP2、GITなどのHugging Face Transformersベースのモデルの基底クラスとして機能します。
          """
          def __init__(self, model_name: str):
              """TransformerModelTagger を初期化します。
              Args:
                  model_name (str): モデルの名前。
              """
              super().__init__(model_name=model_name)
              # 設定ファイルから追加パラメータを取得
              self.max_length = self.config.get("max_length", 75)
              self.processor_path = self.config.get("processor_path", self.model_path)
              self.processor = None  # __enter__でロード

          def __enter__(self) -> "TransformerModelTagger":
              """モデルとプロセッサをロードします。"""

              # Transformers固有の初期化処理
              if "processor" not in self.model:
                  from transformers import AutoProcessor
                  self.model["processor"] = AutoProcessor.from_pretrained(self.processor_path)

              return self
          def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
              """画像からタグを予測します。"""
              results = []
              for image in images:
                  # 前処理
                  inputs = self._preprocess_image(image)
                  # モデル推論
                  outputs = self._run_inference(inputs)
                  # 後処理してタグに変換
                  tags = self._postprocess_output(outputs)
                  # 結果を標準形式で追加
                  results.append(self._generate_result(outputs, tags))
              return results

          @abstractmethod
          def _preprocess_image(self, image: Image.Image) -> Any:
              """画像を前処理してモデル入力形式に変換します。
              Args:
                  image (Image.Image): 入力画像

              Returns:
                  Any: モデル用に処理された入力データ
              """
              pass

          @abstractmethod
          def _run_inference(self, inputs: Any) -> Any:
              """モデル推論を実行します。
              Args:
                  inputs (Any): モデルへの入力データ

              Returns:
                  Any: モデルからの出力
              """
              pass

          @abstractmethod
          def _postprocess_output(self, outputs: Any) -> list[str]:
              """モデル出力を処理してタグのリストに変換します。
              Args:
                  outputs (Any): モデルからの出力

              Returns:
                  list[str]: タグのリスト
              """
              pass
      ```

    - PipelineModelTagger クラス:

      ```python
      class PipelineModelTagger(BaseTagger):
          """パイプラインインターフェースを使用するモデル用の抽象クラス。"""
          def __init__(self, model_name: str):
              """PipelineModelTagger を初期化します。
              Args:
                  model_name (str): モデルの名前。
              """
              super().__init__(model_name=model_name)
              # 設定ファイルから追加パラメータを取得
              self.threshold = self.config.get("threshold", 0.35)

          def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
              """パイプラインモデルで画像からタグを予測します。"""
              results = []
              for image in images:
                  pipeline_model = self.model["pipeline"]
                  raw_output = pipeline_model(image)
                  self.logger.debug(f"モデル '{self.model_name}' の生の出力結果: {raw_output}")
                  tags = self._process_pipeline_output(raw_output)
                  results.append(self._generate_result(raw_output, tags))
              return results

          @abstractmethod
          def _process_pipeline_output(self, raw_output: Any) -> list[str]:
              pass
      ```

    - ONNXModelTagger クラス:

      ```python
      class ONNXModelTagger(BaseTagger):
          """ONNXランタイムを使用するモデル用の抽象クラス。"""
          def __init__(self, model_name: str):
              """ONNXModelTagger を初期化します。
              Args:
                  model_name (str): モデルの名前。
              """
              super().__init__(model_name=model_name)
              # 設定ファイルから追加パラメータを取得
              self.tags_path = self.config["tags_path"]
              self.threshold = self.config.get("threshold", 0.5)
              self.input_size = self.config.get("input_size", (448, 448))
              self.labels = []  # __enter__でロード

          def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
              """画像からタグを予測します。"""
              results = []
              for image in images:
                  # 前処理
                  input_data = self._preprocess_image(image)
                  # モデル推論
                  output_data = self._run_inference(input_data)
                  # 後処理してタグに変換
                  tags = self._postprocess_output(output_data)
                  # 結果を標準形式で追加
                  results.append(self._generate_result(output_data, tags))
              return results

          @abstractmethod
          def _preprocess_image(self, image: Image.Image) -> np.ndarray:
              """画像を前処理してONNXモデル入力形式に変換します。
              Args:
                  image (Image.Image): 入力画像

              Returns:
                  np.ndarray: ONNXモデル用に処理された入力データ
              """
              pass

          @abstractmethod
          def _run_inference(self, input_data: np.ndarray) -> np.ndarray:
              """ONNXモデル推論を実行します。
              Args:
                  input_data (np.ndarray): モデルへの入力データ

              Returns:
                  np.ndarray: モデルからの出力
              """
              pass

          @abstractmethod
          def _postprocess_output(self, output_data: np.ndarray) -> list[str]:
              """ONNXモデル出力を処理してタグのリストに変換します。
              Args:
                  output_data (np.ndarray): モデルからの出力

              Returns:
                  list[str]: タグのリスト
              """
              pass
      ```

    - TorchModelTagger クラス:

      ```python
      class TorchModelTagger(BaseTagger):
          """PyTorch (非Transformers)モデルを使用する抽象クラス。"""
          def __init__(self, model_name: str):
              """TorchModelTagger を初期化します。
              Args:
                  model_name (str): モデルの名前。
              """
              super().__init__(model_name=model_name)
              # 設定ファイルから追加パラメータを取得
              self.tags_path = self.config["tags_path"]
              self.threshold = self.config.get("threshold", 0.5)
              self.input_size = self.config.get("input_size", (512, 512))

          def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
              """画像からタグを予測します。"""
              results = []
              with torch.no_grad():
                  for image in images:
                      # 前処理して入力テンソルに変換
                      input_tensor = self._preprocess_image(image)
                      # モデル推論
                      output_tensor = self._run_inference(input_tensor)
                      # 後処理してタグに変換
                      tags = self._postprocess_output(output_tensor)
                      # 結果を標準形式で追加
                      results.append(self._generate_result(output_tensor, tags))
              return results

          @abstractmethod
          def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
              """画像を前処理してPyTorchモデル入力形式に変換します。
              Args:
                  image (Image.Image): 入力画像

              Returns:
                  torch.Tensor: PyTorchモデル用に処理された入力テンソル
              """
              pass

          @abstractmethod
          def _run_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
              """PyTorchモデル推論を実行します。
              Args:
                  input_tensor (torch.Tensor): モデルへの入力テンソル

              Returns:
                  torch.Tensor: モデルからの出力テンソル
              """
              pass

          @abstractmethod
          def _postprocess_output(self, output_tensor: torch.Tensor) -> list[str]:
              """PyTorchモデル出力を処理してタグのリストに変換します。
              Args:
                  output_tensor (torch.Tensor): モデルからの出力テンソル

              Returns:
                  list[str]: タグのリスト
              """
              pass
      ```

  - [x] 具体的なモデルクラスの設計

    モデルクラスは 3 階層の構造で実装します:

    1. `BaseTagger`: すべてのタガーの抽象基底クラス
    2. モデルタイプ別中間クラス: `TransformerModelTagger`, `PipelineModelTagger`, `ONNXModelTagger`, `TorchModelTagger`
    3. 具体的なモデル実装クラス: 各モデルの独自処理を実装した最終的なクラス

    各モデルタイプ特有の引数と`__enter__`メソッドを含めた具体的なモデル実装クラスの詳細：

    ## TransformerModel ベースのタガー

    ```python
    class BLIPTagger(TransformerModelTagger):
        """BLIP大規模キャプショニングモデルを使用したタガー"""

        def __enter__(self) -> "BLIPTagger":
            """BLIPモデルとプロセッサをロードします。"""
            # TransformerModelTaggerの__enter__を呼び出し
            super().__enter__()

            # BLIPモデル固有のロードロジック
            if "model" not in self.model:
                from transformers import BlipForConditionalGeneration, BlipProcessor

                self.logger.info(f"BLIPモデル '{self.model_path}' をロードしています...")
                self.model["model"] = BlipForConditionalGeneration.from_pretrained(
                    self.model_path
                ).to(self.device)

                self.model["processor"] = BlipProcessor.from_pretrained(
                    self.processor_path or self.model_path
                )

            return self

        def _preprocess_image(self, image: Image.Image) -> dict:
            """画像を前処理してBLIPモデル入力形式に変換します。"""
            # BLIPプロセッサを使用した前処理
            inputs = self.model["processor"](images=image, return_tensors="pt").to(self.device)
            return inputs

        def _run_inference(self, inputs: dict) -> Any:
            """BLIP推論を実行します。"""
            # モデルからキャプションを生成
            outputs = self.model["model"].generate(**inputs, max_length=self.max_length)
            return outputs

        def _postprocess_output(self, outputs: Any) -> list[str]:
            """BLIP出力を処理してタグのリストに変換します。"""
            # トークンをテキストにデコード
            caption = self.model["processor"].decode(outputs[0], skip_special_tokens=True)
            # キャプションをタグに分割
            tags = [tag.strip() for tag in caption.split(",")]
            return tags
    ```

    ## ONNXModel ベースのタガー

    ```python
    class WaifuDiffusionTagger(ONNXModelTagger):
        """Waifu Diffusion Taggerを使用したタガー"""

        def __enter__(self) -> "WaifuDiffusionTagger":
            """WD-Taggerモデルとラベルをロードします。"""
            super().__enter__()

            # WD-Tagger固有のロードロジック
            if "session" not in self.model:
                import onnxruntime as ort
                import numpy as np

                self.logger.info(f"WD-Taggerモデル '{self.model_path}' をロードしています...")

                # ONNXランタイムのセッション作成
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == "cuda" else ['CPUExecutionProvider']
                self.model["session"] = ort.InferenceSession(self.model_path, providers=providers)

                # ラベルのロード
                with open(self.tags_path, 'r', encoding='utf-8') as f:
                    self.model["labels"] = [line.strip() for line in f]

            return self

        def _preprocess_image(self, image: Image.Image) -> np.ndarray:
            """画像を前処理してONNXモデル入力形式に変換します。"""
            import numpy as np

            # 画像のリサイズと正規化
            image = image.resize(self.input_size)
            image_array = np.array(image).astype(np.float32)
            # チャンネル順の変換とバッチ次元の追加
            image_array = image_array.transpose(2, 0, 1)
            image_array = image_array / 255.0
            return image_array[np.newaxis, ...]

        def _run_inference(self, input_data: np.ndarray) -> np.ndarray:
            """ONNX推論を実行します。"""
            # モデルを実行して予測を取得
            ort_session = self.model["session"]
            ort_inputs = {ort_session.get_inputs()[0].name: input_data}
            ort_outputs = ort_session.run(None, ort_inputs)
            return ort_outputs[0]

        def _postprocess_output(self, output_data: np.ndarray) -> list[str]:
            """ONNX出力を処理してタグのリストに変換します。"""
            import numpy as np

            # 閾値以上の確率を持つラベルを抽出
            indices = np.where(output_data[0] > self.threshold)[0]
            tags = [self.model["labels"][idx] for idx in indices]
            return tags
    ```

    ## TorchModel ベースのタガー

    ````python
    class DeepDanbooruTagger(TorchModelTagger):
        """DeepDanbooruモデルを使用したタガー"""

        def __enter__(self) -> "DeepDanbooruTagger":
            """DeepDanbooruモデルとタグをロードします。"""
            super().__enter__()

            # DeepDanbooru固有のロードロジック
            if "model" not in self.model:
                import torch
                from torchvision import transforms

                self.logger.info(f"DeepDanbooruモデル '{self.model_path}' をロードしています...")

                # トランスフォーム設定
                self.model["transform"] = transforms.Compose([
                    transforms.Resize(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ])

                # モデルのロード
                self.model["model"] = torch.load(self.model_path, map_location=self.device)
                self.model["model"].eval()

                # タグのロード
                with open(self.tags_path, 'r', encoding='utf-8') as f:
                    self.model["tags"] = [line.strip() for line in f]

            return self

        def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
            """画像を前処理してPyTorchモデル入力形式に変換します。"""
            import torch

            # 画像のリサイズと正規化
            image = image.convert("RGB")
            image_tensor = self.model["transform"](image).unsqueeze(0).to(self.device)
            return image_tensor

        def _run_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
            """PyTorchモデル推論を実行します。"""
            # モデルを通して予測を取得
            output = self.model["model"](input_tensor)
            return output

        def _postprocess_output(self, output_tensor: torch.Tensor) -> list[str]:
            """PyTorchモデル出力を処理してタグのリストに変換します。"""
            import numpy as np

            # 閾値以上の確率を持つラベルを抽出
            probs = output_tensor[0].cpu().numpy()
            indices = np.where(probs > self.threshold)[0]
            tags = [self.model["tags"][idx] for idx in indices]
            return tags
        ```

    各モデルタイプで必要な設定パラメータまとめ

    1. **BaseTagger**

       - 共通パラメータ: `model_name`, `model_type`, `model_path`, `device`（デフォルト: "cuda"）

    2. **TransformerModelTagger**

       - 追加パラメータ: `max_length`（キャプション生成長）, `processor_path`（オプション）

    3. **ONNXModelTagger**

       - 追加パラメータ: `tags_path`, `threshold`, `input_size`

    4. **TorchModelTagger**

       - 追加パラメータ: `tags_path`, `threshold`, `input_size`（デフォルト: (512, 512)）

    5. **PipelineModelTagger**
       - 追加パラメータ: `tag_prefix`, `threshold`

    この設計により、各モデルタイプに特化した初期化とロードロジックが実装され、それぞれのモデルの特性に合わせた処理が可能になります。

    ````

  - [ ] 公開 API インターフェースの設計

    - 公開 API 関数:

      ```python
      def list_available_taggers() -> list[str]:
          """利用可能なタグ付けモデルのリストを返します。

          Returns:
              list[str]: 利用可能なタグ付けモデルの名前のリスト
          """
          pass

      def tag_images(
          images_list: list[Image.Image],
          model_name_list: list[str]
      ) -> dict[str, list[list[str]]]:
          """画像にタグを付けます。

          Args:
              images_list (list[Image.Image]): タグを付ける画像のリスト
              model_name_list (list[str]): 使用するモデルの名前のリスト

          Returns:
              dict[str, list[list[str]]]: モデル名をキーとし、各モデルの予測結果を値とするディクショナリ。
                  各予測結果は、画像ごとのタグのリストのリスト。
          """
          pass
      ```

    - `tagger_registry.py` の主要関数:

      ```python
      def register_taggers() -> dict[str, Type[BaseTagger]]:
          """利用可能なタグ付けモデルを登録します。

          Returns:
              dict[str, Type[BaseTagger]]: モデル名をキーとし、タガークラスを値とするディクショナリ
          """
          pass

      def get_cls_obj_registry() -> dict[str, Type[BaseTagger]]:
          """登録されたタグ付けモデルのディクショナリを返します。

          Returns:
              dict[str, Type[BaseTagger]]: モデル名をキーとし、タガークラスを値とするディクショナリ
          """
          pass
      ```

  - [x] モデル設定ファイル形式の設計

    - `config/taggers.toml` の形式:

      ```toml
      # タグ付けモデル設定ファイル
      [models.blip]
      model_path = "Salesforce/blip-image-captioning-large"
      device = "cuda"

      [models.deepdanbooru]
      model_path = "deepdanbooru-v3-20211112-sgd-e28.onnx"
      threshold = 0.5
      tags_path = "deepdanbooru-v3-20211112-sgd-e28.txt"
      device = "cuda"

      [models.wd_tagger]
      model_path = "wd-v1-4-swinv2-tagger-v2"
      threshold = 0.35
      device = "cuda"
      ```

    - モデル設定の読み込み関数:

      ```python
      def load_model_config() -> dict[str, dict[str, Any]]:
          """タグ付けモデルの設定を読み込みます。

          Returns:
              dict[str, dict[str, Any]]: モデル名をキーとし、設定を値とするディクショナリ
          """
          pass
      ```

  - [ ] BDD テストシナリオ設計
    - [ ] 基本機能のシナリオ作成
    - [ ] エラーハンドリングのシナリオ作成
    - [ ] リソース管理のシナリオ作成

### フェーズ 2: 基本実装 (優先度: 高)

- [ ] プロジェクト構造のセットアップ

  - [ ] ディレクトリ構造の作成
  - [ ] pyproject.toml の作成
  - [ ] 初期 README の作成

- [ ] コア機能の実装

  - [ ] `BaseTagger` 抽象基底クラスの実装
  - [ ] `tagger_registry.py` の実装
  - [ ] 設定ファイル読み込みシステムの実装
  - [ ] リソース管理機能の実装

- [ ] 公開 API 実装
  - [ ] `list_available_taggers()` 関数の実装
  - [ ] `tag_images()` 関数の実装

### フェーズ 3: モデル実装 (優先度: 中)

- [ ] キャプショニングモデルの実装

  - [ ] BLIP モデルの移行と実装
  - [ ] BLIP2 モデルの移行と実装
  - [ ] GIT モデルの移行と実装

- [ ] タグ付けモデルの実装

  - [ ] DeepDanbooru の移行と実装
  - [ ] WD Tagger の移行と実装

- [ ] 共通処理の抽象化
  - [ ] 画像前処理の共通化
  - [ ] 結果後処理の共通化

### フェーズ 4: テスト実装 (優先度: 中)

- [ ] ユニットテスト

  - [ ] コアクラスのテスト実装
  - [ ] ユーティリティ関数のテスト実装

- [ ] BDD テスト

  - [ ] `.feature` ファイルの実装
  - [ ] ステップ定義の実装

- [ ] 統合テスト
  - [ ] 実際のモデルを使用したエンドツーエンドテスト

### フェーズ 5: ドキュメント整備 (優先度: 中)

- [ ] 設計ドキュメント

  - [ ] アーキテクチャ図の作成
  - [ ] クラス図の作成
  - [ ] シーケンス図の作成

- [ ] ユーザードキュメント

  - [ ] インストール方法の記述
  - [ ] 基本的な使い方の例示
  - [ ] 設定オプションの解説

- [ ] 開発者ドキュメント
  - [ ] 新しい tagger の追加方法
  - [ ] 拡張ポイントの説明

### フェーズ 6: 統合と最適化 (優先度: 低)

- [ ] scorer-wrapper-lib との連携

  - [ ] 共通ユーティリティの共有
  - [ ] 相互運用性の確保

- [ ] パフォーマンス最適化

  - [ ] バッチ処理の効率化
  - [ ] 並列処理の実装

- [ ] dataset-tag-editor への統合
  - [ ] 既存コードからライブラリの利用への移行
  - [ ] 後方互換性の確保

## ディレクトリ構造案

```
tagger-wrapper-lib/
├── src/
│   └── tagger_wrapper_lib/
│       ├── __init__.py        // ライブラリのエントリーポイント
│       ├── tagger.py          // タグ付け実行のメイン処理
│       ├── tagger_registry.py // モデル登録と管理
│       ├── core/
│       │   ├── __init__.py
│       │   ├── base.py        // BaseTagger基底クラス
│       │   ├── utils.py       // 共通ユーティリティ
│       │   └── model_factory.py // モデル作成とパラメータ管理
│       ├── tag_models/        // 各モデルの具体的実装
│       │   ├── __init__.py
│       │   ├── blip_model.py
│       │   ├── blip2_model.py
│       │   ├── git_model.py
│       │   ├── deepdanbooru_model.py
│       │   └── wd_tagger_model.py
│       └── exceptions/        // 例外定義
│           ├── __init__.py
│           └── tagger_errors.py
├── config/                   // 設定ファイル
│   └── taggers.toml          // モデル設定
├── docs/                     // ドキュメント
├── pyproject.toml
└── README.md
```

## 開発ガイドライン

1. **コーディングスタイル**

   - PEP 8 に準拠したコーディングスタイル
   - 型ヒントの活用
   - 適切なドキュメント文字列

2. **コミット規約**

   - 機能単位での小さなコミット
   - 明確なコミットメッセージ

3. **テスト方針**

   - テスト駆動開発を推奨
   - 高いテストカバレッジの維持

4. **リソース管理**
   - GPU/CPU メモリの効率的な利用
   - リソースのクリーンアップ保証

## スケジュール目標

- フェーズ 1: 1 週間
- フェーズ 2: 2 週間
- フェーズ 3: 2 週間
- フェーズ 4: 1 週間
- フェーズ 5: 1 週間
- フェーズ 6: 1 週間

合計: 約 8 週間
