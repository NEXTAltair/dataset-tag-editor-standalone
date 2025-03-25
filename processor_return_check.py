import hashlib
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
import requests
import tensorflow as tf
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 設定
config = {
    "model_path": "https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip",
    "device": "cuda",
}


class DeepDanbooruError(Exception):
    """DeepDanbooruに関連するエラー"""

    pass


class ModelNotFoundError(DeepDanbooruError):
    """モデルファイルが見つからない場合のエラー"""

    pass


class TagFileNotFoundError(DeepDanbooruError):
    """タグファイルが見つからない場合のエラー"""

    pass


# 前処理された画像データを表す型定義
class PreprocessedImage(TypedDict):
    # TransformerModel
    pixel_values: Optional[torch.Tensor]

    # ONNXModelTagger
    input_data: Optional[np.ndarray]


# model_factory によって生成されるモデルのコンポーネントを表す型定義
class ModelComponents(TypedDict, total=False):
    model: Optional[torch.nn.Module]  # PyTorchモデルのインスタンス
    processor: Optional[AutoProcessor]  # Transformers用のプロセッサ
    session: Optional[ort.InferenceSession]  # ONNXセッション
    csv_path: Optional[str]  # ONNXモデル用ラベルCSVのパス
    model_path: Optional[str]  # モデルのパス


# 画像の推論結果を人が読める形式に変換したものを表す型定義
class PredictionResult(TypedDict):
    # BLIP BLIP2 GIT
    caption: Optional[List[str]]

    # ONNX
    output_data: Optional[Dict[str, Dict[str, float]]]


CONFIG_TOML = Path("config") / "taggers.toml"
LOG_FILE = Path("logs/tagger_wrapper_lib.log")
DEFAULT_CACHE_DIR = Path("models")
DEFAULT_TIMEOUT = 30
WD_MODEL_FILENAME = "model.onnx"
WD_LABEL_FILENAME = "selected_tags.csv"


def _get_cache_path(url: str, cache_dir: Path) -> Path:
    """URLからキャッシュファイルパスを生成する"""
    filename = Path(urlparse(url).path).name
    if not filename or len(filename) < 5:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        extension = Path(urlparse(url).path).suffix
        filename = f"{url_hash}{extension}" if extension else f"{url_hash}.bin"
    return cache_dir / filename


def _is_cached(url: str, cache_dir: Path) -> tuple[bool, Path]:
    """URLに対応するファイルがキャッシュに存在するか確認する"""
    local_path = _get_cache_path(url, cache_dir)
    return local_path.is_file(), local_path


def _perform_download(url: str, target_path: Path, expected_hash: Optional[str] = None) -> None:
    """実際のダウンロード処理を行う（進捗表示付き）"""
    logger.info(f"Downloading model from {url} to {target_path}")
    response = requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()

    # ファイルサイズを取得
    total_size = int(response.headers.get("content-length", 0))

    with open(target_path, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc=target_path.name) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))


def _download_from_url(
    url: str, cache_dir: Path = DEFAULT_CACHE_DIR, expected_hash: Optional[str] = None
) -> Path:
    """
    URLからファイルをダウンロードし、キャッシュされたローカルパスを返します。

    Args:
        url: ダウンロードするファイルのURL
        cache_dir: キャッシュディレクトリ（デフォルト: 設定ファイルから）
        expected_hash: 期待されるSHA256ハッシュの先頭部分（オプション）

    Returns:
        Path: 絶対パスに変換されたパスオブジェクト
    """
    # ダウンロード先フォルダを作成
    cache_dir.mkdir(exist_ok=True, parents=True)

    # キャッシュチェック
    is_cached, local_path = _is_cached(url, cache_dir)

    # キャッシュされていなければダウンロード
    if not is_cached:
        _perform_download(url, local_path, expected_hash)

    return local_path.resolve()


@lru_cache(maxsize=128)
def get_file_path(path_or_url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """パスまたはURLからローカルファイルパスを取得（結果をキャッシュ）"""
    parsed = urlparse(path_or_url)

    if parsed.scheme in ("http", "https"):
        return _download_from_url(path_or_url, cache_dir)
    else:
        return _get_local_file_path(path_or_url)


def _get_local_file_path(path: str) -> Path:
    """ローカルファイルパスを検証し、絶対パスを返します。"""
    local_path = Path(path)
    if local_path.exists():
        return local_path.resolve()
    raise FileNotFoundError(f"ローカルファイル '{path}' が見つかりません")


def load_file(path_or_url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """
    指定されたパスまたはURLからファイルを取得し、ローカルパスを返します。
    もしダウンロードしたファイルがZIP形式の場合は、自動的に解凍し、解凍先ディレクトリのパスを返します。
    """
    try:
        file_path = get_file_path(path_or_url, cache_dir)
        if file_path.suffix.lower() == ".zip":
            import zipfile

            extracted_dir = file_path.parent / (file_path.stem)
            if not extracted_dir.exists():
                with zipfile.ZipFile(file_path, "r") as zip_ref:
                    zip_ref.extractall(extracted_dir)
            return extracted_dir
        else:
            return file_path
    except requests.RequestException as e:
        raise RuntimeError(f"URLからのダウンロードに失敗しました: {e}") from e
    except FileNotFoundError as e:
        raise RuntimeError(f"ローカルファイルが見つかりません: {e}") from e
    except Exception as e:
        raise RuntimeError(
            f"'{path_or_url}' からのファイル取得に失敗しました。"
            "有効なローカルパス、または直接URLを指定してください。"
            f"エラー詳細: {e}"
        ) from e


class ModelLoad:
    _MODEL_STATES: dict[str, str] = {}
    logger = logging.getLogger(__name__)

    @staticmethod
    def load_tags(tags_path):
        with open(tags_path, "r") as tags_stream:
            tags = [tag for tag in (tag.strip() for tag in tags_stream) if tag]
            return tags

    @staticmethod
    def load_deepdanbooru_components(
        model_name: str, model_path: str, device: str
    ) -> Optional[dict[str, Any]]:
        """
        DeepDanbooruモデルを独自実装でロードします。
        # NOTE: 依存関係で求められるTensorFlow_IOがWindowsで開発してるので使えない

        引数:
            model_name (str): モデル名
            model_path (str): DeepDanbooru用のモデルパス
            device (str): 使用するデバイス ('cuda' または 'cpu')
        戻り値:
            dict: {'model': ロードされたモデル, 'tags': モデルが持つタグリスト}
        """
        if model_name in ModelLoad._MODEL_STATES:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に読み込まれています。")
            return None
        try:
            model_dir = load_file(model_path)
            # 必須ファイル: タグファイルとモデルファイル
            tags_path = model_dir / "tags.txt"
            tags_character_path = model_dir / "tags-character.txt"
            tags_general_path = model_dir / "tags-general.txt"

            # モデルファイルを検索
            h5_model_path = next(model_dir.glob("*.h5"), None)

            # モデルをロード
            ModelLoad.logger.info(f"DeepDanbooruモデルをロード中: {h5_model_path}")
            model = tf.keras.models.load_model(h5_model_path, compile=True)

            # 必須タグをロード
            tags = ModelLoad.load_tags(tags_path)
            tags_character = ModelLoad.load_tags(tags_character_path)
            tags_general = ModelLoad.load_tags(tags_general_path)

            # 結果の辞書を初期化
            components = {
                "model": model,
                "tags": tags,
                "tags_character": tags_character,
                "tags_general": tags_general,
            }

            # モデル状態を記録
            ModelLoad._MODEL_STATES[model_name] = f"on_{device}"
            return components

        except Exception as e:
            ModelLoad.logger.error(f"DeepDanbooruモデルのロードに失敗しました: {e}")
            return None

    @staticmethod
    def cache_to_main_memory(model_name: str, model: dict[str, Any]) -> dict[str, Any]:
        """モデルを CPU メモリにキャッシュします。

        モデルのすべてのコンポーネントを GPU から CPU メモリに移動します。
        これにより、GPU 上のメモリは解放されますが、モデル自体は保持されるため、
        後で `restore_model_to_cuda` を呼び出して再利用できます。

        主な用途:
        - モデル自体は保持したまま GPU リソースを解放したい場合

        Note:
            このメソッドはモデルを破棄しません。モデルを完全に解放するには
            `release_model` を使用してください。
        """
        if ModelLoad._MODEL_STATES[model_name] == "on_cpu":
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に CPU にあります。")
            return model

        for component_name, component in model.items():
            if component_name == "pipeline":
                # パイプラインの場合は内部モデルを移動
                if hasattr(component, "model"):
                    component.model.to("cpu")
                ModelLoad.logger.debug(f"パイプライン '{component_name}' を CPU に移動しました")
            elif hasattr(component, "to"):  # to メソッドを持つ場合のみ CPU に移動
                component.to("cpu")
                ModelLoad.logger.debug(f"コンポーネント '{component_name}' を CPU に移動しました")

        ModelLoad._MODEL_STATES[model_name] = "on_cpu"
        return model

    @staticmethod
    def restore_model_to_cuda(model_name: str, device: str, model: dict[str, Any]) -> dict[str, Any]:
        """モデルを指定 CUDA に復元します。"""
        # モデル名がMODEL_STATESに存在しない場合は何もせず返す
        if model_name not in ModelLoad._MODEL_STATES:
            return model

        if ModelLoad._MODEL_STATES[model_name] == "on_cpu" and "cuda" in device:
            for component_name, component in model.items():
                if component_name == "pipeline":
                    if hasattr(component, "model"):
                        component.model.to("cuda")

                elif hasattr(component, "to"):
                    component.to("cuda")

                ModelLoad._MODEL_STATES[model_name] = "on_cuda"
                ModelLoad.logger.info(f"モデル '{model_name}' をメインメモリから復元しました。")
            return model

        return model


class TagConfidence(TypedDict):
    """タグの信頼度情報を表す型定義"""

    confidence: float  # 信頼度
    source: str  # タグの情報元（例: deepdanbooru）


class BaseTagger(ABC):
    def __init__(self, model_name: str):
        """
        BaseTagger を初期化します。

        引数:
            model_name (str): モデルの名前。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = config

        self.model_path = self.config["model_path"]
        self.device = self.config.get("device", "cuda")
        self.components: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def __enter__(self) -> "BaseTagger":
        pass

    @abstractmethod
    def __exit__(self, exception_type: type[Exception], exception_value: Exception, traceback: Any) -> None:
        pass

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """画像からタグを予測します。"""
        results = []
        for image in images:
            try:
                processed_image = self._preprocess_image(image)
                # モデル推論
                raw_output = self._run_inference(processed_image)
                # 推論結果をフォーマット
                formatted_output = self._format_predictions(raw_output)
                # タグを生成
                annotation_list = self._generate_tags(formatted_output)
                # 結果を標準形式で追加
                results.append(self._generate_result(formatted_output, annotation_list))
            except ValueError as e:
                self.logger.error(f"推論生成中のエラー: {e}")
                raise
        return results

    @abstractmethod
    def _preprocess_image(self, image: Image.Image) -> Any:
        """画像を前処理してモデル入力形式に変換します。"""
        pass

    @abstractmethod
    def _run_inference(self, processed_image: Any) -> Any:
        """モデル推論を実行します。"""
        pass

    @abstractmethod
    def _format_predictions(self, raw_output: Any) -> Any:
        """モデルの生出力をフォーマットします。"""
        pass

    @abstractmethod
    def _generate_tags(self, raw_output: Any) -> list[str]:
        """モデルの生出力からタグを生成します。"""
        pass

    def _generate_result(self, model_output: Any, annotation_list: list[str]) -> dict[str, Any]:
        """標準化された結果の辞書を生成します。

        Args:
            model_name (str): モデルの名前。
            model_output: モデルの出力。
            annotation_list (list[str]): 各サブクラスで

        Returns:
            dict: モデル出力、モデル名、タグリストを含む辞書。
        """
        return {
            "model_name": self.model_name,
            "model_output": model_output,
            "annotation": annotation_list,
        }


class UseLibModel(BaseTagger, ABC):
    """
    深層学習モデルで共通して利用可能な中間クラス。
    依存関係にあるパッケージに推論機能があるときに使う
    # TODO: 中身はモデルごとに異なるので、モデルごとにサブクラスを作成する
    # TODO: 共通の処理を思いつけばここに実装
    """

    def __init__(self, model_name):
        """
        Args:
            model_name (str): モデル名
        """
        super().__init__(model_name)


# mDeepDanbooruTagger（改良版 DeepDanbooruTagger）
class DeepDanbooruTagger(UseLibModel):
    """DeepDanbooruを使用して画像のタグ予測を行う具象クラス（改良版）

    deepdanbooru.model.load_model を使用してモデルを初期化し、タグ予測を行います。
    """

    def __init__(self, model_name: str):
        """
        DeepDanbooruTagger の初期化

        引数:
            threshold (float): タグ選択のための信頼度閾値
            device (str): 使用するデバイス ('cpu' または 'cuda')
        """
        # グローバル config に定義された model_path を使用
        super().__init__(model_name=model_name)
        self.threshold = 0.7

    def __enter__(self) -> "DeepDanbooruTagger":
        """
        モデルの状態に基づいて、必要な場合のみロードまたは復元
        """
        # モデルコンポーネントをロード
        loaded_model = ModelLoad.load_deepdanbooru_components(
            self.model_name,
            self.model_path,
            self.device,
        )

        # モデルのロードに失敗した場合
        if loaded_model is None:
            self.logger.error(
                f"モデル '{self.model_name}' のロードに失敗しました。DeepDanbooruモデルが正しくインストールされているか確認してください。"
            )
            # 既にロード済みならcomponentsに保持しているはずなので、self.componentsが空であれば初回ロード失敗
            if not self.components:
                # このままでは推論実行時にエラーになるため、警告を出す
                self.logger.warning("モデルがロードされていないため、推論は実行できません。")
        else:
            # モデルのロードに成功した場合、コンポーネントを更新
            self.components = loaded_model
            self.logger.info(f"モデル '{self.model_name}' をロードしました。")

        # CUDAへの復元を試みる（既にロードされている場合のみ）
        if self.components:
            self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, self.components)

        return self

    def __exit__(
        self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]
    ) -> None:
        # TODO: 仮実装 必要に応じてリソース（例:GPUメモリやセッション）の解放を実装する
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        入力画像の前処理を行う

        入力画像を PIL Image に変換し、(512, 512) にリサイズ、正規化します。

        引数:
            image (Image.Image): 入力画像（PIL Image）

        戻り値:
            np.ndarray: 前処理済み画像（浮動小数点数配列）
        """
        # ここでは、入力は PIL Image のみとする
        image = image.convert("RGB")
        image = image.resize((512, 512))
        image_array = np.asarray(image).astype(np.float32) / 255.0
        return image_array

    def _run_inference(self, processed_image: np.ndarray) -> tf.Tensor:
        """
        モデル推論を実行する

        引数:
            processed_image (np.ndarray): 前処理済み画像。
                単一画像の場合は (512, 512, 3)
                複数画像の場合は (バッチサイズ, 512, 512, 3) を想定します。

        戻り値:
            np.ndarray: モデルの生出力
        """
        # 入力の次元をチェック
        if processed_image.ndim == 3:
            # 単一画像の場合、形状は (512, 512, 3) です。
            # モデルはバッチ形式 (None, 512, 512, 3) を要求するため、
            # 1枚の画像でもバッチ次元を追加して (1, 512, 512, 3) に変換します。
            processed_image = np.expand_dims(processed_image, axis=0)
            self.logger.debug(
                "単一画像を検出: バッチ次元を追加して形状を (1, 512, 512, 3) に変換しました。"
            )
        else:
            # 想定外の形状の場合はエラーを発生させます。
            raise ValueError(f"予期しない入力形式: {processed_image.shape}")

        # 推論実行
        raw_output = self.components["model"](processed_image)
        return raw_output

    def _format_predictions(self, raw_output: tf.Tensor) -> dict[str, dict[str, TagConfidence]]:
        """
        予測の生出力を解釈可能な形式に変換する

        タグを3つのカテゴリ（一般タグ、キャラクタータグ、その他タグ）に分けて処理し、
        カテゴリ別にネストした辞書構造で返します。各カテゴリ内のタグは信頼度順にソートされます。

        引数:
            raw_output: モデルの生出力

        戻り値:
            dict[str, dict[str, TagConfidence]]: カテゴリ別にネストされたタグと信頼度の辞書
        """
        # 各カテゴリのタグをフォーマット
        general_tags = self._format_general_tags(raw_output)
        character_tags = self._format_character_tags(raw_output)
        other_tags = self._format_all_tags(raw_output)

        # 各カテゴリのタグを信頼度でソート
        sorted_output = {}
        for category, tags in [
            ("general", general_tags),
            ("character", character_tags),
            ("other", other_tags),
        ]:
            # タグを信頼度でソート
            sorted_tags = dict(sorted(tags.items(), key=lambda x: x[1]["confidence"], reverse=True))
            sorted_output[category] = sorted_tags

        return sorted_output

    def _format_general_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        一般タグに対する予測結果をフォーマットする

        引数:
            raw_output: モデルの生出力

        戻り値:
            dict[str, TagConfidence]: 一般タグと信頼度の辞書
        """
        formatted_output: dict[str, TagConfidence] = {}

        # モデルが一般タグを持っている場合のみ処理
        if "tags_general" in self.components and self.components["tags_general"]:
            # raw_outputの形状が一次元の場合、すべてのタグに対応する確率が1つの配列に格納されている
            all_tags = self.components["tags"]  # すべてのタグのリスト
            general_tags = self.components["tags_general"]  # 一般タグのリスト

            # すべてのタグとその確率をマッピング
            tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0])}

            # 一般タグに対する確率値を取得
            for tag in general_tags:
                if tag in tag_probs:
                    formatted_output[tag] = TagConfidence(
                        confidence=float(tag_probs[tag]), source="deepdanbooru_general"
                    )

        return formatted_output

    def _format_character_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        キャラクタータグに対する予測結果をフォーマットする

        引数:
            raw_output: モデルの生出力

        戻り値:
            dict[str, TagConfidence]: キャラクタータグと信頼度の辞書
        """
        formatted_output: dict[str, TagConfidence] = {}

        # モデルがキャラクタータグを持っている場合のみ処理
        if "tags_character" in self.components and self.components["tags_character"]:
            # raw_outputの形状が一次元の場合、すべてのタグに対応する確率が1つの配列に格納されている
            all_tags = self.components["tags"]  # すべてのタグのリスト
            character_tags = self.components["tags_character"]  # キャラクタータグのリスト

            # すべてのタグとその確率をマッピング
            tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0])}

            # キャラクタータグに対する確率値を取得
            for tag in character_tags:
                if tag in tag_probs:
                    formatted_output[tag] = TagConfidence(
                        confidence=float(tag_probs[tag]), source="deepdanbooru_character"
                    )

        return formatted_output

    def _format_all_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        すべてのタグに対する予測結果をフォーマットする

        一般タグとキャラクタータグに分類されていないタグも含みます。
        タグファイルが存在しない場合は、すべてのタグをここで処理します。

        引数:
            raw_output: モデルの生出力

        戻り値:
            dict[str, TagConfidence]: すべてのタグと信頼度の辞書
        """
        formatted_output: dict[str, TagConfidence] = {}
        all_tags = self.components["tags"]

        # 分類済みタグを取得（存在する場合のみ）
        classified_tags = set()
        if "tags_general" in self.components and self.components["tags_general"]:
            classified_tags.update(self.components["tags_general"])
        if "tags_character" in self.components and self.components["tags_character"]:
            classified_tags.update(self.components["tags_character"])

        # 分類されていないタグのみ処理
        # 分類用のタグファイルが存在しない場合は、すべてのタグをここで処理
        for tag, probability in zip(all_tags, raw_output[0]):
            if tag not in classified_tags:
                formatted_output[tag] = TagConfidence(confidence=float(probability), source="deepdanbooru")

        return formatted_output

    def _generate_tags(self, formatted_output: dict[str, dict[str, TagConfidence]]) -> list[str]:
        """
        フォーマット済み出力から閾値以上の信頼度を持つタグリストを生成する

        引数:
            formatted_output: フォーマット済み生出力（予測結果配列）

        戻り値:
            list[str]: 信頼度が閾値以上のタグリスト
        """
        results: list[str] = []
        for category, tags in formatted_output.items():
            if category == "other":
                continue
            for tag, confidence in tags.items():
                if confidence["confidence"] >= self.threshold:
                    results.append(tag)
        return results


if __name__ == "__main__":
    from datetime import datetime

    # テスト用画像パス
    test_image_path = Path("tests/resources/img/1_img/file05.webp")
    if not test_image_path.exists():
        print(f"エラー: テスト画像が見つかりません: {test_image_path}")
        exit(1)

    # 結果保存用ディレクトリ作成
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # 結果ファイル名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"deepdanbooru_results_{timestamp}.txt"

    try:
        # DeepDanbooruTaggerを使用して画像を解析
        with DeepDanbooruTagger("deepdanbooru-v3-20211112-sgd-e28") as tagger:
            # 画像を読み込んで予測
            image = Image.open(test_image_path)
            predictions = tagger.predict([image])[0]

            # 結果をファイルに保存
            with open(result_file, "w", encoding="utf-8") as f:
                # ヘッダー情報
                f.write("DeepDanbooru 予測結果\n")
                f.write(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"画像: {test_image_path}\n")
                f.write(f"モデル: {predictions['model_name']}\n")
                f.write("-" * 80 + "\n\n")

                # カテゴリー別に結果を表示
                model_output = predictions["model_output"]
                for category, tags in model_output.items():
                    f.write(f"【{category}】\n")
                    # 信頼度0.5以上のタグを表示
                    for tag, data in tags.items():
                        if data["confidence"] >= 0.5:
                            f.write(f"{tag:<30} : {data['confidence']:.4f}\n")
                    f.write("\n")

            print(f"予測結果を {result_file} に保存しました。")
            print(f"検出されたタグ数: {len(predictions['annotation'])}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        exit(1)
