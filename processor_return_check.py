import hashlib
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TypedDict
from urllib.parse import urlparse

import numpy as np
import onnxruntime as ort
import requests
import tensorflow as tf
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor

# ログ設定
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# 設定
config = {
    "model_path": "https://github.com/KichangKim/DeepDanbooru/releases/download/v3-20211112-sgd-e28/deepdanbooru-v3-20211112-sgd-e28.zip",
    "device": "cuda",
}

# 定数
DEFAULT_CACHE_DIR = Path("models")
DEFAULT_TIMEOUT = 30


# 型定義
class PreprocessedImage(TypedDict):
    pixel_values: Optional[torch.Tensor]  # Transformer用
    input_data: Optional[np.ndarray]  # ONNX用


# model_factory によって生成されるモデルのコンポーネントを表す型定義
class ModelComponents(TypedDict, total=False):
    model: Optional[torch.nn.Module]  # PyTorchモデルのインスタンス
    processor: Optional[AutoProcessor]  # Transformers用のプロセッサ
    session: Optional[ort.InferenceSession]  # ONNXセッション
    csv_path: Optional[str]  # ONNXモデル用ラベルCSVのパス
    model_path: Optional[str]  # モデルのパス


class PredictionResult(TypedDict):
    caption: Optional[list[str]]
    output_data: Optional[dict[str, dict[str, float]]]


# ダウンロード関連の関数


def perform_download(url: str, target_path: Path, expected_hash: Optional[str] = None) -> None:
    """
    指定されたURLからファイルをダウンロードし、進捗バーを表示します。

    Args:
        url (str): ダウンロード元のURL。
        target_path (Path): ダウンロードしたファイルを保存するパス。
        expected_hash (Optional[str], optional): ファイルのハッシュ確認用。既定値はNone。
    """
    logger.info(f"{url} から {target_path} へダウンロード中")
    response = requests.get(url, stream=True, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with (
        open(target_path, "wb") as f,
        tqdm(total=total_size, unit="B", unit_scale=True, desc=target_path.name) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def download_file(
    url: str, cache_dir: Path = DEFAULT_CACHE_DIR, expected_hash: Optional[str] = None
) -> Path:
    """
    キャッシュディレクトリを使用して、指定されたURLからファイルをダウンロードします。

    Args:
        url (str): ダウンロード元のURL。
        cache_dir (Path, optional): キャッシュ用ディレクトリ。既定値は DEFAULT_CACHE_DIR。
        expected_hash (Optional[str], optional): ハッシュ確認用。既定値は None。

    Returns:
        Path: ダウンロードまたはキャッシュされたファイルの絶対パス。
    """
    cache_dir.mkdir(exist_ok=True, parents=True)
    filename = Path(urlparse(url).path).name
    if not filename or len(filename) < 5:
        url_hash = hashlib.md5(url.encode()).hexdigest()
        extension = Path(urlparse(url).path).suffix
        filename = f"{url_hash}{extension if extension else '.bin'}"
    file_path = cache_dir / filename
    if not file_path.is_file():
        perform_download(url, file_path, expected_hash)
    return file_path.resolve()


@lru_cache(maxsize=128)
def get_file_path(path_or_url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """
    指定された文字列がローカルファイルパスかURLかを判別し、対応するファイルパスを返します。

    Args:
        path_or_url (str): ローカルパスまたはURLの文字列。
        cache_dir (Path, optional): ダウンロードファイルのキャッシュディレクトリ。既定値は DEFAULT_CACHE_DIR。

    Returns:
        Path: 解決済みのファイルパス。

    Raises:
        FileNotFoundError: ローカルファイルが存在しない場合に発生。
    """
    parsed = urlparse(path_or_url)
    if parsed.scheme in ("http", "https"):
        return download_file(path_or_url, cache_dir)
    else:
        local_path = Path(path_or_url)
        if local_path.exists():
            return local_path.resolve()
        raise FileNotFoundError(f"ローカルファイル '{path_or_url}' が見つかりません")


# ヘルパー関数: ZIP解凍処理


def _extract_zip(file_path: Path) -> Path:
    """
    ZIPアーカイブを解凍して、その解凍先ディレクトリのパスを返します。

    Args:
        file_path (Path): ZIPファイルのパス。

    Returns:
        Path: 解凍先ディレクトリのパス。
    """
    import zipfile

    extract_dir = file_path.parent / file_path.stem
    if not extract_dir.exists():
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)
    return extract_dir


# ヘルパー関数: タグ情報の一括読み込み


def _load_tags_from_dir(model_dir: Path) -> tuple[list[str], list[str], list[str]]:
    """
    モデルディレクトリから、tags.txt, tags-character.txt, tags-general.txtの内容を読み込みます。

    Args:
        model_dir (Path): モデルのディレクトリ。

    Returns:
        tuple[list[str], list[str], list[str]]: それぞれのタグリスト。
    """
    tags = ModelLoad.load_tags(str(model_dir / "tags.txt"))
    tags_character = ModelLoad.load_tags(str(model_dir / "tags-character.txt"))
    tags_general = ModelLoad.load_tags(str(model_dir / "tags-general.txt"))
    return tags, tags_character, tags_general


def load_file(path_or_url: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """
    ローカルパスまたはURLからファイルを取得し、ZIPアーカイブの場合は解凍してディレクトリパスを返します。

    Args:
        path_or_url (str): ローカルファイルパスまたはURL。
        cache_dir (Path, optional): キャッシュディレクトリ。既定値は DEFAULT_CACHE_DIR。

    Returns:
        Path: ファイルのパス、またはZIPの場合は解凍先ディレクトリのパス。

    Raises:
        RuntimeError: ファイルの取得に失敗した場合に発生。
    """
    try:
        file_path = get_file_path(path_or_url, cache_dir)
    except Exception as e:
        raise RuntimeError(f"'{path_or_url}' からファイルを取得できません: {e}") from e
    if file_path.suffix.lower() == ".zip":
        return _extract_zip(file_path)
    return file_path


# モデルロード関連のクラス


class ModelLoad:
    _model_states: dict[str, str] = {}
    logger = logging.getLogger(__name__)

    @staticmethod
    def load_tags(tags_path: str) -> list[str]:
        """
        タグファイルから、空行を除いたタグのリストを読み込みます。

        Args:
            tags_path (str): タグファイルのパス。

        Returns:
            list[str]: タグのリスト。
        """
        with open(tags_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def load_deepdanbooru_components(
        model_name: str, model_path: str, device: str
    ) -> Optional[dict[str, Any]]:
        """
        DeepDanbooruモデルをロードし、その構成要素を返します。

        Args:
            model_name (str): モデルの名称。
            model_path (str): モデルのURLまたはローカルパス。
            device (str): モデルをロードするデバイス（例：'cuda' または 'cpu'）。

        Returns:
            Optional[dict[str, Any]]: モデルの構成要素（モデル、タグなど）の辞書。ロードに失敗した場合はNoneを返す。
        """
        if model_name in ModelLoad._model_states:
            ModelLoad.logger.debug(f"モデル '{model_name}' は既にロード済みです。")
            return None
        try:
            model_dir = load_file(model_path)
            tags, tags_character, tags_general = _load_tags_from_dir(model_dir)
            h5_model = next(model_dir.glob("*.h5"), None)
            ModelLoad.logger.info(f"DeepDanbooruモデルをロード中: {h5_model}")
            model = tf.keras.models.load_model(h5_model, compile=True)
            components = {
                "model": model,
                "tags": tags,
                "tags_character": tags_character,
                "tags_general": tags_general,
            }
            ModelLoad._model_states[model_name] = f"on_{device}"
            return components
        except Exception as e:
            ModelLoad.logger.error(f"モデル '{model_name}' のロードに失敗しました: {e}")
            return None

    @staticmethod
    def cache_to_main_memory(model_name: str, model: dict[str, Any]) -> dict[str, Any]:
        """
        モデルの構成要素をGPUからCPUに移動します。

        Args:
            model_name (str): モデルの名称。
            model (dict[str, Any]): モデル構成要素の辞書。

        Returns:
            dict[str, Any]: CPUに移動後のモデル構成要素の辞書。
        """
        if ModelLoad._MODEL_STATES[model_name] == "on_cpu":
            ModelLoad.logger.debug(f"モデル '{model_name}' は既に CPU にあります。")
            return model
        for comp_name, comp in model.items():
            if hasattr(comp, "to"):
                comp.to("cpu")
                ModelLoad.logger.debug(f"{comp_name} をCPUに移動しました")
        ModelLoad._model_states[model_name] = "on_cpu"
        return model

    @staticmethod
    def restore_model_to_cuda(model_name: str, device: str, model: dict[str, Any]) -> dict[str, Any]:
        """
        必要に応じて、モデルの構成要素をCPUからCUDAに復元します。

        Args:
            model_name (str): モデルの名称。
            device (str): 復元先のデバイス（'cuda'を含む必要があります）。
            model (dict[str, Any]): モデル構成要素の辞書。

        Returns:
            dict[str, Any]: CUDAに移動した（またはそのままの場合もある）モデル構成要素の辞書。
        """
        if ModelLoad._model_states.get(model_name) == "on_cpu" and "cuda" in device:
            for comp_name, comp in model.items():
                if hasattr(comp, "to"):
                    comp.to("cuda")
            ModelLoad._model_states[model_name] = "on_cuda"
            ModelLoad.logger.info(f"モデル '{model_name}' をCUDAに復元しました")
        return model


# タグの信頼度情報
class TagConfidence(TypedDict):
    """タグの信頼度情報を表す型定義"""

    confidence: float  # 信頼度
    source: str  # タグの情報元（例: deepdanbooru）


class BaseTagger(ABC):
    def __init__(self, model_name: str):
        """
        BaseTagger を初期化します。

        Args:
            model_name (str): モデルの名称。
        """
        self.model_name = model_name
        self.config: dict[str, Any] = config

        self.model_path = self.config["model_path"]
        self.device = self.config.get("device", "cuda")
        self.components: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def __enter__(self) -> "BaseTagger":
        """
        タガーのコンテキストに入ります。

        Returns:
            BaseTagger: タガーのインスタンス。
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type: type, exc_value: Exception, traceback: Any) -> None:
        """
        タガーのコンテキストを終了し、リソースを解放します。
        """
        pass

    def predict(self, images: list[Image.Image]) -> list[dict[str, Any]]:
        """
        画像のリストに対してタグ予測を実施します。

        Args:
            images (list[Image.Image]): PIL Imageオブジェクトのリスト。

        Returns:
            list[dict[str, Any]]: 予測結果の辞書リスト。
        """
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
        """
        入力画像を前処理し、モデル入力に適した形式に変換します。

        Args:
            image (Image.Image): 入力画像。

        Returns:
            Any: 推論可能な形式に前処理された画像。
        """
        pass

    @abstractmethod
    def _run_inference(self, processed: Any) -> Any:
        """
        前処理済み画像に対して推論を実施します。

        Args:
            processed (Any): 前処理済みの画像。

        Returns:
            Any: モデルからの生出力。
        """
        pass

    @abstractmethod
    def _format_predictions(self, raw_output: Any) -> Any:
        """
        モデルの生出力を構造化された形式に整形します。

        Args:
            raw_output (Any): モデルからの生出力。

        Returns:
            Any: 整形済みの予測結果。
        """
        pass

    @abstractmethod
    def _generate_tags(self, formatted_output: Any) -> list[str]:
        """
        整形済み予測結果からタグのリストを生成します。

        Args:
            formatted_output (Any): 整形済みの予測結果。

        Returns:
            list[str]: 生成されたタグのリスト。
        """
        pass

    def _generate_result(self, output: Any, tags: list[str]) -> dict[str, Any]:
        """
        標準化された結果の辞書を生成します。

        Args:
            output (Any): 整形済みのモデル出力。
            tags (list[str]): 生成されたタグのリスト。

        Returns:
            dict[str, Any]: モデル名、出力、タグ情報を含む辞書。
        """
        return {
            "model_name": self.model_name,
            "model_output": output,
            "annotation": tags,
        }


# 共通モデル用中間クラス
class UseLibModel(BaseTagger, ABC):
    def __init__(self, model_name: str):
        super().__init__(model_name)


# DeepDanbooruを用いた具体的なタグ予測クラス
class DeepDanbooruTagger(UseLibModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.threshold = 0.7

    def __enter__(self) -> "DeepDanbooruTagger":
        """
        モデル構成要素をロードし、CUDAへの復元を実施してDeepDanbooruTaggerのコンテキストに入ります。

        Returns:
            DeepDanbooruTagger: ロード済みモデル構成を持つタガーのインスタンス。
        """
        components = ModelLoad.load_deepdanbooru_components(self.model_name, self.model_path, self.device)
        if components:
            self.components = ModelLoad.restore_model_to_cuda(self.model_name, self.device, components)
            self.logger.info(f"モデル '{self.model_name}' を正常にロードしました")
        else:
            self.logger.error(f"モデル '{self.model_name}' のロードに失敗しました")
        return self

    def __exit__(
        self, exc_type: Optional[type], exc_value: Optional[BaseException], traceback: Optional[Any]
    ) -> None:
        """
        DeepDanbooruTaggerのコンテキストを終了し、モデル構成要素をCPUに移動します。
        """
        self.components = ModelLoad.cache_to_main_memory(self.model_name, self.components)

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        入力画像をRGBに変換し、512x512にリサイズ、正規化します。

        Args:
            image (Image.Image): 入力画像。

        Returns:
            np.ndarray: float32型のNumPy配列として前処理された画像。
        """
        image = image.convert("RGB").resize((512, 512))
        return np.asarray(image).astype(np.float32) / 255.0

    def _run_inference(self, processed: np.ndarray) -> tf.Tensor:
        """
        前処理された画像に対し、モデル推論を実行します。

        Args:
            processed (np.ndarray): 前処理済みの画像配列。

        Returns:
            tf.Tensor: モデルからの生出力。

        Raises:
            ValueError: 入力次元が期待値と異なる場合に発生。
        """
        if processed.ndim == 3:
            processed = np.expand_dims(processed, axis=0)
        else:
            raise ValueError(f"予期しない入力形式: {processed.shape}")
        return self.components["model"](processed)

    def _format_predictions(self, raw_output: tf.Tensor) -> dict[str, dict[str, TagConfidence]]:
        """
        生の出力を整形・分類し、タグの信頼度情報の辞書にまとめます。

        Args:
            raw_output (tf.Tensor): モデルからの生出力。

        Returns:
            dict[str, dict[str, TagConfidence]]: 'general', 'character', 'other' の各カテゴリごとにタグと信頼度情報をマッピングした辞書。
        """
        general = self._format_general_tags(raw_output)
        character = self._format_character_tags(raw_output)
        other = self._format_all_tags(raw_output)
        return {
            "general": dict(sorted(general.items(), key=lambda x: x[1]["confidence"], reverse=True)),
            "character": dict(sorted(character.items(), key=lambda x: x[1]["confidence"], reverse=True)),
            "other": dict(sorted(other.items(), key=lambda x: x[1]["confidence"], reverse=True)),
        }

    def _format_general_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        生の出力から一般タグの情報を抽出し、信頼度情報を付与します。

        Args:
            raw_output (tf.Tensor): モデルからの生出力。

        Returns:
            dict[str, TagConfidence]: 一般タグをキーに、信頼度と情報元を値とする辞書。
        """
        all_tags = self.components["tags"]
        tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0])}
        return {
            tag: TagConfidence(confidence=float(tag_probs[tag]), source="deepdanbooru_general")
            for tag in self.components["tags_general"]
            if tag in tag_probs
        }

    def _format_character_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        生の出力からキャラクタータグの情報を抽出し、信頼度情報を付与します。

        Args:
            raw_output (tf.Tensor): モデルの生出力。

        Returns:
            dict[str, TagConfidence]: キャラクタータグをキーに、信頼度と情報元を値とする辞書。
        """
        all_tags = self.components["tags"]
        tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0])}
        return {
            tag: TagConfidence(confidence=float(tag_probs[tag]), source="deepdanbooru_character")
            for tag in self.components["tags_character"]
            if tag in tag_probs
        }

    def _format_all_tags(self, raw_output: tf.Tensor) -> dict[str, TagConfidence]:
        """
        生の出力から、一般タグ・キャラクタータグ以外のタグ情報を抽出します。

        Args:
            raw_output (tf.Tensor): モデルの生出力。

        Returns:
            dict[str, TagConfidence]: 残りのタグをキーに、信頼度と情報元を値とする辞書。
        """
        all_tags = self.components["tags"]
        tag_probs = {tag: prob for tag, prob in zip(all_tags, raw_output[0])}
        classified = set(self.components["tags_general"]) | set(self.components["tags_character"])
        return {
            tag: TagConfidence(confidence=float(prob), source="deepdanbooru")
            for tag, prob in tag_probs.items()
            if tag not in classified
        }

    def _generate_tags(self, formatted_output: dict[str, dict[str, TagConfidence]]) -> list[str]:
        """
        整形済みの予測結果から閾値以上の信頼度をもつタグのみを抽出してリストを生成します。

        Args:
            formatted_output (dict[str, dict[str, TagConfidence]]): 整形済みの予測結果。

        Returns:
            list[str]: 信頼度がしきい値以上のタグのリスト。
        """
        return [
            tag
            for cat, tags in formatted_output.items()
            if cat != "other"
            for tag, conf in tags.items()
            if conf["confidence"] >= self.threshold
        ]


if __name__ == "__main__":
    from datetime import datetime

    test_image = Path("tests/resources/img/1_img/file05.webp")
    if not test_image.exists():
        print(f"エラー: テスト画像が見つかりません: {test_image}")
        exit(1)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"deepdanbooru_results_{timestamp}.txt"

    try:
        with DeepDanbooruTagger("deepdanbooru-v3-20211112-sgd-e28") as tagger:
            image = Image.open(test_image)
            prediction = tagger.predict([image])[0]

            with open(result_file, "w", encoding="utf-8") as f:
                f.write("DeepDanbooru 予測結果\n")
                f.write(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"画像: {test_image}\n")
                f.write(f"モデル: {prediction['model_name']}\n")
                f.write("-" * 80 + "\n\n")

                for category, tags in prediction["model_output"].items():
                    f.write(f"【{category}】\n")
                    for tag, data in tags.items():
                        if data["confidence"] >= 0.5:
                            f.write(f"{tag:<30} : {data['confidence']:.4f}\n")
                    f.write("\n")

            print(f"予測結果を {result_file} に保存しました。")
            print(f"検出されたタグ数: {len(prediction['annotation'])}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        exit(1)
