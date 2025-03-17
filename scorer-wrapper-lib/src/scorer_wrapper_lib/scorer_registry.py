import importlib
import inspect
import logging
import traceback
from pathlib import Path
from types import ModuleType
from typing import Type, TypeVar

from .core.base import BaseScorer
from .core.utils import load_model_config

T = TypeVar("T", bound=BaseScorer)
ModelClass = Type[BaseScorer]

_MODEL_CLASS_OBJ_REGISTRY: dict[str, ModelClass] = {}  # モデル名 → クラスオブジェクト
logger = logging.getLogger(__name__)


def list_module_files(directory: str) -> list[Path]:
    base_dir = Path(__file__).parent
    abs_path = (base_dir / directory).resolve()
    logger.debug(f"base_dir: {base_dir}, searching in abs_path: {abs_path}")
    module_files = []
    for p in abs_path.glob("*.py"):
        if p.name != "__init__.py":
            logger.debug(f"Found module file: {p}")
            module_files.append(p)
    logger.debug(f"見つかった合計モジュールファイル: {len(module_files)}")
    return module_files


def import_module_from_file(module_file: Path, base_module_path: str) -> ModuleType | None:
    module_name = module_file.stem
    full_module_path = f"{base_module_path}.{module_name}"
    try:
        return importlib.import_module(full_module_path)
    except ImportError as e:
        logger.error(f"Error importing {full_module_path}: {e}", exc_info=True)
        return None


def recursive_subclasses(cls: Type[T]) -> set[Type[T]]:
    """指定クラスのすべての再帰的サブクラスを返す"""
    subclasses = set(cls.__subclasses__())
    for subclass in cls.__subclasses__():
        subclasses.update(recursive_subclasses(subclass))
    return subclasses


def gather_available_classes(directory: str) -> dict[str, ModelClass]:
    """score_models 内の全モジュールから、BaseScorer のサブクラスまたは
    predict() メソッドを持つクラスを抽出して返す"""
    available: dict[str, ModelClass] = {}
    module_files = list_module_files(directory)
    for module_file in module_files:
        module = import_module_from_file(module_file, "scorer_wrapper_lib.score_models")
        if module is None:
            continue
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # BaseScorer のサブクラスか、または predict メソッドを持つ場合､独自色の強いBLIP用
            if (issubclass(obj, BaseScorer) and obj is not BaseScorer) or hasattr(obj, "predict"):
                available[name] = obj
                for subcls in recursive_subclasses(obj):
                    available[subcls.__name__] = subcls
    logger.debug(f"SCORE_MODELSから利用可能なクラス: {list(available.keys())}")
    return available


def register_scorers() -> dict[str, ModelClass]:
    """利用可能なスコアラークラスを登録"""
    # 開始ログ
    caller = traceback.extract_stack()[-2]
    logger.debug(f"register_scorers開始: 呼び出し元={caller[0]}:{caller[1]} in {caller[2]}")
    logger.debug(f"現在のレジストリ状態: {list(_MODEL_CLASS_OBJ_REGISTRY.keys())}")

    config = load_model_config()
    logger.debug(f"設定ファイル読み込み完了: {len(config)}モデル: {list(config.keys())}")

    # 利用可能なクラスを収集
    target_directory = "score_models"
    available = gather_available_classes(target_directory)
    logger.debug(f"利用可能クラス一覧: {len(available)}クラス: {list(available.keys())}")

    # 設定ファイルに基づいてモデルとクラスのマッピングを作成
    for model_name, model_config in config.items():
        desired_class = model_config.get("class")
        if not desired_class:
            logger.warning(f"モデルにクラスが指定されていません{model_name}")
            continue

        # 利用可能なクラスから登録
        if desired_class in available:
            _MODEL_CLASS_OBJ_REGISTRY[model_name] = available[desired_class]
            logger.debug(f"モデル登録: {model_name} → {desired_class}")
        else:
            logger.error(
                f"モデル '{model_name}' で指定されたクラス '{desired_class}' は、'score_models' ディレクトリ内に定義されていません。クラス名が正しいか、または 'score_models' ディレクトリにクラスファイルが存在するか確認してください。"
            )

    logger.debug(f"register_scorers完了: 最終レジストリ状態: {list(_MODEL_CLASS_OBJ_REGISTRY.keys())}")
    return _MODEL_CLASS_OBJ_REGISTRY


def get_cls_obj_registry() -> dict[str, ModelClass]:
    """モデルクラスオブジェクトのレジストリを取得"""
    return _MODEL_CLASS_OBJ_REGISTRY


def list_available_scorers() -> list[str]:
    # TODO: スコアラー以外にも対応したら関数名は変更する
    """
    register_scorersで登録された利用可能なスコアラーモデル名のリストを返す

    Returns:
        list[str]: 設定ファイルで定義され、使用可能なモデル名のリスト

    Example:
        >>> from scorer_wrapper_lib import list_available_scorers
        >>> models = list_available_scorers()
        >>> print(models)
        ['aesthetic_shadow_v1', 'ImprovedAesthetic', 'WaifuAesthetic', ...]
    """
    return list(_MODEL_CLASS_OBJ_REGISTRY.keys())


# モジュールロード時の初期化ログ
logger.debug("モジュール初期化: scorer_registry.py がロードされました")
register_scorers()
logger.debug(f"初期レジストリ構築完了: {len(_MODEL_CLASS_OBJ_REGISTRY)}モデル登録済み")
