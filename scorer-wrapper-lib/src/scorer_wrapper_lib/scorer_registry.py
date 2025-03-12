import importlib
import inspect
import logging
from pathlib import Path
from types import ModuleType
from typing import Type, TypeVar

from .core.base import BaseScorer
from .core.utils import load_model_config

T = TypeVar("T", bound=BaseScorer)
ScorerClass = Type[BaseScorer]

_SCORER_REGISTRY: dict[str, ScorerClass] = {}  # クラス名 → クラス定義
_MODEL_TO_CLASS_MAP: dict[str, str] = {}  # モデル名 → クラス名
logger = logging.getLogger("scorer_wrapper_lib")


def list_module_files(directory: str) -> list[Path]:
    base_dir = Path(__file__).parent
    abs_path = (base_dir / directory).resolve()
    logger.debug(f"base_dir: {base_dir}, searching in abs_path: {abs_path}")
    module_files = []
    for p in abs_path.glob("*.py"):
        if p.name != "__init__.py":
            logger.debug(f"Found module file: {p}")
            module_files.append(p)
    logger.debug(f"Total module files found: {len(module_files)}")
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


def gather_available_classes(directory: str) -> dict[str, ScorerClass]:
    """score_models 内の全モジュールから、BaseScorer のサブクラスまたは
    predict() メソッドを持つクラスを抽出して返す"""
    available: dict[str, ScorerClass] = {}
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


def gather_core_classes() -> dict[str, ScorerClass]:
    """core.base 内の BaseScorer サブクラスを抽出して返す"""
    core_module = importlib.import_module(".core.base", package="scorer_wrapper_lib")
    core: dict[str, ScorerClass] = {}
    for name, obj in inspect.getmembers(core_module, inspect.isclass):
        if issubclass(obj, BaseScorer) and obj is not BaseScorer:
            core[name] = obj
    return core


def register_scorers() -> dict[str, ScorerClass]:
    """利用可能なスコアラークラスを登録し、モデル名とクラス名のマッピングも作成"""
    config = load_model_config()

    # 利用可能なクラスを収集
    target_directory = "score_models"
    available = gather_available_classes(target_directory)
    core = gather_core_classes()

    # コアクラスをレジストリに追加
    for name, obj in core.items():
        _SCORER_REGISTRY[name] = obj

    # 設定ファイルに基づいてモデルとクラスのマッピングを作成
    for model_name, model_config in config.items():
        desired_class = model_config.get("class")
        if not desired_class:
            logger.warning(f"No class specified for model {model_name}")
            continue

        # モデル名→クラス名のマッピングを保存
        _MODEL_TO_CLASS_MAP[model_name] = desired_class

        # 利用可能なクラスから登録
        if desired_class in available:
            _SCORER_REGISTRY[desired_class] = available[desired_class]
            logger.debug(f"Registered {desired_class} from score_models")
        else:
            logger.error(f"Class {desired_class} not found for model {model_name}")

    return _SCORER_REGISTRY


def get_registry() -> dict[str, ScorerClass]:
    """スコアラークラスのレジストリを取得"""
    return _SCORER_REGISTRY


def get_class_for_model(model_name: str) -> str:
    """モデル名に対応するクラス名を取得"""
    if model_name not in _MODEL_TO_CLASS_MAP:
        raise ValueError(f"Model not found or has no class specified: {model_name}")
    return _MODEL_TO_CLASS_MAP[model_name]


def list_available_scorers() -> list[str]:
    """
    利用可能なスコアラーモデル名のリストを返す

    Returns:
        list[str]: 設定ファイルで定義され、使用可能なモデル名のリスト

    Example:
        >>> from scorer_wrapper_lib import list_available_scorers
        >>> models = list_available_scorers()
        >>> print(models)
        ['aesthetic_shadow_v1', 'ImprovedAesthetic', 'WaifuAesthetic', ...]
    """
    return list(_MODEL_TO_CLASS_MAP.keys())


register_scorers()
