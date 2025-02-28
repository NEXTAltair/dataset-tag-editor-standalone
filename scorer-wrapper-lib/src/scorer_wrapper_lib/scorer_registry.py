import importlib
import inspect
import logging
from pathlib import Path

from .core.base import BaseScorer
from .core.utils import load_model_config

_SCORER_REGISTRY = {}
_MODEL_NAME_TO_CLASS = {}  # モデル名とクラス名の対応を保持する辞書
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


def import_module_from_file(module_file: Path, base_module_path: str):
    module_name = module_file.stem
    full_module_path = f"{base_module_path}.{module_name}"
    try:
        return importlib.import_module(full_module_path)
    except ImportError as e:
        logger.error(f"Error importing {full_module_path}: {e}", exc_info=True)
        return None


def recursive_subclasses(cls) -> set:
    """指定クラスのすべての再帰的サブクラスを返す"""
    subclasses = set(cls.__subclasses__())
    for subclass in cls.__subclasses__():
        subclasses.update(recursive_subclasses(subclass))
    return subclasses


def gather_available_classes(directory: str) -> dict:
    """score_models 内の全モジュールから、BaseScorer のサブクラスまたは
    predict() メソッドを持つクラスを抽出して返す"""
    available = {}
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


def gather_core_classes() -> dict:
    """core.base 内の BaseScorer サブクラスを抽出して返す"""
    core_module = importlib.import_module(".core.base", package="scorer_wrapper_lib")
    core = {}
    for name, obj in inspect.getmembers(core_module, inspect.isclass):
        if issubclass(obj, BaseScorer) and obj is not BaseScorer:
            core[name] = obj
    return core


def register_models_from_config(config: dict, available: dict, core: dict) -> None:
    """config の各モデル設定に基づき、利用可能なクラスまたはコアクラスからレジストリに登録する"""
    for model_name, model_config in config.items():
        desired_class = model_config.get("class")
        if not desired_class:
            logger.warning(f"Model {model_name} missing class definition.")
            continue
        if desired_class in available:
            _SCORER_REGISTRY[desired_class] = available[desired_class]
            _MODEL_NAME_TO_CLASS[model_name] = desired_class
            logger.debug(f"Registered {model_name} with class {desired_class} from score_models")
        elif desired_class in core:
            _SCORER_REGISTRY[desired_class] = core[desired_class]
            _MODEL_NAME_TO_CLASS[model_name] = desired_class
            logger.debug(f"Registered {model_name} with class {desired_class} from core.base")
        else:
            logger.error(f"Class {desired_class} not found for model {model_name}.")


def register_scorers() -> dict:
    """メインのレジストリ登録処理。"""
    config = load_model_config()
    target_directory = "score_models"
    available = gather_available_classes(target_directory)
    core = gather_core_classes()
    # 既に core.base から登録したものをレジストリに追加
    for name, obj in core.items():
        _SCORER_REGISTRY[name] = obj
    register_models_from_config(config, available, core)
    return _SCORER_REGISTRY


def get_registry() -> dict:
    return _SCORER_REGISTRY


def list_available_scorers() -> list[str]:
    return list(_MODEL_NAME_TO_CLASS.keys())


register_scorers()
