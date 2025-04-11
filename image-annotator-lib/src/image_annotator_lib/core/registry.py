import importlib
import inspect
import logging
from pathlib import Path
from types import ModuleType
from typing import TypeVar

from .base import BaseAnnotator
from .config import config_registry

T = TypeVar("T", bound=BaseAnnotator)
ModelClass = type[BaseAnnotator]

_MODEL_CLASS_OBJ_REGISTRY: dict[str, ModelClass] = {}

logger = logging.getLogger(__name__)

# --- プライベートヘルパー関数 ---


def _list_module_files(directory: str) -> list[Path]:
    """指定されたディレクトリ内のPythonモジュールファイル (__init__.pyを除く) をリストアップします。"""
    try:
        # 'directory'が'src/image_annotator_lib'ディレクトリからの相対パスであると仮定
        base_dir = Path(__file__).parent.parent
        abs_path = (base_dir / directory).resolve()
        logger.debug(f"モジュール検索中: {abs_path}")
        if not abs_path.is_dir():
            logger.warning(f"モジュールディレクトリが見つかりません: {abs_path}")
            return []
        module_files = [p for p in abs_path.glob("*.py") if p.name != "__init__.py"]
        logger.debug(f"{abs_path} で {len(module_files)} 個のモジュールファイルが見つかりました")
        return module_files
    except Exception as e:
        logger.error(f"{directory} 内のモジュールファイルのリストアップ中にエラー: {e}", exc_info=True)
        return []


def _import_module_from_file(module_file: Path, base_module_path: str) -> ModuleType | None:
    """ファイルパスからPythonモジュールをインポートします。"""
    module_name = module_file.stem
    full_module_path = f"{base_module_path}.{module_name}"
    try:
        module = importlib.import_module(full_module_path)
        logger.debug(f"モジュールのインポート成功: {full_module_path}")
        return module
    except ImportError as e:
        # ImportErrorは一般的であるため、具体的にログに記録
        logger.error(f"モジュール {full_module_path} のインポート中にエラー: {e}", exc_info=True)
        return None
    except Exception as e:
        # インポート中に発生する可能性のある他の例外をキャッチ
        logger.error(f"モジュール {full_module_path} のインポート中に予期せぬエラー: {e}", exc_info=True)
        return None


def _recursive_subclasses(cls: type[T]) -> set[type[T]]:
    """指定されたクラスのすべての再帰的サブクラスを返します。"""
    # この関数はそのままですが、他で使用する場合はTが適切にバインドされていることを確認してください
    # _gather_available_classesでは、BaseAnnotatorを直接使用します。
    subclasses: set[type[T]] = set()
    try:
        # より良い互換性のためにinspect.getmro()を使用し、__subclasses__をチェック
        for subclass in cls.__subclasses__():
            subclasses.add(subclass)
            subclasses.update(_recursive_subclasses(subclass))  # type: ignore
    except TypeError as e:
        logger.warning(f"{cls.__name__} のサブクラスを取得できませんでした (C拡張の可能性があります): {e}")
    except Exception as e:
        logger.error(f"{cls.__name__} のサブクラス検索中にエラー: {e}", exc_info=True)
    return subclasses


def _gather_available_classes(directory: str) -> dict[str, ModelClass]:
    """指定ディレクトリ内の全モジュールから、BaseAnnotator のサブクラスまたは
    predict() メソッドを持つクラスを抽出して返す"""
    available: dict[str, ModelClass] = {}
    module_files = _list_module_files(directory)
    for module_file in module_files:
        # モジュールインポートパスを修正 (ユーザー変更を反映)
        module = _import_module_from_file(module_file, "image_annotator_lib.model_class")
        if module is None:
            continue
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # BaseAnnotator のサブクラスか、または predict メソッドを持つ場合 (ユーザー変更を反映)
            # objがtypeであることを確認 (Mypy対策)
            if isinstance(obj, type) and (
                (issubclass(obj, BaseAnnotator) and obj is not BaseAnnotator) or hasattr(obj, "predict")
            ):
                # Mypyエラー回避のため issubclass チェックを追加
                if issubclass(obj, BaseAnnotator):
                    available[name] = obj
                    # 再帰的サブクラス収集 (ユーザー変更を反映)
                    for subcls in _recursive_subclasses(obj):
                        available[subcls.__name__] = subcls
                elif hasattr(
                    obj, "predict"
                ):  # predictを持つがBaseAnnotatorサブクラスでない場合も登録 (元のロジック踏襲)
                    # このケースでは型が合わない可能性があるため注意が必要だが、元のロジックを維持
                    # 必要であれば、より厳密な型チェックや別の扱いを検討
                    available[name] = obj  # type: ignore

    logger.debug(f"{directory} から利用可能なクラス: {list(available.keys())}")
    return available


def _register_models(
    registry: dict[str, ModelClass],  # ModelClass を使用
    model_type_name: str,  # ログ用の "annotator" など
    directory: str,  # モデルクラスファイルを含むディレクトリ (例: "model_class")
    base_module_path: str,  # 基本Pythonインポートパス (例: "image_annotator_lib.model_class")
    base_class: type,  # モデルが継承すべき基底クラス (ABC互換性のために 'type' を使用)
    # config_filter は削除
) -> None:
    """設定に基づいてモデルを登録する汎用関数。"""
    current_frame = inspect.currentframe()
    caller_frame = current_frame.f_back if current_frame else None
    caller_info = inspect.getframeinfo(caller_frame) if caller_frame else None
    caller_repr = (
        f"{caller_info.filename}:{caller_info.lineno} in {caller_info.function}"
        if caller_info
        else "不明な呼び出し元"
    )

    logger.debug(f"{model_type_name} モデルの登録を開始します。呼び出し元: {caller_repr}")
    logger.debug(f"現在のレジストリの状態: {list(registry.keys())}")

    registered_count = 0  # 変数を初期化

    try:
        # config_registry からロード済みの設定データを取得
        config = config_registry.get_all_config()

        if not config:
            logger.warning("モデル設定が空か、ロードに失敗しました。モデルは登録されません。")
            return
        logger.debug(f"モデル設定をロードしました: 合計 {len(config)} エントリ。")

        # 指定されたディレクトリから利用可能なクラスを収集
        available_classes = _gather_available_classes(directory)
        logger.debug(
            f"{len(available_classes)} 個の利用可能な {model_type_name} クラスが見つかりました: {list(available_classes.keys())}"
        )

        registered_count = len(available_classes)  # 成功時にカウントを更新
        # 設定ファイルに基づいてモデルを登録
        for model_name, model_config in config.items():
            # config_filter によるフィルタリングを削除

            desired_class_name = model_config.get("class")
            if not desired_class_name:
                logger.warning(
                    f"設定でモデル '{model_name}' のクラス名が指定されていません。スキップします。"
                )
                continue

            # 収集された利用可能なクラスからクラスを検索
            model_cls = available_classes.get(desired_class_name)

            if model_cls:
                # 期待されるbase_classのサブクラスであるか、または predict を持つかを確認 (元のロジック踏襲)
                if issubclass(model_cls, base_class) or hasattr(model_cls, "predict"):
                    if model_name in registry:
                        logger.warning(
                            f"モデル名 '{model_name}' は既に登録されています。クラス '{model_cls.__name__}' で上書きします。"
                        )
                    registry[model_name] = model_cls
                    logger.info(
                        f"{model_type_name} モデルを登録しました: '{model_name}' -> {model_cls.__name__}"
                    )
                else:
                    logger.error(
                        f"モデル '{model_name}' のクラス '{desired_class_name}' が見つかりましたが、{base_class.__name__} を継承しておらず predict メソッドも持ちません。スキップします。"
                    )
            else:
                # 設定で指定されたクラスがスキャンされたディレクトリで見つかりません
                logger.warning(
                    f"{model_type_name} モデル '{model_name}' で指定されたクラス '{desired_class_name}' が、'{directory}' 内の利用可能なクラスの中に見つかりません。"
                    f"クラス名が正しいか、それを定義するファイルが '{base_module_path}' に存在するか確認してください。"
                )

    except Exception as e:
        logger.error(
            f"予期せぬエラーのため {model_type_name} モデルの登録に失敗しました: {e}", exc_info=True
        )

    logger.debug(
        f"{model_type_name} モデルの登録が完了しました。登録済み合計: {registered_count}。最終レジストリ状態: {list(registry.keys())}"
    )


# --- パブリックAPI ---


def register_annotators() -> dict[str, ModelClass]:
    """
    設定ファイルに基づいて、利用可能なすべてのアノテータクラスを登録します。

    Returns:
        dict[str, ModelClass]: 登録されたモデルクラスのレジストリ。
    """
    logger.info("すべてのアノテータを登録中...")
    _register_models(
        registry=_MODEL_CLASS_OBJ_REGISTRY,
        model_type_name="annotator",
        directory="model_class",
        base_module_path="image_annotator_lib.model_class",
        base_class=BaseAnnotator,
    )
    logger.info("アノテータの登録が完了しました。")
    return _MODEL_CLASS_OBJ_REGISTRY


# TODO[DESIGN]: レジストリの責務分離を検討
# - 現状: クラス登録とインスタンス管理が分離している(_MODEL_CLASS_OBJ_REGISTRYとapi.pyの_MODEL_INSTANCE_REGISTRY)
# - 課題:
#   1. レジストリ関連の機能が複数箇所に分散
#   2. get_cls_obj_registry()が内部実装を直接露出
# - 改善案:
#   - レジストリ機能の一元管理を検討
#   - ただし、既存コードへの影響が大きいため、メジャーバージョンアップ時に対応
def get_cls_obj_registry() -> dict[str, ModelClass]:
    """モデルクラスオブジェクトのレジストリを取得

    Note:
        このAPIは非推奨となる可能性があります。
        将来的にはより適切なインターフェースに置き換えられる可能性があります。

    Returns:
        dict[str, ModelClass]: 登録されたモデルクラスのレジストリ
    """
    return _MODEL_CLASS_OBJ_REGISTRY


# list_available_annotators のような統合されたリスト関数に変更 (あるいは既存を維持)
def list_available_annotators() -> list[str]:
    """利用可能なアノテータモデルの名前のリストを返します。"""
    return list(_MODEL_CLASS_OBJ_REGISTRY.keys())


# --- 初期化 ---
# モジュールがロードされるときにモデルを登録します。
logger.debug("アノテータレジストリを初期化中...")
register_annotators()
logger.debug(f"レジストリの初期化が完了しました。登録済みアノテータ: {len(_MODEL_CLASS_OBJ_REGISTRY)}")
