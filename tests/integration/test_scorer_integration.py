"""スコアラーモジュールの統合テスト

このモジュールでは、スコアラーモジュールの統合テストを実装します
"""

import random
from pathlib import Path
from PIL import Image
from pytest_bdd import given, scenarios, then, when
from typing import Any

from scorer_wrapper_lib.scorer import (
    _MODEL_INSTANCE_REGISTRY,
    get_scorer_instance,
    evaluate,
)  # type: ignore
from scorer_wrapper_lib.scorer_registry import (
    get_cls_obj_registry,
    ModelClass,
    list_available_scorers,
)  # type: ignore

scenarios("../features/scorer.feature")


# resourcesディレクトリのパス
resources_dir = Path(__file__).parent.parent / "resources"


def load_image_files(count=1):
    """指定された枚数の画像ファイルをリストとして読み込む"""
    image_path = resources_dir / "img" / "1_img"
    files = list(image_path.glob("*.webp"))

    # 指定された枚数だけファイルを取得（ディレクトリ内のファイル数を超えないように）
    count = min(count, len(files))
    files = files[:count]

    # すべての画像をリストに格納して返す
    return [Image.open(file) for file in files]


# given ----------------
@given("モデルクラスレジストリが初期化されている", target_fixture="scorer_registry")
def given_initialize_scorer_library() -> dict[str, ModelClass]:
    return get_cls_obj_registry()


@given(
    "レジストリに登録されたモデルのリストを取得する", target_fixture="available_models"
)
def given_get_available_models() -> list[str]:
    return list_available_scorers()


@given(
    "インスタンス化済みのモデルクラスが存在する", target_fixture="instantiated_models"
)
def given_instantiated_models(scorer_registry: dict[str, ModelClass]) -> dict[str, Any]:
    instantiated = {}
    # scorer_registryからモデル名を取得してインスタンス化
    for model_name in scorer_registry.keys():
        instantiated[model_name] = get_scorer_instance(model_name)
    return instantiated  # _MODEL_INSTANCE_REGISTRY と中身同じ


@given("有効な画像ファイルが準備されている", target_fixture="valid_image")
def given_valid_single_image() -> list[Image.Image]:
    return load_image_files(count=1)  # 1枚の画像を読み込む


@given("スコアラーがインスタンス化されている", target_fixture="model_for_scoring")
def given_scorer_instances(scorer_registry: dict[str, Any]) -> list[str]:
    # テスト用に単一のモデル名を返す
    return [next(iter(scorer_registry.keys()))]


@given("複数の有効な画像ファイルが準備されている", target_fixture="valid_images")
def given_valid_images_multiple() -> list[Image.Image]:
    return load_image_files(count=5)  # 明示的に5枚の画像を指定


@given("複数のモデルが指定されている", target_fixture="multiple_models")
def given_multiple_models(scorer_registry) -> list[str]:
    # 利用可能なモデル名のリスト
    available_models = list(scorer_registry.keys())

    # モデルが3つ以上ある場合は、ランダムに3つを選択
    # そうでない場合は全モデルを使用
    if len(available_models) > 3:
        selected_models = random.sample(available_models, 3)
    else:
        selected_models = available_models

    return selected_models


# when ----------------
@when(
    "これらのモデルをそれぞれインスタンス化する", target_fixture="instantiated_models"
)
def when_instantiate_all_models(available_models: list[str]) -> dict[str, Any]:
    instantiated = {}
    for model_name in available_models:
        instantiated[model_name] = get_scorer_instance(model_name)
    return instantiated


@when("同じモデルクラスを再度インスタンス化する", target_fixture="reused_instance")
def when_instantiate_same_model(instantiated_models: dict[str, Any]) -> dict:
    # 最初のモデル名を取得（どのモデルでもキャッシュ機能のテストには十分）
    model_name = next(iter(instantiated_models.keys()))

    # 元のインスタンスを記録
    original_instance = instantiated_models[model_name]

    # get_scorer_instanceを使ってキャッシュから取得（_create_scorer_instanceではない）
    reused_instance = get_scorer_instance(model_name)

    # 比較のために必要な情報を返す
    return {
        "model_name": model_name,
        "original_instance": original_instance,
        "reused_instance": reused_instance,
    }


@when("この画像をスコアリングする", target_fixture="scoring_results")
def when_score_image(
    valid_image: list[Image.Image], model_for_scoring: list[str]
) -> dict[str, list[dict[str, Any]]]:
    # 単一のモデルで評価する
    return evaluate(valid_image, model_for_scoring)


@when("これらの画像を一括評価する", target_fixture="scoring_results")
def when_score_images(
    valid_images: list[Image.Image], model_for_scoring: list[str]
) -> dict[str, list[dict[str, Any]]]:
    # 単一のモデルで複数画像を評価する
    return evaluate(valid_images, model_for_scoring)


@when("この画像を複数のモデルで評価する", target_fixture="scoring_results")
def when_score_image_multiple_models(
    valid_image: list[Image.Image], multiple_models: list[str]
) -> dict[str, list[dict[str, Any]]]:
    # 単一のモデルで複数画像を評価する
    return evaluate(valid_image, multiple_models)


@when("これらの画像を複数のモデルで一括評価する", target_fixture="scoring_results")
def when_score_images_multiple_models(
    valid_images: list[Image.Image], multiple_models: list[str]
) -> dict[str, list[dict[str, Any]]]:
    # 単一のモデルで複数画像を評価する
    return evaluate(valid_images, multiple_models)


# then ----------------
@then("各モデルが正常にインスタンス化される")
def then_all_models_instantiated(
    available_models: list[str], instantiated_models: dict[str, object]
):
    # すべてのモデルがインスタンス化されていることを確認
    for model_name in available_models:
        # キャッシュに存在することを確認
        assert model_name in _MODEL_INSTANCE_REGISTRY, (
            f"モデル '{model_name}' がキャッシュに存在しません"
        )

        # インスタンスが取得できて None でないことを確認
        model_instance = _MODEL_INSTANCE_REGISTRY[model_name]
        assert model_instance is not None, (
            f"モデル '{model_name}' のインスタンスが None です"
        )

        # 必要なメソッドが存在することを確認
        assert hasattr(model_instance, "load_or_restore_model"), (
            f"モデル '{model_name}' に load_or_restore_model メソッドがありません"
        )
        assert hasattr(model_instance, "predict"), (
            f"モデル '{model_name}' に predict メソッドがありません"
        )
        assert hasattr(model_instance, "cache_to_main_memory"), (
            f"モデル '{model_name}' に cache_to_main_memory メソッドがありません"
        )


@then("キャッシュされた同一のモデルインスタンスが返される")
def then_cached_model_instance_returned(reused_instance: dict) -> None:
    model_name = reused_instance["model_name"]
    original = reused_instance["original_instance"]
    reused = reused_instance["reused_instance"]

    # 同一のオブジェクト参照であることを確認
    assert original is reused, (
        f"モデル '{model_name}' は同じインスタンスを返していません。"
        f"キャッシュ機能が正しく動作していない可能性があります。"
    )


@then("画像に対するモデルの処理結果が返される")
def then_valid_score_returned_single_image(
    scoring_results: dict[str, list[dict[str, Any]]],
    valid_image: list[Image.Image],
) -> None:
    verify_scoring_results(scoring_results, valid_image, expect_multiple_models=False)


@then("各画像に対するモデルの処理結果が返される")
def then_valid_score_returned_multiple_images(
    scoring_results: dict[str, list[dict[str, Any]]],
    valid_images: list[Image.Image],
) -> None:
    verify_scoring_results(scoring_results, valid_images, expect_multiple_models=False)


@then("画像に対する各モデルの処理結果が返される")
def then_valid_score_returned_multiple_models(
    scoring_results: dict[str, list[dict[str, Any]]],
    valid_image: list[Image.Image],
) -> None:
    verify_scoring_results(scoring_results, valid_image, expect_multiple_models=True)


@then("各画像に対する各モデルの処理結果が返される")
def then_valid_score_returned_multiple_models_multiple_images(
    scoring_results: dict[str, list[dict[str, Any]]],
    valid_images: list[Image.Image],
) -> None:
    verify_scoring_results(scoring_results, valid_images, expect_multiple_models=True)


# 共通の検証ロジック
def verify_scoring_results(
    scoring_results: dict[str, list[dict[str, Any]]],
    images: list[Image.Image],
    expect_multiple_models: bool = False,
) -> None:
    """スコアリング結果を検証する共通ロジック

    Args:
        scoring_results: スコアリング結果
        images: 評価された画像リスト
        expect_multiple_models: 複数モデルの結果が期待されるかどうか
    """
    # 結果が存在するか確認
    assert len(scoring_results) > 0, "スコアリング結果が空です"

    # モデル数の確認
    if expect_multiple_models:
        assert len(scoring_results) > 1, "2つ以上のモデルで評価されていません"
    else:
        assert len(scoring_results) == 1, "2つ以上のモデルが評価されています"

    # 各モデルの結果をチェック
    model_count = 0
    for model_name, results in scoring_results.items():
        # 評価された画像の枚数が正しいことを確認
        assert len(results) == len(images), "評価された画像の枚数が正しくありません"

        # 結果の形式が正しいことを確認
        assert all(isinstance(result, dict) for result in results), (
            "結果の形式が不正です"
        )

        # 各結果に必要なキーが含まれているか
        for result in results:
            assert "model_name" in result, "結果に 'model_name' キーがありません"
            assert "score_tag" in result, "結果に 'score_tag' キーがありません"
            assert "model_output" in result, "結果に 'model_output' キーがありません"

        model_count += 1

    # 複数モデルの場合、モデル数が正しいか確認
    if expect_multiple_models:
        assert model_count == len(scoring_results), "モデルの数が正しくありません"
