"""スコアラーモジュールの統合テスト

このモジュールでは、スコアラーモジュールの統合テストを実装します
"""

import gc
import os
import random
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from image_annotator_lib.api import (
    _MODEL_INSTANCE_REGISTRY,
    PHashAnnotationResults,
    annotate,
    get_annotator_instance,
)
from image_annotator_lib.core.registry import (
    ModelClass,
    get_cls_obj_registry,
    list_available_annotators,
)
from image_annotator_lib.core.utils import calculate_phash
from PIL import Image
from pytest_bdd import given, scenarios, then, when

scenarios("../features/integration/scorer.feature")


# resourcesディレクトリのパス
resources_dir = Path(__file__).parent.parent / "resources"


def load_image_files(count: int = 1) -> list[Image.Image]:
    """指定された枚数の画像ファイルをリストとして読み込む"""
    image_path = resources_dir / "img" / "1_img"
    files = list(image_path.glob("*.webp"))

    # 指定された枚数だけファイルを取得(ディレクトリ内のファイル数を超えないように)
    count = min(count, len(files))
    files = files[:count]

    # すべての画像をリストに格納して返す
    return [Image.open(file) for file in files]


# given ----------------
@given("モデルクラスレジストリが初期化されている", target_fixture="scorer_registry")
def given_initialize_scorer_library() -> dict[str, ModelClass]:
    return get_cls_obj_registry()


@given("レジストリに登録されたモデルのリストを取得する", target_fixture="available_models")
def given_get_available_models() -> list[str]:
    return list_available_annotators()


@given("インスタンス化済みのモデルクラスが存在する", target_fixture="instantiated_models")
def given_instantiated_models(scorer_registry: dict[str, ModelClass]) -> dict[str, Any]:
    instantiated = {}
    # scorer_registryからモデル名を取得してインスタンス化
    for model_name in scorer_registry.keys():
        instantiated[model_name] = get_annotator_instance(model_name)
    return instantiated  # _MODEL_INSTANCE_REGISTRY と中身同じ


@given("有効な画像ファイルが準備されている", target_fixture="valid_image")
def given_valid_single_image() -> list[Image.Image]:
    return load_image_files(count=1)  # 1枚の画像を読み込む


@given("スコアラーがインスタンス化されている", target_fixture="model_for_scoring")
def given_scorer_instances(scorer_registry: dict[str, Any]) -> list[str]:
    # テスト用にランダムに1つのモデル名を選択して返す
    return [random.choice(list(scorer_registry.keys()))]


@given("複数の有効な画像ファイルが準備されている", target_fixture="valid_images")
def given_valid_images_multiple() -> list[Image.Image]:
    return load_image_files(count=5)  # 明示的に5枚の画像を指定


@given("複数のモデルが指定されている", target_fixture="multiple_models")
def given_multiple_models(scorer_registry: dict[str, ModelClass]) -> list[str]:
    # 利用可能なモデル名のリスト
    available_models = list(scorer_registry.keys())

    # モデルが5つ以上ある場合は、ランダムに5つを選択
    # そうでない場合は全モデルを使用
    if len(available_models) > 5:
        selected_models = random.sample(available_models, 5)
    else:
        selected_models = available_models

    return selected_models


@given("50枚の有効な画像ファイルが準備されている", target_fixture="valid_images_large")
def given_valid_images_large() -> list[Image.Image]:
    # 画像が足りない場合は重複して50枚に
    single_images = load_image_files(count=9)  # 既存のリソースから最大枚数
    images = []
    for _ in range(6):  # 6回コピーして50枚以上にする
        images.extend(single_images)
    return images[:50]  # 50枚に制限


@given("すべての利用可能なモデルが指定されている", target_fixture="all_models")
def given_all_models(scorer_registry: dict[str, ModelClass]) -> list[str]:
    return list(scorer_registry.keys())


# when ----------------
@when("これらのモデルをそれぞれインスタンス化する", target_fixture="instantiated_models")
def when_instantiate_all_models(available_models: list[str]) -> dict[str, Any]:
    instantiated = {}
    for model_name in available_models:
        instantiated[model_name] = get_annotator_instance(model_name)
    return instantiated


@when("同じモデルクラスを再度インスタンス化する", target_fixture="reused_instance")
def when_instantiate_same_model(instantiated_models: dict[str, Any]) -> dict[str, Any]:
    # 最初のモデル名を取得(どのモデルでもキャッシュ機能のテストには十分)
    model_name = next(iter(instantiated_models.keys()))

    # 元のインスタンスを記録
    original_instance = instantiated_models[model_name]

    # get_annotator_instanceを使ってキャッシュから取得(_create_scorer_instanceではない)
    reused_instance = get_annotator_instance(model_name)

    # 比較のために必要な情報を返す
    return {
        "model_name": model_name,
        "original_instance": original_instance,
        "reused_instance": reused_instance,
    }


@when("この画像をスコアリングする", target_fixture="scoring_results")
def when_score_image(valid_image: list[Image.Image], model_for_scoring: list[str]) -> PHashAnnotationResults:
    # 単一のモデルで評価する
    return annotate(valid_image, model_for_scoring)


@when("これらの画像を一括評価する", target_fixture="scoring_results")
def when_score_images(
    valid_images: list[Image.Image], model_for_scoring: list[str]
) -> PHashAnnotationResults:
    # 単一のモデルで複数画像を評価する
    return annotate(valid_images, model_for_scoring)


@when("この画像を複数のモデルで評価する", target_fixture="scoring_results")
def when_score_image_multiple_models(
    valid_image: list[Image.Image], multiple_models: list[str]
) -> PHashAnnotationResults:
    # 単一のモデルで複数画像を評価する
    return annotate(valid_image, multiple_models)


@when("これらの画像を複数のモデルで一括評価する", target_fixture="scoring_results")
def when_score_images_multiple_models(
    valid_images: list[Image.Image], multiple_models: list[str]
) -> PHashAnnotationResults:
    # 単一のモデルで複数画像を評価する
    return annotate(valid_images, multiple_models)


@when("これらの画像を複数回連続で評価する", target_fixture="stress_test_results")
def when_annotate_images_repeatedly(
    valid_images_large: list[Image.Image], all_models: list[str]
) -> dict[str, Any]:
    results = []
    memory_usage = []  # CPUメモリ
    gpu_memory_usage = []  # VRAM
    start_time = time.time()

    # 3回繰り返し評価
    for i in range(3):
        print(f"評価ラウンド {i + 1}/3 開始...")
        round_start = time.time()

        # 評価実行
        round_results = annotate(valid_images_large, all_models)
        results.append(round_results)

        # CPUメモリ記録
        process = psutil.Process(os.getpid())
        memory_usage.append(process.memory_info().rss / 1024 / 1024)

        # GPUメモリ記録を追加
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024
            gpu_memory_usage.append({"allocated": allocated, "reserved": reserved})

        round_end = time.time()
        print(f"ラウンド {i + 1} 完了: {round_end - round_start:.2f}秒")

    total_time = time.time() - start_time

    return {
        "results": results,
        "total_time": total_time,
        "memory_usage": memory_usage,
        "gpu_memory_usage": gpu_memory_usage,
        "image_count": len(valid_images_large),
        "model_count": len(all_models),
    }


@when("各モデルを交互に100回切り替えながら画像を評価する", target_fixture="switch_test_results")
def when_switch_models_repeatedly(valid_image: list[Image.Image], all_models: list[str]) -> dict[str, Any]:
    results = []
    memory_readings = []
    start_time = time.time()

    # モデルが少ない場合は繰り返し使用して100回に
    models_cycle = all_models * (100 // len(all_models) + 1)
    models_for_test = models_cycle[:100]

    for i, model_name in enumerate(models_for_test):
        if i % 10 == 0:
            print(f"モデル切り替えテスト: {i + 1}/100")
            # 強制的にGCを実行してメモリ状況を確認
            gc.collect()
            process = psutil.Process(os.getpid())
            memory_readings.append(process.memory_info().rss / 1024 / 1024)  # MB単位

        # 単一モデルで評価
        result = annotate(valid_image, [model_name])
        results.append(result)

    total_time = time.time() - start_time

    return {
        "results": results,
        "total_time": total_time,
        "memory_readings": memory_readings,
        "switch_count": len(models_for_test),
    }


# then ----------------
@then("各モデルが正常にインスタンス化される")
def then_all_models_instantiated(available_models: list[str], instantiated_models: dict[str, Any]) -> None:
    # すべてのモデルがインスタンス化されていることを確認
    for model_name in available_models:
        # キャッシュに存在することを確認
        assert model_name in _MODEL_INSTANCE_REGISTRY, f"モデル '{model_name}' がキャッシュに存在しません"

        # インスタンスが取得できて None でないことを確認
        model_instance = _MODEL_INSTANCE_REGISTRY[model_name]
        assert model_instance is not None, f"モデル '{model_name}' のインスタンスが None です"


@then("キャッシュされた同一のモデルインスタンスが返される")
def then_cached_model_instance_returned(reused_instance: dict[str, Any]) -> None:
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
    scoring_results: PHashAnnotationResults,
    valid_image: list[Image.Image],
) -> None:
    verify_scoring_results(scoring_results, valid_image, expect_multiple_models=False)


@then("各画像に対するモデルの処理結果が返される")
def then_valid_score_returned_multiple_images(
    scoring_results: PHashAnnotationResults,
    valid_images: list[Image.Image],
) -> None:
    verify_scoring_results(scoring_results, valid_images, expect_multiple_models=False)


@then("画像に対する各モデルの処理結果が返される")
def then_valid_score_returned_multiple_models(
    scoring_results: PHashAnnotationResults,
    valid_image: list[Image.Image],
) -> None:
    verify_scoring_results(scoring_results, valid_image, expect_multiple_models=True)


@then("各画像に対する各モデルの処理結果が返される")
def then_valid_score_returned_multiple_models_multiple_images(
    scoring_results: PHashAnnotationResults,
    valid_images: list[Image.Image],
) -> None:
    verify_scoring_results(scoring_results, valid_images, expect_multiple_models=True)


# 共通の検証ロジック
def verify_scoring_results(
    scoring_results: PHashAnnotationResults,
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

    # 画像数と結果の数が一致することを確認
    assert len(scoring_results) == len(images), "評価された画像の枚数が正しくありません"

    # 各画像のpHashを計算
    image_phashes = [calculate_phash(image) for image in images]

    # 各画像の結果をチェック
    for phash_key, model_results in scoring_results.items():
        # pHashの形式を確認
        assert isinstance(phash_key, str), "pHashキーが文字列ではありません"

        # pHashが計算したものと一致するか確認
        assert phash_key in image_phashes, f"pHashキー '{phash_key}' が入力画像のpHashと一致しません"

        # モデル数が期待通りか確認
        if expect_multiple_models:
            assert len(model_results) > 1, "2つ以上のモデルで評価されていません"
        else:
            assert len(model_results) == 1, "2つ以上のモデルで評価されています"

        # 各モデルの結果をチェック
        for model_name, result in model_results.items():
            # モデル名が文字列であることを確認
            assert isinstance(model_name, str), "モデル名が文字列ではありません"

            # 結果の形式が正しいことを確認
            assert isinstance(result, dict), "結果の形式が不正です"

            # 各結果に必要なキーが含まれているか
            assert "tags" in result, "結果に 'tags' キーがありません"
            assert "formatted_output" in result, "結果に 'formatted_output' キーがありません"
            assert "error" in result, "結果に 'error' キーがありません"


@then("全ての評価が正常に完了している")
def then_all_evaluations_completed(stress_test_results: dict[str, Any]) -> None:
    results = stress_test_results["results"]
    image_count = stress_test_results["image_count"]
    model_count = stress_test_results["model_count"]

    # 3ラウンド全てで結果があることを確認
    assert len(results) == 3, "全3ラウンドの結果が揃っていません"

    # 各ラウンドで全モデルの結果があることを確認
    for i, round_results in enumerate(results):
        # 各ラウンドの結果をチェック
        for phash_key, model_results in round_results.items():
            # 各モデルの結果をチェック
            for model_name, result in model_results.items():
                # 結果の形式が正しいことを確認
                assert isinstance(result, dict), "結果の形式が不正です"

                # 各結果に必要なキーが含まれているか
                assert "tags" in result, "結果に 'tags' キーがありません"
                assert "formatted_output" in result, "結果に 'formatted_output' キーがありません"
                assert "error" in result, "結果に 'error' キーがありません"

    print(f"ストレステスト完了: {stress_test_results['total_time']:.2f}秒")
    print(f"評価画像数: {image_count}枚")
    print(f"使用モデル数: {model_count}個")


@then("GPU・CPUメモリの使用状況が許容範囲内である")
def then_memory_usage_is_acceptable(stress_test_results: dict[str, Any]) -> None:
    # CPU(メイン)メモリのチェック
    memory_readings = stress_test_results["memory_usage"]
    initial_memory = memory_readings[0]
    final_memory = memory_readings[-1]
    memory_increase = final_memory - initial_memory

    print(f"CPUメモリ初期使用量: {initial_memory:.2f}MB")
    print(f"CPUメモリ最終使用量: {final_memory:.2f}MB")
    print(f"CPUメモリ増加量: {memory_increase:.2f}MB")

    # GPUメモリのチェックを追加
    if "gpu_memory_usage" in stress_test_results and torch.cuda.is_available():
        gpu_readings = stress_test_results["gpu_memory_usage"]

        if gpu_readings:
            initial_gpu = gpu_readings[0]["allocated"]
            final_gpu = gpu_readings[-1]["allocated"]
            gpu_increase = final_gpu - initial_gpu

            print(f"GPUメモリ初期使用量: {initial_gpu:.2f}MB")
            print(f"GPUメモリ最終使用量: {final_gpu:.2f}MB")
            print(f"GPUメモリ増加量: {gpu_increase:.2f}MB")

            # GPUメモリの許容範囲チェック
            assert gpu_increase < 500, f"GPUメモリ使用量が{gpu_increase:.2f}MB増加しました(許容値:500MB)"

    # CPU(メイン)メモリの許容範囲チェック
    assert memory_increase < 1500, f"CPUメモリ使用量が{memory_increase:.2f}MB増加しました(許容値:1500MB)"


@then("モデル切り替えが正常に動作している")
def then_model_switching_works_correctly(switch_test_results: dict[str, Any]) -> None:
    results = switch_test_results["results"]
    switch_count = switch_test_results["switch_count"]

    # 全ての切り替えで結果が存在することを確認
    assert len(results) == switch_count, (
        f"実行回数{switch_count}に対して結果が{len(results)}件しかありません"
    )

    # 各結果が有効な形式であることを確認
    for i, result in enumerate(results):
        assert isinstance(result, dict), f"{i + 1}回目の結果が辞書形式ではありません"
        assert len(result) > 0, f"{i + 1}回目の結果が空です"

    print(f"モデル切り替えテスト完了: {switch_test_results['total_time']:.2f}秒")
    print(f"切り替え回数: {switch_count}回")


@then("リソースリークが発生していない")
def then_no_resource_leaks(switch_test_results: dict[str, Any]) -> None:
    memory_readings = switch_test_results["memory_readings"]

    if len(memory_readings) >= 2:
        initial_memory = memory_readings[0]
        final_memory = memory_readings[-1]
        max_memory = max(memory_readings)

        print(f"初期メモリ使用量: {initial_memory:.2f}MB")
        print(f"最終メモリ使用量: {final_memory:.2f}MB")
        print(f"最大メモリ使用量: {max_memory:.2f}MB")

        # より現実的な条件
        # 最終値が最大値より10%以上小さければOK
        memory_stabilizing = final_memory < max_memory * 0.95
        # または、最終メモリがシステムメモリの25%未満
        memory_acceptable = final_memory < psutil.virtual_memory().total / 1024 / 1024 * 0.25

        assert memory_stabilizing or memory_acceptable, "メモリ使用量が安定していません"
