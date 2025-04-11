"""タガーモジュールの統合テスト

このモジュールでは、タガーモジュールの統合テストを実装します
"""

import gc
import os
import random
import time
from typing import Any

import psutil
import pytest
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
from PIL import Image
from pytest_bdd import given, scenarios, then, when

scenarios("../features/integration/tagger.feature")


# given ----------------
@given("モデルクラスレジストリが初期化されている", target_fixture="tagger_registry")
def given_initialize_tagger_library() -> dict[str, ModelClass]:
    return get_cls_obj_registry()


@given("レジストリに登録されたモデルのリストを取得する", target_fixture="available_models")
def given_get_available_models() -> list[str]:
    return list_available_annotators()


@given("インスタンス化済みのモデルクラスが存在する", target_fixture="instantiated_models")
def given_instantiated_models(tagger_registry: dict[str, ModelClass]) -> dict[str, Any]:
    instantiated = {}
    # tagger_registryからモデル名を取得してインスタンス化
    for model_name in tagger_registry.keys():
        instantiated[model_name] = get_annotator_instance(model_name)
    return instantiated  # _MODEL_INSTANCE_REGISTRY と中身同じ


@given("有効な画像ファイルが準備されている", target_fixture="valid_image")
def given_valid_single_image(load_image_files: list[Image.Image]) -> list[Image.Image]:
    return load_image_files(count=1)  # 1枚の画像を読み込む


@given("タガーがインスタンス化されている", target_fixture="model_for_tagging")
def given_tagger_instances(tagger_registry: dict[str, Any]) -> list[str]:
    # テスト用に単一のモデル名を返す
    return [next(iter(tagger_registry.keys()))]


@given("複数の有効な画像ファイルが準備されている", target_fixture="valid_images")
def given_valid_images_multiple(load_image_files: list[Image.Image]) -> list[Image.Image]:
    return load_image_files(count=5)  # 明示的に5枚の画像を指定


@given("複数のモデルが指定されている", target_fixture="multiple_models")
def given_multiple_models(tagger_registry: dict[str, Any]) -> list[str]:
    # 利用可能なモデル名のリスト
    available_models = list(tagger_registry.keys())

    # モデルが3つ以上ある場合は、ランダムに3つを選択
    # そうでない場合は全モデルを使用
    if len(available_models) > 3:
        selected_models = random.sample(available_models, 3)
    else:
        selected_models = available_models

    return selected_models


@given("30枚の有効な画像ファイルが準備されている", target_fixture="valid_images_large")
def given_valid_images_large() -> list[Image.Image]:
    # 画像が足りない場合は重複して30枚に
    single_images = load_image_files(count=9)  # 既存のリソースから最大枚数
    images = []
    for _ in range(4):  # 4回コピーして50枚以上にする
        images.extend(single_images)
    return images[:30]  # 30枚に制限


@given("すべての利用可能なモデルが指定されている", target_fixture="all_models")
def given_all_models(tagger_registry) -> list[str]:
    return list(tagger_registry.keys())


# when ----------------
@when("これらのモデルをそれぞれインスタンス化する", target_fixture="instantiated_models")
def when_instantiate_all_models(available_models: list[str]) -> dict[str, Any]:
    instantiated = {}
    for model_name in available_models:
        instantiated[model_name] = get_annotator_instance(model_name)
    return instantiated


@when("同じモデルクラスを再度インスタンス化する", target_fixture="reused_instance")
def when_instantiate_same_model(instantiated_models: dict[str, Any]) -> dict:
    # 最初のモデル名を取得(どのモデルでもキャッシュ機能のテストには十分)
    model_name = next(iter(instantiated_models.keys()))

    # 元のインスタンスを記録
    original_instance = instantiated_models[model_name]

    # get_tagger_instanceを使ってキャッシュから取得(_create_tagger_instanceではない)
    reused_instance = get_annotator_instance(model_name)

    # 比較のために必要な情報を返す
    return {
        "model_name": model_name,
        "original_instance": original_instance,
        "reused_instance": reused_instance,
    }


@when("この画像をタグ付けする", target_fixture="tagging_results")
def when_tag_image(valid_image: list[Image.Image], model_for_tagging: list[str]) -> PHashAnnotationResults:
    # 単一のモデルで評価する
    return annotate(valid_image, model_for_tagging)


@when("これらの画像を一括アノテーションを実行", target_fixture="tagging_results")
def when_tag_images(
    valid_images: list[Image.Image], model_for_tagging: list[str]
) -> PHashAnnotationResults:
    # 単一のモデルで複数画像を評価する
    return annotate(valid_images, model_for_tagging)


@when("この画像を複数のモデルでアノテーションを実行", target_fixture="tagging_results")
def when_tag_image_multiple_models(
    valid_image: list[Image.Image], multiple_models: list[str]
) -> PHashAnnotationResults:
    # 単一のモデルで複数画像を評価する
    return annotate(valid_image, multiple_models)


@when("これらの画像を複数のモデルで一括アノテーションを実行", target_fixture="tagging_results")
def when_tag_images_multiple_models(
    valid_images: list[Image.Image], multiple_models: list[str]
) -> PHashAnnotationResults:
    # 単一のモデルで複数画像を評価する
    return annotate(valid_images, multiple_models)


@when("これらの画像を複数回連続でアノテーションを実行", target_fixture="test_results")
def when_annotate_images_repeatedly(valid_images_large: list[Image.Image], all_models: list[str]) -> dict:
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


@when("各モデルを交互に100回切り替えながら画像をアノテーションを実行", target_fixture="test_results")
def when_switch_models_repeatedly(valid_image: list[Image.Image], all_models: list[str]) -> dict:
    results = []
    memory_usage = []
    gpu_memory_usage = []
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
            memory_usage.append(process.memory_info().rss / 1024 / 1024)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                gpu_memory_usage.append({"allocated": allocated, "reserved": reserved})
        # 単一モデルで評価
        result = annotate(valid_image, [model_name])
        results.append(result)

    total_time = time.time() - start_time

    return {
        "results": results,
        "total_time": total_time,
        "memory_usage": memory_usage,
        "gpu_memory_usage": gpu_memory_usage,
        "image_count": len(valid_image),
        "model_count": len(all_models),
        "switch_count": 100,
    }


# then ----------------
@then("各モデルが正常にインスタンス化される")
def then_all_models_instantiated(available_models: list[str], instantiated_models: dict[str, object]):
    # すべてのモデルがインスタンス化されていることを確認
    for model_name in available_models:
        # キャッシュに存在することを確認
        assert model_name in _MODEL_INSTANCE_REGISTRY, f"モデル '{model_name}' がキャッシュに存在しません"

        # インスタンスが取得できて None でないことを確認
        model_instance = _MODEL_INSTANCE_REGISTRY[model_name]
        assert model_instance is not None, f"モデル '{model_name}' のインスタンスが None です"

        # 必要なメソッドが存在することを確認
        assert hasattr(model_instance, "predict"), f"モデル '{model_name}' に predict メソッドがありません"
        assert hasattr(model_instance, "_generate_tags"), (
            f"モデル '{model_name}' に _generate_tags メソッドがありません"
        )
        assert hasattr(model_instance, "_generate_result"), (
            f"モデル '{model_name}' に _generate_result メソッドがありません"
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
def then_valid_tag_returned_single_image(
    tagging_results: dict[str, dict[str, dict[str, Any]]],
    valid_image: list[Image.Image],
) -> None:
    verify_tagging_results(tagging_results, valid_image, expect_multiple_models=False)


@then("各画像に対するモデルの処理結果が返される")
def then_valid_tag_returned_multiple_images(
    tagging_results: dict[str, dict[str, dict[str, Any]]],
    valid_images: list[Image.Image],
) -> None:
    verify_tagging_results(tagging_results, valid_images, expect_multiple_models=False)


@then("画像に対する各モデルの処理結果が返される")
def then_valid_tag_returned_multiple_models(
    tagging_results: dict[str, dict[str, dict[str, Any]]],
    valid_image: list[Image.Image],
) -> None:
    verify_tagging_results(tagging_results, valid_image, expect_multiple_models=True)


@then("各画像に対する各モデルの処理結果が返される")
def then_valid_tag_returned_multiple_models_multiple_images(
    tagging_results: dict[str, dict[str, dict[str, Any]]],
    valid_images: list[Image.Image],
) -> None:
    verify_tagging_results(tagging_results, valid_images, expect_multiple_models=True)


# 共通の検証ロジック
def verify_tagging_results(
    tagging_results: dict[str, dict[str, dict[str, Any]]],
    images: list[Image.Image],
    expect_multiple_models: bool = False,
) -> None:
    """タグ付け結果を検証する共通ロジック

    Args:
        tagging_results: タグ付け結果
        images: 評価された画像リスト
        expect_multiple_models: 複数モデルの結果が期待されるかどうか
    """
    # 結果が存在するか確認
    assert len(tagging_results) > 0, "タグ付け結果が空です (pHashが見つかりません)"
    assert len(tagging_results) == len(images), (
        "評価された画像の枚数 (pHashの数) が入力画像数と一致しません"
    )

    # 最初の画像の結果からモデル数を取得して確認
    first_image_results = next(iter(tagging_results.values()))
    num_models_in_result = len(first_image_results)
    if expect_multiple_models:
        assert num_models_in_result > 1, "結果に複数のモデルが含まれていません"
    else:
        assert num_models_in_result == 1, "結果に単一のモデルのみが含まれていません"

    # 各モデルの結果をチェック
    # 各画像の結果をチェック
    for phash, model_results in tagging_results.items():
        # この画像のモデル数が期待通りか確認
        assert len(model_results) == num_models_in_result, (
            f"画像 {phash} のモデル数が期待値 ({num_models_in_result}) と異なります"
        )

        # 各モデルの結果をチェック
        for model_name, result_dict in model_results.items():
            # 結果の形式が正しいことを確認 (result_dictが辞書であること)
            assert isinstance(result_dict, dict), (
                f"画像 {phash}, モデル {model_name} の結果形式が不正です (dictではありません)"
            )

            # 必要なキーが含まれているか確認
            assert "tags" in result_dict, (
                f"画像 {phash}, モデル {model_name} の結果に 'tags' キーがありません"
            )
            assert "formatted_output" in result_dict, (
                f"画像 {phash}, モデル {model_name} の結果に 'formatted_output' キーがありません"
            )
            assert "error" in result_dict, (
                f"画像 {phash}, モデル {model_name} の結果に 'error' キーがありません"
            )
            # エラーチェック
            error_message = result_dict["error"]
            if error_message is not None:
                # メモリ不足エラーの場合はテストをスキップ
                if error_message == "メモリ不足エラー":  # 文字列完全一致で判定
                    pytest.skip(f"メモリ不足エラーのためスキップ: 画像 {phash}, モデル {model_name}")
                else:
                    # その他の予期せぬエラーはテストを失敗させる
                    pytest.fail(
                        f"画像 {phash}, モデル {model_name} で予期せぬエラーが発生しました: {error_message}"
                    )


@then("全ての評価が正常に完了している")
def then_all_evaluations_completed(test_results: dict) -> None:
    results = test_results["results"]
    image_count = test_results["image_count"]
    model_count = test_results["model_count"]

    # 3ラウンド全てで結果があることを確認
    assert len(results) == 3, "全3ラウンドの結果が揃っていません"

    # 各ラウンドで全モデルの結果があることを確認
    for i, round_results in enumerate(results):
        assert len(round_results) == model_count, f"ラウンド{i + 1}で一部のモデル結果が欠落しています"

        # 各モデルの結果が画像数と一致していることを確認
        for model_name, model_results in round_results.items():
            assert len(model_results) == image_count, (
                f"ラウンド{i + 1}のモデル{model_name}で画像{image_count}枚分の結果がありません"
            )

    print(f"ストレステスト完了: {test_results['total_time']:.2f}秒")
    print(f"評価画像数: {image_count}枚")
    print(f"使用モデル数: {model_count}個")


@then("モデル切り替えが正常に動作している")
def then_model_switching_works_correctly(test_results: dict) -> None:
    results = test_results["results"]
    switch_count = test_results["switch_count"]

    # 全ての切り替えで結果が存在することを確認
    assert len(results) == switch_count, (
        f"実行回数{switch_count}に対して結果が{len(results)}件しかありません"
    )

    # 各結果が有効な形式であることを確認
    for i, result in enumerate(results):
        assert isinstance(result, dict), f"{i + 1}回目の結果が辞書形式ではありません"
        assert len(result) > 0, f"{i + 1}回目の結果が空です"

    print(f"モデル切り替えテスト完了: {test_results['total_time']:.2f}秒")
    print(f"切り替え回数: {switch_count}回")


@then("リソースリークが発生していない")
def then_no_resource_leaks(test_results: dict) -> None:  # 引数名を汎用的な test_results に戻す
    """メモリ使用量の増加とリーク傾向をチェックする"""

    # 渡された結果がどちらのテストのものか判別
    is_stress_test = (
        "memory_usage" in test_results and len(test_results["memory_usage"]) == 3
    )  # ストレステストは3回記録
    is_switch_test = (
        "memory_usage" in test_results and len(test_results["memory_usage"]) == 10
    )  # 切り替えテストは10回記録

    if is_stress_test:
        # CPU(メイン)メモリのチェック
        memory_readings = test_results["memory_usage"]
        if len(memory_readings) >= 2:
            initial_memory = memory_readings[0]  # 最初の測定値
            final_memory = memory_readings[-1]
            memory_increase = final_memory - initial_memory

            print(f"[ストレステスト] CPUメモリ初期使用量: {initial_memory:.2f}MB")
            print(f"[ストレステスト] CPUメモリ最終使用量: {final_memory:.2f}MB")
            print(f"[ストレステスト] CPUメモリ増加量: {memory_increase:.2f}MB")
            # CPU(メイン)メモリの許容範囲チェック
            assert memory_increase < 1500, (
                f"CPUメモリ使用量が{memory_increase:.2f}MB増加しました(許容値:1500MB)"
            )

        # GPUメモリのチェックを追加
        if "gpu_memory_usage" in test_results and torch.cuda.is_available():
            gpu_readings = test_results["gpu_memory_usage"]
            if len(gpu_readings) >= 2:
                initial_gpu = gpu_readings[0]["allocated"]
                final_gpu = gpu_readings[-1]["allocated"]
                gpu_increase = final_gpu - initial_gpu

                print(f"[ストレステスト] GPUメモリ初期使用量: {initial_gpu:.2f}MB")
                print(f"[ストレステスト] GPUメモリ最終使用量: {final_gpu:.2f}MB")
                print(f"[ストレステスト] GPUメモリ増加量: {gpu_increase:.2f}MB")
                # GPUメモリの許容範囲チェック
                assert gpu_increase < 500, (
                    f"GPUメモリ使用量が{gpu_increase:.2f}MB増加しました(許容値:500MB)"
                )

    elif is_switch_test:
        memory_readings = test_results["memory_usage"]  # キー名を memory_usage に変更
        if len(memory_readings) >= 2:
            final_memory = memory_readings[-1]
            max_memory = max(memory_readings)
            avg_memory = sum(memory_readings) / len(memory_readings)

            print(f"[切り替えテスト] 初期メモリ使用量: {memory_readings[0]:.2f}MB")  # 修正: 初期メモリ表示
            print(f"[切り替えテスト] 最終メモリ使用量: {final_memory:.2f}MB")
            print(f"[切り替えテスト] 最大メモリ使用量: {max_memory:.2f}MB")
            print(f"[切り替えテスト] 平均メモリ使用量: {avg_memory:.2f}MB")

            # 平均値との比較を追加(より安定した指標)
            final_to_avg_ratio = final_memory / avg_memory
            print(f"[切り替えテスト] 最終/平均メモリ比: {final_to_avg_ratio:.2f}")

            # より現実的な判定条件(最終値が平均の2倍未満)
            memory_stable = final_to_avg_ratio < 2

            assert memory_stable, (
                "モデル切り替えテスト中のメモリ使用量に持続的な増加(メモリリークの可能性)が検出されました"
            )

    else:
        # どちらのテスト結果でもない場合(通常は発生しないはず)
        raise ValueError("不明なテスト結果が渡されました")
