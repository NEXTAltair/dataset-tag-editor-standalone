import re

from PIL import Image
from pytest_bdd import given, parsers, scenarios, then, when

from scorer_wrapper_lib import evaluate

scenarios("features/image_scoring.feature")


# ---------- Given ステップ ----------
"""
single_image と images は conftest.py で定義
"""


@given("サポートされていない画像フォーマットまたは破損した画像が提供される", target_fixture="single_image")
def given_invalid_image():
    from io import BytesIO

    invalid_data = BytesIO(b"invalid single_image data")
    try:
        return Image.open(invalid_data)
    except Exception:
        return None


@given(parsers.parse("{num_images:d}枚の有効な画像が用意されている"), target_fixture="images")
def given_num_valid_images(num_images):
    from PIL import Image

    valid_image = Image.new("RGB", (10, 10))
    # 指定された枚数の画像リストを返す
    return [valid_image for _ in range(num_images)]


# ---------- When ステップ ----------


@when(
    parsers.parse("ユーザーが画像評価機能を起動し、{model_name} を指定してその画像を入力する"),
    target_fixture="result",
)
def when_evaluate_single_image(model_name, single_image):
    return evaluate(single_image, [model_name])


@when(
    parsers.parse("ユーザーが画像評価機能を起動し、{model_name} を指定してそれらの画像を入力する"),
    target_fixture="result",
)
def when_evaluate_multiple_images_single_model(model_name, images):
    return evaluate(images, [model_name])


@when(
    parsers.parse("ユーザーが画像評価機能を起動し、以下のモデルリストを指定してその画像を入力する:"),
    target_fixture="result",
)
def when_evaluate_single_image_multiple_models(single_image, datatable):
    models = datatable[1]
    return evaluate(single_image, models)


@when(
    parsers.parse("ユーザーが画像評価機能を起動し、以下のモデルリストを指定してそれら画像を入力する:"),
    target_fixture="result",
)
def when_evaluate_multiple_image_multiple_models(images, datatable):
    models = datatable[1]
    return evaluate(images, models)


@when("ユーザーが画像評価機能を起動する", target_fixture="result")
def when_evaluate_invalid_image(single_image):
    try:
        return evaluate(single_image, ["dummy"])
    except Exception as e:
        return e


# ---------- Then ステップ ----------


def verify_score_tag(actual_tag: str, expected_tag: str) -> bool:
    """スコアタグを検証します。

    以下の条件を確認します：
    1. プレフィックスが一致する（[CAFE], [IAP], [WAIFU]など）
    2. "score_" の後に0-10の整数が続く
    3. プレフィックスがない場合は、aesthetic関連の文字列（aesthetic, very aesthetic）と一致

    Args:
        actual_tag (str): 実際のスコアタグ
        expected_tag (str): 期待されるスコアタグ

    Returns:
        bool: 検証結果
    """
    # プレフィックスがある場合（[CAFE], [IAP], [WAIFU]など）
    if expected_tag.startswith("["):
        prefix = expected_tag.split("score_")[0]
        if not actual_tag.startswith(prefix):
            return False
        # score_の後に0-10の整数が続くことを確認
        match = re.search(r"score_(\d+)$", actual_tag)
        if not match:
            return False
        score = int(match.group(1))
        return 0 <= score <= 10
    # aesthetic関連の文字列の場合
    else:
        return actual_tag in [
            "aesthetic",
            "very aesthetic",
            "neutral",
            "not aesthetic",
            "displeasing",
            "very displeasing",
        ]


@then(parsers.parse("システムは以下の結果を返す: {model_name} {model_output} {score_tag}"))
def then_verify_result(result, model_name, score_tag):
    # 単一画像の場合は、返り値(result)はリストの先頭を検証
    if isinstance(result, list):
        result = result[0]
    assert result["model_name"] == model_name
    if score_tag:
        assert verify_score_tag(result["score_tag"], score_tag)


@then(parsers.parse("システムは {num_images:d} 件の評価結果を返す"))
def then_verify_number_of_results(result, num_images):
    # result が複数画像の場合 (リストとして返っていることを想定)
    assert len(result) == num_images


@then(parsers.parse("{index:d}番目の画像の評価結果は {model_name} {model_output} {score_tag} である"))
def then_verify_indexed_result(result, index, model_name, model_output, score_tag):
    idx = index - 1
    assert result[idx]["model_name"] == model_name
    if score_tag:
        assert verify_score_tag(result[idx]["score_tag"], score_tag)


@then("システムは2件のモデル評価結果を返す")
def then_verify_multiple_models_result(result):
    assert len(result) == 2


@then(parsers.parse("{index:d}番目のモデルの評価結果は {model_name} {model_output} {score_tag} である"))
def then_verify_multiple_images_multiple_models_result(result, index, model_name, model_output, score_tag):
    # result の該当モデルは単一の辞書、またはリストの場合がある
    model_res = result.get(model_name)
    if isinstance(model_res, list):
        model_res = model_res[index - 1]  # 1-indexed
    # 1件の場合は単に検証
    assert model_res["model_name"] == model_name
    if score_tag:
        assert verify_score_tag(model_res["score_tag"], score_tag)


@then(parsers.parse("システムは{num_images:d}枚の画像に対して、各モデルの評価結果を返す"))
def then_verify_images_multiple_models(result, num_images):
    # 各モデルの評価結果（リスト）の件数を検証する
    for model_results in result.values():
        assert len(model_results) == num_images


@then(
    parsers.parse(
        "{model_index:d}番目のモデルの{image_index:d}番目の画像の評価結果は {model_name} {output_type} {score_tag} である"
    )
)
def then_verify_each_model_image(result, model_index, image_index, model_name, output_type, score_tag):
    # result はモデル名をキーに持つ辞書
    # シナリオでは、複数モデルの順番はデータテーブルの行順に対応していると想定し、
    # キーの順序が期待と一致している場合、または直接 model_name で検証する。
    model_results = result.get(model_name)
    assert model_results is not None, f"Model '{model_name}' not found in result"
    assessment = model_results[image_index - 1]  # 1-indexed
    assert assessment["model_name"] == model_name
    if score_tag:
        assert verify_score_tag(assessment["score_tag"], score_tag)


@then("システムは適切なエラーメッセージを表示するか、例外を発生させる")
def then_verify_error(result):
    assert isinstance(result, Exception)
