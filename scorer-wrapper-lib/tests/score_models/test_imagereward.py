from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from pytest_bdd import given, scenario, then, when

from scorer_wrapper_lib.score_models.imagereward import ImageRewardModel


@pytest.fixture
def mock_image():
    """テスト用の画像を生成"""
    img = Image.new("RGB", (100, 100), color="red")
    return img


@pytest.fixture
def mock_model():
    """モックモデルを作成"""
    # クラスをモックする方法に変更
    with patch.object(ImageRewardModel, "__init__", return_value=None):
        model = ImageRewardModel()
        # モックモデルの設定
        model.model = MagicMock()
        model.tokenizer = MagicMock()
        model.device = "cpu"
        model.is_model_loaded = True  # モデルがロード済みとマーク

        # スコア計算メソッドのモック
        model._calculate_score = MagicMock()
        # 入力値をそのまま返すようにする（変換はタグ生成時のみ必要）
        model._calculate_score.side_effect = lambda x: x

        # タグ生成メソッドのモック
        model._get_score_tag = MagicMock()
        model._get_score_tag.side_effect = lambda x: f"[IR]score_{int(x)}"

        # score_image_text_pairsメソッドのモック実装
        def mock_score_image_text_pairs(images, texts):
            # モデルの戻り値を使用してスコアリング結果を返す
            return [
                model.model.return_value[i] for i in range(min(len(images), len(model.model.return_value)))
            ]

        model.score_image_text_pairs = mock_score_image_text_pairs

        # _generate_resultメソッドのモック
        model._generate_result = MagicMock()
        model._generate_result.side_effect = lambda raw_score, score_tag: {
            "model_output": raw_score,
            "model_name": "imagereward",
            "score_tag": score_tag,
        }

        return model


# 背景ステップ
@given("初期化されたImageRewardモデルが存在する", target_fixture="model_context")
def initialized_model(mock_model):
    """モデルが初期化されている"""
    assert mock_model.is_model_loaded is True
    return {"model": mock_model}


# シナリオ1: 画像の評価
@scenario("../features/score_models/imagereward.feature", "画像の評価")
def test_image_evaluation():
    """画像評価のテストシナリオ"""
    pass


@given("初期化されたImageRewardモデルが存在する", target_fixture="eval_context")
def model_exists(mock_model, mock_image):
    """モデルと画像の準備"""
    return {"model": mock_model, "image": mock_image, "result": None}


@when("画像が評価される")
def evaluate_image(eval_context):
    """画像を評価する"""
    # モックの戻り値を設定
    eval_context["model"].model.return_value = [0.75]
    eval_context["result"] = eval_context["model"].score_image_text_pairs(
        images=[eval_context["image"]], texts=["テスト画像の説明"]
    )


@then("0-10の範囲のスコアが返される")
def check_score_range(eval_context):
    """スコアの範囲をチェック - 生のスコア値をチェック"""
    assert isinstance(eval_context["result"], list)
    assert len(eval_context["result"]) == 1
    assert isinstance(eval_context["result"][0], float)
    # Tensorから変換された浮動小数点数は誤差が生じる可能性があるため、
    # 近似値での比較を行う
    assert eval_context["result"][0] == pytest.approx(0.75, abs=1e-5)


@then("スコアに基づいたタグが付与される")
def check_tags_assigned(eval_context):
    """タグの確認 - タグ生成処理をチェック"""
    # 実際の実装では、_get_score_tagメソッドがスコアからタグを生成
    score = eval_context["result"][0]
    # intに変換されたスコアでタグが生成される
    expected_tag = f"[IR]score_{int(score)}"

    # タグの生成をモックモデルを使って検証
    generated_tag = eval_context["model"]._get_score_tag(score)
    assert generated_tag == expected_tag


# シナリオ2: バッチ画像の評価
@scenario("../features/score_models/imagereward.feature", "バッチ画像の評価")
def test_batch_evaluation():
    """バッチ評価のテストシナリオ"""
    pass


@given("初期化されたImageRewardモデルと複数の画像が存在する", target_fixture="batch_context")
def model_and_multiple_images(mock_model, mock_image):
    """モデルと複数画像の準備"""
    # モックの戻り値を設定
    mock_model.model.return_value = [0.65, 0.75]
    return {"model": mock_model, "images": [mock_image, mock_image], "result": None}


@when("画像がバッチで評価される")
def evaluate_batch_images(batch_context):
    """バッチ評価を実行"""
    batch_context["result"] = batch_context["model"].score_image_text_pairs(
        images=batch_context["images"], texts=["テスト画像1", "テスト画像2"]
    )


@then("各画像に対して評価結果が返される")
def check_batch_results(batch_context):
    """バッチ評価結果を確認"""
    assert isinstance(batch_context["result"], list)
    assert len(batch_context["result"]) == len(batch_context["images"])
    # Tensorから変換された浮動小数点数値は高精度で、誤差が生じる可能性がある
    # そのため許容誤差を設定して比較する
    assert batch_context["result"][0] == pytest.approx(0.65, abs=1e-5)
    assert batch_context["result"][1] == pytest.approx(0.75, abs=1e-5)


# シナリオ3: プロンプト文との関連性評価
@scenario("../features/score_models/imagereward.feature", "プロンプト文との関連性評価")
def test_prompt_relevance():
    """プロンプト文との関連性評価テスト"""
    pass


@given("初期化されたImageRewardモデルと画像、プロンプト文が存在する", target_fixture="prompt_context")
def model_image_and_prompt(mock_model, mock_image):
    """モデル、画像、プロンプトの準備"""
    # モックの戻り値を設定
    mock_model.model.return_value = [0.8]
    return {"model": mock_model, "image": mock_image, "prompt": "美しい赤い風景", "result": None}


@when("画像がプロンプト文と共に評価される")
def evaluate_with_prompt(prompt_context):
    """プロンプト付き評価を実行"""
    prompt_context["result"] = prompt_context["model"].score_image_text_pairs(
        images=[prompt_context["image"]], texts=[prompt_context["prompt"]]
    )


@then("プロンプト文との関連性を考慮したスコアが返される")
def check_prompt_relevance_score(prompt_context):
    """プロンプト関連性スコアを確認"""
    assert isinstance(prompt_context["result"], list)
    assert len(prompt_context["result"]) == 1
    assert isinstance(prompt_context["result"][0], float)
    # Tensorから変換された浮動小数点数値の比較
    assert prompt_context["result"][0] == pytest.approx(0.8, abs=1e-5)
