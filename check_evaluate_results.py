# ライブラリ戻り値チェック用

import json
import pprint
from datetime import datetime
from pathlib import Path

from PIL import Image
from scorer_wrapper_lib import evaluate as scorer
from scorer_wrapper_lib import list_available_models as scorer_list_available_models
from tagger_wrapper_lib import evaluate as tagger
from tagger_wrapper_lib import list_available_models as tagger_list_available_models

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

# 画像ファイルの読み込み
test_image_dir = Path("tests/resources/img/1_img")
image_files = list(test_image_dir.glob("*.webp"))
if not image_files:
    print(f"エラー: テスト画像が見つかりません: {test_image_dir}")
    exit(1)

# 画像パスを保存しておく
image_paths = [str(img_file) for img_file in image_files]

image_list = [Image.open(img_file) for img_file in image_files]
print(f"読み込んだ画像数: {len(image_list)}枚")

# モデルリストの取得
scorer_model_list = scorer_list_available_models()
tagger_model_list = tagger_list_available_models()
print(f"利用可能なscorerモデル数: {len(scorer_model_list)}")
for model in scorer_model_list:
    print(f"  - {model}")
print(f"利用可能なtaggerモデル数: {len(tagger_model_list)}")
for model in tagger_model_list:
    print(f"  - {model}")

# タイムスタンプの生成
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

try:
    # Scorerモデルの評価実行
    print("Scorerモデルの評価を実行中...")
    scorer_result = scorer(image_list, scorer_model_list)

    # デバッグ: scorer_resultの構造を確認
    print("===== Scorer結果の構造 =====")
    print(f"scorer_result type: {type(scorer_result)}")
    if isinstance(scorer_result, dict):
        print("scorer_result keys:")
        model_names = list(scorer_result.keys())
        for model_name in model_names[:3]:  # 最初の3つだけ表示
            print(f"  - {model_name}")
        if len(model_names) > 3:
            print(f"  - ... 他 {len(model_names) - 3} 個のモデル")

        # 最初のキーの値を表示
        if scorer_result:
            first_key = list(scorer_result.keys())[0]
            print(f"First key: {first_key}")
            print(f"Value type: {type(scorer_result[first_key])}")
            print("First value sample:")
            pprint.pprint(scorer_result[first_key][0])
            print(f"画像ファイルパス: {image_paths}")

    # Taggerモデルの評価実行
    print("Taggerモデルの評価を実行中...")
    tagger_result = tagger(image_list, tagger_model_list)

    # デバッグ: tagger_resultの構造を確認
    print("===== Tagger結果の構造 =====")
    print(f"tagger_result type: {type(tagger_result)}")
    if isinstance(tagger_result, dict):
        print("tagger_result keys:")
        model_names = list(tagger_result.keys())
        for model_name in model_names[:3]:  # 最初の3つだけ表示
            print(f"  - {model_name}")
        if len(model_names) > 3:
            print(f"  - ... 他 {len(model_names) - 3} 個のモデル")

        # 最初のキーの値を表示
        if tagger_result:
            first_key = list(tagger_result.keys())[0]
            print(f"First key: {first_key}")
            print(f"Value type: {type(tagger_result[first_key])}")
            print("First value sample:")
            pprint.pprint(tagger_result[first_key][0])
            print(f"画像ファイルパス: {image_paths}")

    # 結果をファイルに保存
    result_file = results_dir / f"evaluation_results_{timestamp}.txt"
    with open(result_file, "w", encoding="utf-8") as f:
        f.write("評価結果\n")
        f.write(f"日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"画像数: {len(image_list)}枚\n")
        f.write("-" * 80 + "\n\n")

        # 画像ファイル一覧を追加
        f.write("画像ファイル一覧:\n")
        for i, path in enumerate(image_paths):
            f.write(f"{i + 1}. {path}\n")
        f.write("-" * 80 + "\n\n")

        # Scorer結果
        f.write("===== Scorer評価結果 =====\n")
        for model_idx, model_name in enumerate(scorer_result.keys()):
            f.write(f"--- モデル {model_idx + 1}: {model_name} ---\n")
            scores = scorer_result[model_name]

            # scoresがリスト形式の場合（モデルで評価された各画像の結果リスト）
            if isinstance(scores, list):
                for i, image_data in enumerate(scores):
                    if isinstance(image_data, dict):
                        # 画像情報は含まれないので、インデックスから画像パスを取得
                        img_path = image_paths[i]

                        score_value = image_data["score_tag"]
                        score_str = (
                            f"{score_value:.4f}"
                            if isinstance(score_value, (int, float))
                            else str(score_value)
                        )
                        f.write(f"画像ファイル: {img_path}\n")
                        f.write(f"スコア: {score_str}\n")
                    else:
                        f.write(f"画像データ: {image_data}\n")
                    f.write("-" * 40 + "\n")
            # scoresが辞書形式の場合（モデル内画像情報をキーとする辞書）
            elif isinstance(scores, dict):
                # 重複したキーを防ぐために処理済みのキーを記録
                processed_keys = set()
                for key, score_value in scores.items():
                    if key in processed_keys:
                        continue
                    processed_keys.add(key)
                    score_str = (
                        f"{score_value:.4f}" if isinstance(score_value, (int, float)) else str(score_value)
                    )
                    f.write(f"画像キー: {key}\n")
                    f.write(f"スコア: {score_str}\n")
                    f.write("-" * 40 + "\n")
            else:
                f.write(f"未対応のスコアデータ形式: {type(scores)}\n")
                f.write(f"{scores}\n")
                f.write("-" * 40 + "\n")
            f.write("\n")

        # Tagger結果
        f.write("\n===== Tagger評価結果 =====\n")
        for model_idx, model_name in enumerate(tagger_result.keys()):
            f.write(f"--- モデル {model_idx + 1}: {model_name} ---\n")
            tags_by_image = tagger_result[model_name]

            # tags_by_imageがリスト形式の場合（モデルでタグ付けされた各画像の結果リスト）
            if isinstance(tags_by_image, list):
                for i, image_data in enumerate(tags_by_image):
                    if isinstance(image_data, dict):
                        # 画像情報は含まれないので、インデックスから画像パスを取得
                        img_path = image_paths[i]

                        # タグ情報を取得
                        tags = image_data["annotation"]

                        f.write(f"画像ファイル: {img_path}\n")
                        f.write(f"検出タグ数: {len(tags)}\n")

                        if tags:
                            f.write("検出タグ:\n")
                            for tag in tags:
                                f.write(f"  - {tag}\n")

                        # model_outputの内容を詳細に表示
                        model_output = image_data["model_output"]
                    else:
                        f.write(f"画像データ: {image_data}\n")
                    f.write("-" * 40 + "\n")
            # tags_by_imageが辞書形式の場合（モデル内画像情報をキーとする辞書）
            elif isinstance(tags_by_image, dict):
                # 重複したキーを防ぐために処理済みのキーを記録
                processed_keys = set()
                for key, tags in tags_by_image.items():
                    if key in processed_keys:
                        continue
                    processed_keys.add(key)
                    f.write(f"画像キー: {key}\n")

                    if isinstance(tags, list):
                        f.write(f"検出タグ数: {len(tags)}\n")
                        if tags:
                            f.write("検出タグ:\n")
                            for tag in tags:
                                f.write(f"  - {tag}\n")
                    elif isinstance(tags, dict):
                        # カテゴリ別にタグが格納されている場合
                        total_tags = sum(
                            len(category_tags)
                            for category_tags in tags.values()
                            if isinstance(category_tags, (list, dict))
                        )
                        f.write(f"検出タグ数: {total_tags}\n")

                        for category, category_tags in tags.items():
                            f.write(f"【{category}】\n")

                            if isinstance(category_tags, dict):
                                # confidence値でソート
                                sorted_items = []
                                for item_tag, item_data in category_tags.items():
                                    item_conf: float = 0.0
                                    if isinstance(item_data, dict) and "confidence" in item_data:
                                        item_conf = float(item_data["confidence"])
                                    elif isinstance(item_data, (int, float)):
                                        item_conf = float(item_data)
                                    sorted_items.append((item_tag, item_conf))

                                sorted_items.sort(key=lambda x: x[1], reverse=True)

                                for item_tag, item_conf in sorted_items:
                                    if item_conf >= 0.5:  # 閾値以上のタグのみ表示
                                        f.write(f"  {item_tag:<30} : {item_conf:.4f}\n")
                            elif isinstance(category_tags, list):
                                for tag in category_tags:
                                    f.write(f"  - {tag}\n")

                            f.write("\n")
                    else:
                        # その他の型の場合
                        f.write(f"タグ情報: {tags}\n")

                    f.write("-" * 40 + "\n")
            else:
                f.write(f"未対応のタグデータ形式: {type(tags_by_image)}\n")
                f.write(f"{tags_by_image}\n")
                f.write("-" * 40 + "\n")
            f.write("\n")

    # JSONファイルとしても保存
    json_file = results_dir / f"evaluation_results_{timestamp}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            {"scorer_result": scorer_result, "tagger_result": tagger_result}, f, indent=2, default=str
        )

    print(f"評価結果を {result_file} に保存しました")
    print(f"JSONデータを {json_file} に保存しました")
    print("評価処理が完了しました")

except Exception as e:
    print(f"エラーが発生しました: {e}")
    import traceback

    traceback.print_exc()
    exit(1)
