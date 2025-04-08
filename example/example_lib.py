import json
import pprint
from pathlib import Path
from typing import Any

from image_annotator_lib import annotate, list_available_annotators
from image_annotator_lib.core.utils import calculate_phash
from PIL import Image

available_models = list_available_annotators()
print("利用可能なモデル:", available_models)
print("-" * 80)

image_dir = Path("tests/resources/img/1_img")
image_paths = sorted(list(image_dir.glob("*.webp")))
images: list[Image.Image] = [Image.open(p) for p in image_paths]
image_path_strs = [str(p) for p in image_paths]

if not images:
    print(f"エラー: テスト画像が見つかりません: {image_dir}")
    exit(1)

print(f"読み込んだ画像数: {len(images)}枚")
print("画像ファイル一覧:")
for i, path in enumerate(image_path_strs):
    print(f"{i + 1}. {path}")
print("-" * 80)

phash_to_path: dict[str, str] = {}
print("画像のpHashを計算中...")
for i, img in enumerate(images):
    phash = calculate_phash(img)
    phash_to_path[phash] = image_path_strs[i]
    print(f"  - {image_path_strs[i]}: {phash}")
print("-" * 80)

available_models = [
    item for item in available_models if "ToriiGate-v0.3" in item]

print(f"\n===== 全利用可能モデル ({', '.join(available_models)}) の一括評価 =====\n")
try:
    results = annotate(images, available_models)

    path_to_phash = {v: k for k, v in phash_to_path.items()}
    sorted_phashes = [path_to_phash[p] for p in image_path_strs if p in path_to_phash]

    for phash in sorted_phashes:
        image_path = phash_to_path.get(phash, f"不明なpHash: {phash}")
        print(f"--- 画像: {image_path} (pHash: {phash}) ---")

        if phash not in results:
            print(f"  エラー: pHash '{phash}' の結果が annotate 関数の戻り値に存在しません。")
            print("-" * 40)
            continue

        model_results_dict = results[phash]

        if not model_results_dict:
            print("  この画像の評価結果はありません。")

        for model_name, model_result in model_results_dict.items():
            print(f"--- モデル: {model_name} ---")

            if error_msg := model_result.get("error"):
                print(f"  エラー: {error_msg}")
                print("-" * 20)
                continue

            tags = model_result.get("tags")
            formatted_output: Any | None = model_result.get("formatted_output")

            is_score = False
            if tags and len(tags) == 1:
                try:
                    score_value = float(tags[0])
                    is_score = True
                    print(f"  スコア: {score_value:.4f}")
                except (ValueError, TypeError):
                    pass

            if not is_score and tags:
                print(f"  検出タグ数: {len(tags)}")
                if tags:
                    print("  検出タグ:")
                    for tag in tags:
                        print(f"    - {tag}")

            if formatted_output:
                if isinstance(formatted_output, dict) and "tags_by_category" in formatted_output:
                    print("  カテゴリ別タグ:")
                    tags_by_cat = formatted_output["tags_by_category"]
                    if isinstance(tags_by_cat, dict):
                        for category, category_tags in tags_by_cat.items():
                            print(f"  【{category}】")
                            if isinstance(category_tags, dict):
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
                                    print(f"    {item_tag:<30} : {item_conf:.4f}")
                            elif isinstance(category_tags, list):
                                for tag in category_tags:
                                    print(f"    - {tag}")
                            print()
                    else:
                        print("  カテゴリ別タグ (予期しない形式):")
                        pprint.pprint(tags_by_cat, indent=4)

                else:
                    print("  追加情報 (formatted_output):")
                    pprint.pprint(formatted_output, indent=4)

            if not tags and not formatted_output:
                print("  結果情報がありません。")

            print("-" * 20)

        print("-" * 40)

    # 結果をJSONファイルに保存
    output_dir = Path("results")
    # 保存先ディレクトリを確実に作成
    print(f"\n結果保存ディレクトリの作成: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # テキストファイルに保存
    output_text_file = output_dir / "annotation_results.txt"
    try:
        print(f"ファイル保存開始: {output_text_file}")
        with open(output_text_file, "w", encoding="utf-8") as f:
            # 単純なテキスト形式で書き出し
            f.write("=== 評価結果 ===\n\n")
            for phash, model_results in results.items():
                f.write(f"画像 pHash: {phash}\n")
                f.write("-" * 40 + "\n")
                for model_name, model_result in model_results.items():
                    f.write(f"モデル: {model_name}\n")
                    # エラーを含む場合
                    if error_msg := model_result.get("error"):
                        f.write(f"  エラー: {error_msg}\n")
                    # タグがある場合
                    if tags := model_result.get("tags"):
                        f.write(f"  タグ数: {len(tags)}\n")
                        f.write("  タグ:\n")
                        for tag in tags:
                            f.write(f"    - {tag}\n")
                    f.write("-" * 20 + "\n")
                f.write("\n")
        print(f"評価結果を {output_text_file} に保存しました。")

        # JSON形式でも保存
        output_json_file = output_dir / "annotation_results.json"
        print(f"JSON形式で保存開始: {output_json_file}")

        try:
            # JSON形式に変換
            json_compatible = {}

            for phash, model_results in results.items():
                json_compatible[phash] = {}

                for model_name, result in model_results.items():
                    json_compatible[phash][model_name] = {}

                    # エラー情報がある場合はそのまま保存
                    if error := result.get("error"):
                        json_compatible[phash][model_name]["error"] = error
                        continue

                    # タグ情報を保存
                    if tags := result.get("tags"):
                        json_compatible[phash][model_name]["tags"] = tags

                    # フォーマット済み出力を保存
                    if formatted := result.get("formatted_output"):
                        json_compatible[phash][model_name]["formatted_output"] = formatted

            # JSONファイルに保存
            with open(output_json_file, "w", encoding="utf-8") as f:
                json.dump(json_compatible, f, ensure_ascii=False, indent=2)

            print(f"評価結果をJSON形式で保存しました: {output_json_file}")

            # JSONファイルの内容確認
            with open(output_json_file, "r", encoding="utf-8") as f:
                json_content = json.load(f)

            print("\n=== JSON形式の内容確認 ===")
            print(f"画像数: {len(json_content)}")
            if json_content:
                first_image = next(iter(json_content.values()))
                print(f"モデル数: {len(first_image)}")
                print(f"保存されたモデル: {', '.join(first_image.keys())}")

        except Exception as e:
            print(f"JSON形式での保存に失敗しました: {e}")

    except Exception as e:
        print(f"評価結果の保存に失敗しました: {e}")
        print(f"保存先パス: {output_text_file.absolute()}")

except Exception as e:
    print(f"評価中に予期せぬエラーが発生しました: {e}")
    import traceback

    traceback.print_exc()

print("\n全モデルの一括評価が完了しました。")
