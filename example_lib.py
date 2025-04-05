# 簡易テスト用コード
# 戻り値の確認のため

from PIL import Image

from image_annotator_lib import annotate, list_available_annotators

# 利用可能なスコアラーを表示
print("Available models:", list_available_annotators())

# アノテーションする画像を用意
image = [Image.open("tests/resources/img/1_img/file01.webp")]
images = [
    Image.open("tests/resources/img/1_img/file01.webp"),
    Image.open("tests/resources/img/1_img/file02.webp"),
    Image.open("tests/resources/img/1_img/file03.webp"),
    Image.open("tests/resources/img/1_img/file04.webp"),
]

# # 各モデルでアノテーションを実行 (単一画像x単一モデル)
# for model_name in list_available_annotators():
#     print(f"\\nEvaluating with {model_name} (single image):")
#     try:
#         result = annotate(images, [model_name])
#         print(result)
#     except Exception as e:
#         print(f"Error evaluating with {model_name}: {e}")

# 各モデルでアノテーションを実行 (複数画像)
for model_name in list_available_annotators():
    print(f"\\nEvaluating with {model_name} (multiple images):")
    try:
        results = annotate(images, [model_name])
        print(results)
    except Exception as e:
        print(f"Error evaluating with {model_name}: {e}")
