# 簡易テスト用コード
# 戻り値の確認のため

from PIL import Image

from scorer_wrapper_lib import evaluate, list_available_scorers

# 利用可能なスコアラーを表示
print("Available scorers:", list_available_scorers())

# 評価する画像を用意
image = Image.open("tests/resources/img/1_img/file01.webp")
images = [
    Image.open("tests/resources/img/1_img/file01.webp"),
    Image.open("tests/resources/img/1_img/file02.webp"),
]

# # 各モデルで評価を実行 (単一画像x単一モデル)
for model_name in list_available_scorers():
    print(f"\\nEvaluating with {model_name} (single image):")
    try:
        result = evaluate([image], [model_name])
        print(result)
    except Exception as e:
        print(f"Error evaluating with {model_name}: {e}")

# 各モデルで評価を実行 (複数画像)
# for model_name in list_available_scorers():
#     print(f"\\nEvaluating with {model_name} (multiple images):")
#     try:
#         results = evaluate(images, [model_name])
#         print(results)
#     except Exception as e:
#         print(f"Error evaluating with {model_name}: {e}")
