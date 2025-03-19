# 簡易テスト用コード
# 戻り値の確認のため
import logging

from PIL import Image

from tagger_wrapper_lib import evaluate
from tagger_wrapper_lib.core.utils import setup_logger

setup_logger("tagger_wrapper_lib")
logger = logging.getLogger(__name__)


# 利用可能なスコアラーを表示
print("Available taggers:", list_available_taggers())

# アノテーションする画像を用意
image = Image.open("tests/resources/img/1_img/file01.webp")
images = [
    Image.open("tests/resources/img/1_img/file01.webp"),
    Image.open("tests/resources/img/1_img/file02.webp"),
]

# # 各モデルでアノテーションを実行 (単一画像x単一モデル)
for model_name in list_available_taggers():
    print(f"\\nEvaluating with {model_name} (single image):")
    try:
        result = evaluate([image], [model_name])
        print(result)
    except Exception as e:
        print(f"Error evaluating with {model_name}: {e}")

# 各モデルでアノテーションを実行 (複数画像)
# for model_name in list_available_taggers():
#     print(f"\\nEvaluating with {model_name} (multiple images):")
#     try:
#         results = evaluate(images, [model_name])
#         print(results)
#     except Exception as e:
#         print(f"Error evaluating with {model_name}: {e}")
