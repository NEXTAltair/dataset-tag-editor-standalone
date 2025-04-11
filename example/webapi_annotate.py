from image_annotator_lib.model_class.annotator_webapi import AnthropicApiAnnotator
from PIL import Image

# 画像を読み込み
image = Image.open("tests/resources/img/1_img/file01.webp")

# アノテーターを初期化
with AnthropicApiAnnotator("claude-3.5sonnet-20240620") as annotator:
    # 単一画像に対する処理
    results = annotator.predict([image], ["test_hash"])
    print(results)
    # 結果を表示
    for result in results:
        print(f"タグ: {result['tags']}")
        print(f"エラー: {result['error']}")
