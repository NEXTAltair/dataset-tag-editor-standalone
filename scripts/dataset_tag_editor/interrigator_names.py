
BLIP2_CAPTIONING_NAMES = {
    "blip2-opt-2.7b": (
        "- 2.7B パラメータのOPTをベースとした大規模マルチモーダルモデル\n"
        "- 凍結された画像エンコーダと大規模言語モデルを組み合わせたアーキテクチャ\n"
        "- Q-Former(Querying Transformer)による効率的な画像-言語表現の橋渡し\n"
        "- 画像キャプション生成、VQA、画像についての対話的な応答が可能"
    ),
    "blip2-opt-2.7b-coco": (
        "BLIP2-OPT-2.7B-COCO\n"
        "- COCOデータセットでファインチューニング済み\n"
        "- 3つのコンポーネント構成:\n"
        "  ・CLIPライク画像エンコーダ\n"
        "  ・Querying Transformer (Q-Former)\n"
        "  ・OPT-2.7b言語モデル\n"
        "- 画像キャプション生成、VQA、対話的応答に対応\n"
        "- float16対応で効率的なメモリ使用を実現"
    ),
    "blip2-opt-6.7b": (
        "BLIP2-OPT-6.7B\n"
        "- 67億パラメータのOPTベース大規模言語モデル\n"
        "- 3つのコアコンポーネント構成：\n"
        "  ・CLIPベース画像エンコーダ\n"
        "  ・Querying Transformer (Q-Former)\n"
        "  ・OPT-6.7B言語モデル\n"
        "- 主な機能：\n"
        "  ・高精度な画像キャプション生成\n"
        "  ・視覚的質問応答(VQA)\n"
        "  ・画像ベースの対話システム\n"
        "- 技術的特徴：\n"
        "  ・凍結された事前学習済みエンコーダと言語モデル\n"
        "  ・Q-Formerによる効率的な画像-言語変換\n"
        "  ・総パラメータ数7.75B\n"
        "- 高度な画像理解と自然な文章生成を実現"
    ),
    "blip2-opt-6.7b-coco": (
        "BLIP2-OPT-6.7B-COCO\n"
        "- 性能諸元:\n"
        "  ・総パラメータ数: 7.75B\n"
        "  ・ライセンス: MIT\n"
        "  ・開発: Salesforce\n"
        "- アーキテクチャ構成:\n"
        "  ・CLIP形式画像エンコーダ\n"
        "  ・Querying Transformer (Q-Former)\n"
        "  ・OPT-6.7b言語モデル\n"
        "- COCOデータセットで最適化:\n"
        "  ・高精度な物体認識\n"
        "  ・自然な説明文生成\n"
        "  ・文脈を考慮した応答\n"
        "- 処理性能(RTX A6000):\n"
        "  ・FP16: 0.39秒/画像\n"
        "  ・INT8: 2.01秒/画像\n"
        "  ・INT4: 0.74秒/画像"
    ),
    "blip2-flan-t5-xl": (
        "BLIP2-FLAN-T5-XL\n"
        "- 基本仕様:\n"
        "  ・パラメータ数: 3.94B\n"
        "  ・ライセンス: MIT\n"
        "  ・開発元: Salesforce\n"
        "- アーキテクチャ:\n"
        "  ・CLIP形式画像エンコーダ\n"
        "  ・Querying Transformer (Q-Former)\n"
        "  ・Flan-T5-XL言語モデル\n"
        "- 最適化対応:\n"
        "  ・float32（フル精度）\n"
        "  ・float16（半精度）\n"
        "  ・int8（8ビット量子化）\n"
        "- 主要機能:\n"
        "  ・画像キャプション生成\n"
        "  ・視覚的質問応答(VQA)\n"
        "  ・画像ベースの対話処理"
    ),
    "blip2-flan-t5-xl-coco": (
        "BLIP2-FLAN-T5-XL-COCO\n"
        "- 基本仕様:\n"
        "  ・パラメータ数: 3.94B\n"
        "  ・ライセンス: MIT\n"
        "  ・開発元: Salesforce\n"
        "- アーキテクチャ:\n"
        "  ・CLIP形式画像エンコーダ\n"
        "  ・Querying Transformer (Q-Former)\n"
        "  ・Flan-T5-XL言語モデル\n"
        "  ・COCOデータセット最適化済み\n"
        "- 精度設定:\n"
        "  ・画像エンコーダ: FP16\n"
        "  ・Q-Former: FP32\n"
        "  ・言語モデル: BF16\n"
        "- 主要機能:\n"
        "  ・高精度画像キャプション\n"
        "  ・視覚的質問応答(VQA)\n"
        "  ・コンテキスト考慮型対話"
    ),
    "blip2-flan-t5-xxl": (
        "BLIP2-FLAN-T5-XXL\n"
        "- 基本仕様:\n"
        "  ・パラメータ数: 12.2B\n"
        "  ・ライセンス: MIT\n"
        "  ・開発元: Salesforce\n"
        "- アーキテクチャ:\n"
        "  ・CLIPスタイル画像エンコーダ\n"
        "  ・Querying Transformer (Q-Former)\n"
        "  ・Flan-T5-XXL言語モデル\n"
        "- 最適化対応:\n"
        "  ・float32（フル精度）\n"
        "  ・float16（半精度）\n"
        "  ・int8（8ビット量子化）\n"
        "- 技術的特徴:\n"
        "  ・凍結された事前学習済みコンポーネント\n"
        "  ・Q-Formerによる効率的な特徴変換\n"
        "  ・BERTベースの高度な言語理解\n"
        "  ・自動デバイスマッピングサポート"
    )
}



# {tagger name : default tagger threshold}
# v1: idk if it's okay  v2, v3: P=R thresholds on each repo https://huggingface.co/SmilingWolf
WD_TAGGERS = {
    "wd-v1-4-vit-tagger" : 0.35,
    "wd-v1-4-convnext-tagger" : 0.35,
    "wd-v1-4-vit-tagger-v2" : 0.3537,
    "wd-v1-4-convnext-tagger-v2" : 0.3685,
    "wd-v1-4-convnextv2-tagger-v2" : 0.371,
    "wd-v1-4-moat-tagger-v2" : 0.3771
}
WD_TAGGERS_TIMM = {
    "wd-v1-4-swinv2-tagger-v2" : 0.3771,
    "wd-vit-tagger-v3" : 0.2614,
    "wd-convnext-tagger-v3" : 0.2682,
    "wd-swinv2-tagger-v3" : 0.2653,
    "wd-vit-large-tagger-v3" : 0.2606,
    "wd-eva02-large-tagger-v3" : 0.5296
}