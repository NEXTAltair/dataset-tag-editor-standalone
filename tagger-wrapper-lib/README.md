# Tagger Wrapper Library

画像タグ付けモデル（BLIP、DeepDanbooru、WD Tagger など）のための統一インターフェースを提供する Python ライブラリです。

## 機能

- 複数のタグ付けモデルを統一的なインターフェースで使用
- 画像からのタグ生成の簡素化
- モデルのロード・アンロードの自動管理
- 各種画像タグ付けモデルのサポート（BLIP、DeepDanbooru、WD Tagger 等）

## インストール

```bash
pip install -e .
```

## 使用例

```python
from PIL import Image
from tagger_wrapper_lib import tag_images, list_available_taggers

# 利用可能なタガーを表示
print(list_available_taggers())

# 画像の読み込み
image = Image.open("example.jpg")

# タグ付け実行
results = tag_images([image], ["blip", "deepdanbooru"])
print(results)
```

## 注意

現在開発中のアルファ版です。Windows 11 環境でのみテスト済みです。
