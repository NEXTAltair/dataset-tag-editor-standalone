# feature/tagger-family-enhancement

このブランチは フォーク元のブランチに対して、以下の変更を行う
元の実装コードを変更､追加して｡Wrapperからの呼び出しに対応した処理を実装する


# やったこと

 - a2bc37d: ラッパーからの呼び出しを判別して処理を分岐させるメソッドの追加

 - 必要とは言えない変更 fd5b81f: BLIP2モデルの説明を返すメソッドの追加 #TODO: 説明書きの他言語対応や､もっと読みやすくするのは今度

# やるべきこと

 - gradio: 5.13.1 対応版のバグ修正

 - ラップするインテロゲーターの修正前殿整合性を確認するためのテストコードの追加

 - Wrapperで呼び出した時のオプションの引数の扱いを考える

# やりたいこと

古いモデルは消してもいいかもしれない


# toshiaki1729/dataset-tag-editor-standalone

## フォーク元の処理の流れ

1. ```scripts.launch``` で ```interface.main()``` を呼び出す

2. ```interface.main()``` で ```dte_instance.load_interrogators()``` を呼び出す

3. ```dte_instance.load_interrogators()``` で userscript ディレクトリ内に実装されたものを含むインタロゲータcustom_taggers(list) に 使用できるタガーとしてguiに表示するものをリストアップ

    3.1 ```scripts.dataset_tag_editor.taggers_builtin``` にある ```scripts.tagger.Tagger``` を継承したクラスがGUIで使用するWrapper

    3.2 このタイミングで各インタロゲータの```__init__```が呼び出されるが､ processor､model の値は None である

4. guiで lordがクリックされたときに ```scripts.dataset_tag_editor.dte_logic.DatasetTagEditor.load_dataset``` に interrogator_names(list[str]) を渡してinterrogatorを呼び出す

    4.1 interrogator に メソッド ```predict_pipe``` がある場合は ```predict_pipe``` を呼び出す｡ない場合 ```predict``` を呼び出す

5. ```Tagger``` を継承したクラスは scripts.dataset_tag_editor.interrogators に実装したから ```predict``` は ```apply``` にImageを  ```predict_pipe```は ```apply_multi``` にlist[Imeg] とバッチサイズを渡す､返り値は各モデルでタグ付け結果を返す

# モデルにわたす引数オプション

thresholds (しきい値) 値が小さいとタグが多くなるが､精度が下がる､値が大きいとタグが少なくなるが､精度が上がる `scripts.dataset_tag_editor.interrigator_names` に一部はデフォルト値を設定している
 

# 調べた

predict_pipe を全ての Interrogator に実装せずに predict だけを実装している理由
    モデルそのものがバッリ処理に対応していないため｡
    predict_pipeをそれでも実装するときは､predict_pipe内でpredictを呼び出す｡

# 調べる

実装の構造の層が思っていたよりも1層多い｡この実装の利点について 4.1 の箇所

# 他のタグ付けツール
https://github.com/picobyte/stable-diffusion-webui-wd14-tagger 2年前に更新停止
 - プルリク 8件

https://github.com/pharmapsychotic/clip-interrogator-ext 2年前に更新停止
 - プルリク 4件

 https://github.com/KichangKim/DeepDanbooru 更新中
 - プルリク 3件

https://huggingface.co/spaces/SmilingWolf/wd-tagger/tree/main 更新中

https://github.com/discus0434/aesthetic-predictor-v2-5 更新中

https://github.com/Nenotriple/img-txt_viewer 更新中

https://github.com/mikeknapp/candy-machine 更新中


# 他のタグ付けモデル

https://github.com/THUDM/CogVLM マルチモーダルのタグ付けモデル

https://huggingface.co/laion/CLIP-ViT-g-14-laion2B-s34B-b88K

https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k
    https://huggingface.co/silveroxides/CLIP-ViT-bigG-14-laion2B-39B-b160k-fp16 VRAM16GB以上必要動かせない