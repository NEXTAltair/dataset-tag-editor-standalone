# Mypy エラー修正計画

## 課題

以下の Mypy エラーが報告されていました。

1.  `Item "None" of "dict[str, Any] | None" has no attribute "get"Mypyunion-attr` (`config.py` の `get` メソッド内)
2.  `Argument 3 to "load_transformers_components" of "ModelLoad" has incompatible type "str | float"; expected "str"` (`base.py` で `self.device` を渡す箇所)
3.  (当初) `Argument 2 to "load_transformers_components" of "ModelLoad" has incompatible type "Any | None"; expected "str"` (`base.py` で `self.model_path` を渡す箇所 - これはユーザーが `get_model_path` 作成で解決済み)

## 原因分析

*   エラー 1 は、`config.py` の `get` メソッド内で `model_config` 変数が `None` になる可能性があり、その `None` に対して `.get(key, default)` を呼び出そうとしていたためです。これは、`self._config_data.get(model_name)` の呼び出し方 (またはその後の編集) に起因していました。
*   エラー 2 は、`config.py` の `get` メソッドの戻り値型ヒント (例: `str | float | None`) により、`base.py` で `self.device` が `str | float` と推論され、`str` を期待する関数に渡されたためです。

## 解決策 (承認済み)

`src/image_annotator_lib/core/config.py` の `ModelConfigRegistry.get` メソッドを以下のように修正します。

```python
# src/image_annotator_lib/core/config.py

# (必要であれば) from typing import Any を修正 (例: Union, Optional)
# (必要であれば) T = TypeVar("T") を削除

class ModelConfigRegistry:
    # ... (他のメソッド) ...

    # get メソッドの修正
    # 型ヒントは実際の戻り値に合わせて str | float | None とする (int が不要な場合)
    # default 引数の型も Any から None に変更 (T を使わないため)
    def get(self, model_name: str, key: str, default: None = None) -> str | float | None:
        """指定されたモデルとキーに対応する設定値を取得します。
        # ... (docstring は現状維持 or 修正) ...
        """
        # model_name が存在しない場合は KeyError が発生する (意図通り)
        # タイポを修正: "model_name" -> model_name
        try:
            model_config = self._config_data[model_name]
        except KeyError:
             # model_name が見つからない場合は KeyError を発生させる (ユーザー承認済み)
             # 呼び出し元でハンドリングされることを期待
             raise KeyError(f"設定にモデル '{model_name}' が見つかりません。")

        # model_config は辞書であることが保証される
        # model_config (辞書) から key に対応する値を取得。見つからなければ default (None) を返す
        # model_config.get の戻り値も str | float | None になるはず
        value = model_config.get(key, default)

        # 念のため、戻り値の型チェック (オプションだが推奨)
        if value is not None and not isinstance(value, (str, float)):
             # 想定外の型の場合の処理 (ログ出力、エラー、None を返すなど)
             logger.warning(f"設定値 {model_name}.{key} が予期しない型です: {type(value)}")
             # ここでは None を返すか、エラーにするか要検討 (None を返すのが無難か)
             return None # あるいは raise TypeError(...)

        return value # str | float | None

```

**修正のポイント:**

1.  **タイポ修正と KeyError:** `self._config_data[model_name]` を使用し、`model_name` が存在しない場合は意図通り `KeyError` を発生させます。これにより `model_config` が `None` になる可能性がなくなり、エラー 1 が解消されます。`try...except KeyError` を追加してエラーメッセージを明確にすることも可能です。
2.  **型ヒントの精緻化:** 戻り値の型ヒントを、実際に返しうる `str | float | None` に修正します (もし `int` を返さない場合)。`default` 引数の型も `None` に限定します。これにより Mypy の推論精度が向上し、エラー 2 の解消につながる可能性があります。
3.  **(任意) 戻り値の型チェック:** 念のため、`model_config.get` から取得した値が本当に `str`, `float`, `None` のいずれかであるかチェックする処理を追加することも検討できます。

## 次のステップ

1.  この計画を `../memory-bank/mypy_fix_plan.md` として保存します。
2.  Code モードに切り替えて、上記の修正を `src/image_annotator_lib/core/config.py` に適用します。
3.  再度 Mypy を実行し、エラーが解消されたことを確認します。