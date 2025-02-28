class ModelNotFoundError(Exception):
    """指定されたモデルファイルが見つからない場合に発生する例外"""

    pass


class ModelLoadError(Exception):
    """モデルのロード中にエラーが発生した場合に発生する例外"""

    pass


class InvalidModelFileError(Exception):
    """モデルファイルが不正な形式である場合に発生する例外"""

    pass


class OutOfMemoryError(Exception):
    """モデルのロードまたは推論中にメモリ不足が発生した場合に発生する例外"""

    pass