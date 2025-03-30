import polars as pl

from tagger_wrapper_lib.core.base import ONNXModel


# E621カテゴリ番号の定義
# 参考: https://e621.net/wiki_pages/11262
# 0: general, 1: artist, 3: copyright, 4: character, 5: species, 6: invalid, 7: meta, 8: lore
# これらの番号は公式E621タグカテゴリシステムに基づいており、Z3D_E621モデルのCSVで使用されている
# カテゴリナンバーと一致していることを確認済み
class E621Categories:
    GENERAL = 0
    ARTIST = 1
    COPYRIGHT = 3
    CHARACTER = 4
    SPECIES = 5
    INVALID = 6
    META = 7
    LORE = 8


class WDTagger(ONNXModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        # カテゴリマッピング：WD-Taggerのカテゴリ番号を定義
        self.CATEGORY_MAPPING = {"rating": 9, "general": 0, "character": 4}
        # カテゴリIDとインデックス属性名の対応マップ
        self._category_attr_map = {
            "rating": "rating_indexes",
            "general": "general_indexes",
            "character": "character_indexes",
        }

    def _load_labels(self) -> None:
        """ラベル情報をロードし、カテゴリごとのインデックスを設定します。"""
        try:
            # ラベルファイルをpolarsで読み込み
            tags_df = pl.read_csv(self.components["csv_path"])

            # ラベル名を取得
            self.labels = tags_df["name"].to_list()

            # カテゴリー情報を取得
            categories = tags_df["category"].to_list()

            # 全てのインデックスを初期化
            self._init_empty_indexes()

            # 各カテゴリのインデックスを抽出
            self._extract_category_indexes(categories)

            self.logger.info(f"WDタガーラベル情報を読み込みました: 合計{len(self.labels)}個のタグ")
        except Exception as e:
            self.logger.error(f"ラベル情報の読み込みに失敗しました: {e}")
            # デフォルト値を設定
            self._init_empty_indexes()
            self.labels = []

    def _init_empty_indexes(self) -> None:
        """各カテゴリのインデックスリストを空で初期化します。"""
        for attr_name in self._category_attr_map.values():
            setattr(self, attr_name, [])

    def _extract_category_indexes(self, categories: list[int]) -> None:
        """カテゴリリストから各カテゴリのインデックスを抽出します。"""
        for category_key, category_id in self.CATEGORY_MAPPING.items():
            attr_name = self._category_attr_map[category_key]
            indexes = [i for i, cat in enumerate(categories) if cat == category_id]
            setattr(self, attr_name, indexes)


class Z3D_E621Tagger(ONNXModel):
    def __init__(self, model_name: str):
        super().__init__(model_name=model_name)
        # カテゴリとインデックス属性の対応マップ
        self._category_attr_map = {
            E621Categories.GENERAL: "general_indexes",
            E621Categories.ARTIST: "artist_indexes",
            E621Categories.CHARACTER: "character_indexes",
            E621Categories.SPECIES: "species_indexes",
            E621Categories.COPYRIGHT: "copyright_indexes",
        }
        # レーティングタグ
        self._rating_tags = ["explicit", "questionable", "safe"]

    def _load_labels(self) -> None:
        """Z3D_E621用のラベル情報をロードします。

        E621のカテゴリナンバーを使用:
        0: general - 一般タグ
        1: artist - アーティストタグ
        3: copyright - 著作権タグ
        4: character - キャラクタータグ
        5: species - 種族タグ
        """
        try:
            # CSVファイルの読み込みとラベル取得
            tags_df = pl.read_csv(self.components["csv_path"])
            self.labels = tags_df["name"].to_list()
            self._init_empty_indexes()  # 各カテゴリのインデックスを初期化

            # カテゴリ処理
            self._process_categories(tags_df)

            self.logger.info(f"Z3D_E621ラベル情報を読み込みました: 合計{len(self.labels)}個のタグ")

        except Exception as e:
            self.logger.error(f"ラベル情報の読み込みに失敗しました: {e}")
            # エラー時はデフォルト値を設定
            self._init_empty_indexes()
            self.labels = []

    def _process_categories(self, tags_df: pl.DataFrame) -> None:
        """CSVからカテゴリ情報を処理します。"""
        if "category" in tags_df.columns:
            # カテゴリカラムがある場合
            categories = tags_df["category"].to_list()

            # 3つのステップでカテゴリ処理を行う
            self._extract_category_indexes(categories)  # 1. カテゴリに基づくインデックス抽出
            self._extract_rating_indexes()  # 2. レーティングタグの抽出
            self._handle_unclassified_tags()  # 3. 未分類タグの処理
        else:
            # カテゴリカラムがない場合はすべてを一般タグとして扱う
            self.general_indexes = list(range(len(self.labels)))

    def _init_empty_indexes(self) -> None:
        """各カテゴリのインデックスリストを空で初期化します。"""
        for attr_name in self._category_attr_map.values():
            setattr(self, attr_name, [])
        self.rating_indexes = []

    def _extract_category_indexes(self, categories: list[int]) -> None:
        """カテゴリリストから各カテゴリのインデックスを抽出します。"""
        for category_id, attr_name in self._category_attr_map.items():
            indexes = [i for i, cat in enumerate(categories) if cat == category_id]
            setattr(self, attr_name, indexes)

    def _extract_rating_indexes(self) -> None:
        """レーティングタグのインデックスを抽出します。"""
        self.rating_indexes = [i for i, name in enumerate(self.labels) if name in self._rating_tags]

    def _handle_unclassified_tags(self) -> None:
        """未分類のタグを一般タグとして処理します。"""
        # 特殊カテゴリに分類されたインデックスを集約
        all_special_indexes = set()
        for attr_name in self._category_attr_map.values():
            if attr_name != "general_indexes":  # 一般タグは除外
                all_special_indexes.update(getattr(self, attr_name))
        all_special_indexes.update(self.rating_indexes)

        # 未分類のタグを一般タグに追加
        unclassified = [
            i
            for i in range(len(self.labels))
            if i not in all_special_indexes and i not in self.general_indexes
        ]
        self.general_indexes.extend(unclassified)
