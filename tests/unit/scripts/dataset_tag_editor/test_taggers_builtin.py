from unittest.mock import patch

from scripts.dataset_tag_editor.taggers_builtin import (
    BLIP,
    BLIP2,
    Z3D_E621,
    DeepDanbooru,
    GITLarge,
    WaifuDiffusion,
    WaifuDiffusionTimm,
)


class TestBLIP:
    """BLIPタガーのテスト"""

    def test_init(self):
        """初期化のテスト"""
        tagger = BLIP()
        assert tagger.name() == "BLIP"

    def test_predict(self, test_image, mock_interrogator):
        """予測機能のテスト - タグの分割処理を確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.BLIPLargeCaptioning",
            return_value=mock_interrogator,
        ):
            mock_interrogator.apply.return_value = ["a person, walking, sunny day"]

            tagger = BLIP()
            result = tagger.predict(test_image)
            assert len(result) == 3
            assert all(isinstance(tag, str) for tag in result)
            mock_interrogator.apply.assert_called_once_with(test_image)

    def test_start_stop(self, mock_interrogator):
        """start/stopメソッドのテスト"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.BLIPLargeCaptioning",
            return_value=mock_interrogator,
        ):
            tagger = BLIP()
            tagger.start()
            mock_interrogator.load.assert_called_once()
            
            tagger.stop()
            mock_interrogator.unload.assert_called_once()


class TestBLIP2:
    """BLIP2タガーのテスト"""

    def test_init(self):
        """初期化のテスト - リポジトリ名の設定を確認"""
        repo_name = "blip2-opt-2.7b"
        tagger = BLIP2(repo_name)
        assert tagger.name() == repo_name

    def test_predict(self, test_image, mock_interrogator):
        """予測機能のテスト - タグの分割処理を確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.BLIP2Captioning",
            return_value=mock_interrogator,
        ):
            mock_interrogator.apply.return_value = ["a cat, sleeping, on bed"]

            tagger = BLIP2("blip2-opt-2.7b")
            result = tagger.predict(test_image)
            assert len(result) == 3
            assert all(isinstance(tag, str) for tag in result)
            mock_interrogator.apply.assert_called_once_with(test_image)

    def test_start_stop(self, mock_interrogator):
        """start/stopメソッドのテスト"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.BLIP2Captioning",
            return_value=mock_interrogator,
        ):
            tagger = BLIP2("blip2-opt-2.7b")
            tagger.start()
            mock_interrogator.load.assert_called_once()
            
            tagger.stop()
            mock_interrogator.unload.assert_called_once()


class TestGITLarge:
    """GITLargeタガーのテスト"""

    def test_init(self):
        """初期化のテスト"""
        tagger = GITLarge()
        assert tagger.name() == "GIT-large-COCO"

    def test_predict(self, test_image, mock_interrogator):
        """予測機能のテスト - タグの分割処理を確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.GITLargeCaptioning",
            return_value=mock_interrogator,
        ):
            mock_interrogator.apply.return_value = ["a dog, running, in park"]

            tagger = GITLarge()
            result = tagger.predict(test_image)
            assert len(result) == 3
            assert all(isinstance(tag, str) for tag in result)
            mock_interrogator.apply.assert_called_once_with(test_image)

    def test_start_stop(self, mock_interrogator):
        """start/stopメソッドのテスト"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.GITLargeCaptioning",
            return_value=mock_interrogator,
        ):
            tagger = GITLarge()
            tagger.start()
            mock_interrogator.load.assert_called_once()
            
            tagger.stop()
            mock_interrogator.unload.assert_called_once()


class TestDeepDanbooru:
    """DeepDanbooruタガーのテスト"""

    def test_init(self):
        """初期化のテスト - レーティング設定を確認"""
        tagger = DeepDanbooru()
        assert tagger.name() == "DeepDanbooru"
        assert not tagger.use_rating

    def test_predict_with_threshold(self, test_image, mock_interrogator):
        """閾値処理のテスト - 確率による閾値フィルタリングを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.DepDanbooruTagger",
            return_value=mock_interrogator,
        ):
            # レーティングタグは末尾3つに配置
            mock_labels = [
                ("1girl", 0.9),
                ("blue_eyes", 0.8),
                ("normal_tags", 0.7),
                ("rating:safe", 0.95),
                ("rating:questionable", 0.05),
                ("rating:explicit", 0.0),
            ]
            mock_interrogator.apply.return_value = mock_labels

            tagger = DeepDanbooru(use_rating=False)
            result = tagger.predict(test_image, threshold=0.85)

            # レーティングタグを除外した後（labels[:-3]）に閾値フィルタリング
            # 0.85より大きいタグは"1girl"のみ
            expected_tags = ["1girl"]
            assert result == expected_tags
            mock_interrogator.apply.assert_called_once_with(test_image)

    def test_predict_with_rating(self, test_image, mock_interrogator):
        """レーティング処理のテスト - レーティングタグの扱いを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.DepDanbooruTagger",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                ("1girl", 0.9),
                ("blue_eyes", 0.8),
                ("rating:safe", 0.95),
                ("rating:questionable", 0.05),
                ("rating:explicit", 0.0),
            ]
            mock_interrogator.apply.return_value = mock_labels

            tagger = DeepDanbooru(use_rating=True)
            result = tagger.predict(test_image)
            assert "1girl" in result
            assert "blue_eyes" in result
            assert "rating:safe" in result

    def test_start_stop(self, mock_interrogator):
        """start/stopメソッドのテスト"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.DepDanbooruTagger",
            return_value=mock_interrogator,
        ):
            tagger = DeepDanbooru()
            tagger.start()
            mock_interrogator.load.assert_called_once()
            
            tagger.stop()
            mock_interrogator.unload.assert_called_once()


class TestWaifuDiffusion:
    """WaifuDiffusionタガーのテスト"""

    def test_init(self):
        """初期化のテスト - 閾値とレーティング設定を確認"""
        repo_name = "waifu-diffusion-v1-4"
        threshold = 0.5
        tagger = WaifuDiffusion(repo_name, threshold)
        assert tagger.name() == repo_name
        assert tagger.threshold == threshold
        assert not tagger.use_rating

    def test_predict_with_threshold(self, test_image, mock_interrogator):
        """閾値処理のテスト - レーティングを除外した閾値フィルタリングを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTagger",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                ("rating:safe", 0.99),
                ("rating:questionable", 0.01),
                ("rating:explicit", 0.00),
                ("rating:general", 0.00),
                ("1girl", 0.95),
                ("blue_hair", 0.85),
                ("school_uniform", 0.75),
            ]
            mock_interrogator.apply.return_value = mock_labels

            tagger = WaifuDiffusion(
                "waifu-diffusion-v1-4", threshold=0.8, use_rating=False
            )
            result = tagger.predict(test_image, threshold=0.8)  # 明示的に閾値を指定
            assert set(result) == {"1girl", "blue_hair"}

    def test_predict_without_threshold(self, test_image, mock_interrogator):
        """閾値なしのテスト - 全タグを返すことを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTagger",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                ("rating:safe", 0.99),
                ("rating:questionable", 0.01),
                ("rating:explicit", 0.00),
                ("rating:general", 0.00),
                ("1girl", 0.95),
                ("blue_hair", 0.85),
                ("school_uniform", 0.75),
            ]
            mock_interrogator.apply.return_value = mock_labels

            tagger = WaifuDiffusion(
                "waifu-diffusion-v1-4", threshold=0.8, use_rating=False
            )
            result = tagger.predict(test_image)  # 閾値を指定しない
            assert set(result) == {"1girl", "blue_hair", "school_uniform"}

    def test_predict_with_rating(self, test_image, mock_interrogator):
        """レーティング処理のテスト - レーティングタグを含むケースを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTagger",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                ("rating:safe", 0.99),
                ("rating:questionable", 0.01),
                ("rating:explicit", 0.00),
                ("rating:general", 0.00),
                ("1girl", 0.95),
                ("blue_hair", 0.85),
            ]
            mock_interrogator.apply.return_value = mock_labels

            tagger = WaifuDiffusion(
                "waifu-diffusion-v1-4", threshold=0.8, use_rating=True
            )
            result = tagger.predict(test_image, threshold=0.8)
            assert set(result) == {"rating:safe", "1girl", "blue_hair"}

    def test_predict_with_negative_threshold(self, test_image, mock_interrogator):
        """負の閾値処理のテスト - デフォルト閾値の使用を確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTagger",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                ("rating:safe", 0.99),
                ("rating:questionable", 0.01),
                ("rating:explicit", 0.00),
                ("rating:general", 0.00),
                ("1girl", 0.95),
                ("blue_hair", 0.85),
                ("school_uniform", 0.75),
            ]
            mock_interrogator.apply.return_value = mock_labels

            default_threshold = 0.8
            tagger = WaifuDiffusion(
                "waifu-diffusion-v1-4", threshold=default_threshold, use_rating=False
            )
            result = tagger.predict(test_image, threshold=-1)  # 負の閾値を指定
            assert set(result) == {"1girl", "blue_hair"}  # デフォルト閾値0.8が使用される

    def test_start_stop(self, mock_interrogator):
        """start/stopメソッドのテスト"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTagger",
            return_value=mock_interrogator,
        ):
            tagger = WaifuDiffusion("waifu-diffusion-v1-4", threshold=0.5)
            tagger.start()
            mock_interrogator.load.assert_called_once()
            
            tagger.stop()
            mock_interrogator.unload.assert_called_once()


class TestWaifuDiffusionTimm:
    """WaifuDiffusionTimmタガーのテスト"""

    def test_init(self):
        """初期化のテスト - バッチサイズ設定を確認"""
        repo_name = "waifu-diffusion-v1-4"
        threshold = 0.5
        batch_size = 4
        tagger = WaifuDiffusionTimm(repo_name, threshold, batch_size=batch_size)
        assert tagger.name() == repo_name
        assert tagger.threshold == threshold
        assert tagger.batch_size == batch_size

    def test_predict(self, test_image, mock_interrogator):
        """予測機能のテスト - 閾値フィルタリングを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTaggerTimm",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                ("rating:safe", 0.99),
                ("rating:questionable", 0.01),
                ("rating:explicit", 0.00),
                ("rating:general", 0.00),
                ("1girl", 0.95),
                ("blue_hair", 0.85),
            ]
            mock_interrogator.apply.return_value = mock_labels

            tagger = WaifuDiffusionTimm(
                "waifu-diffusion-v1-4", threshold=0.8, use_rating=False
            )
            result = tagger.predict(test_image, threshold=0.8)
            assert set(result) == {"1girl", "blue_hair"}

    def test_predict_without_threshold(self, test_image, mock_interrogator):
        """予測機能のテスト - 閾値なしの場合を確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTaggerTimm",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                ("rating:safe", 0.99),
                ("rating:questionable", 0.01),
                ("rating:explicit", 0.00),
                ("rating:general", 0.00),
                ("1girl", 0.95),
                ("blue_hair", 0.85),
                ("school_uniform", 0.75),
            ]
            mock_interrogator.apply.return_value = mock_labels

            tagger = WaifuDiffusionTimm(
                "waifu-diffusion-v1-4", threshold=0.8, use_rating=False
            )
            result = tagger.predict(test_image)  # 閾値を指定しない
            assert set(result) == {"1girl", "blue_hair", "school_uniform"}

    def test_predict_with_negative_threshold(self, test_image, mock_interrogator):
        """予測機能のテスト - 負の閾値でのデフォルト閾値使用を確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTaggerTimm",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                ("rating:safe", 0.99),
                ("rating:questionable", 0.01),
                ("rating:explicit", 0.00),
                ("rating:general", 0.00),
                ("1girl", 0.95),
                ("blue_hair", 0.85),
                ("school_uniform", 0.75),
            ]
            mock_interrogator.apply.return_value = mock_labels

            default_threshold = 0.8
            tagger = WaifuDiffusionTimm(
                "waifu-diffusion-v1-4", threshold=default_threshold, use_rating=False
            )
            result = tagger.predict(test_image, threshold=-1)  # 負の閾値を指定
            assert set(result) == {"1girl", "blue_hair"}  # デフォルト閾値0.8が使用される

    def test_predict_with_rating(self, test_image, mock_interrogator):
        """予測機能のテスト - レーティングタグを含むケースを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTaggerTimm",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                ("rating:safe", 0.99),
                ("rating:questionable", 0.01),
                ("rating:explicit", 0.00),
                ("rating:general", 0.00),
                ("1girl", 0.95),
                ("blue_hair", 0.85),
            ]
            mock_interrogator.apply.return_value = mock_labels

            tagger = WaifuDiffusionTimm(
                "waifu-diffusion-v1-4", threshold=0.8, use_rating=True
            )
            result = tagger.predict(test_image, threshold=0.8)
            assert set(result) == {"rating:safe", "1girl", "blue_hair"}

    def test_predict_pipe(self, test_image, mock_interrogator):
        """バッチ処理のテスト - バッチ処理での閾値フィルタリングを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTaggerTimm",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                [
                    [
                        ("rating:safe", 0.99),
                        ("rating:questionable", 0.01),
                        ("rating:explicit", 0.00),
                        ("rating:general", 0.00),
                        ("1girl", 0.95),
                        ("blue_hair", 0.85),
                    ]
                ]
            ]
            mock_interrogator.apply_multi.return_value = mock_labels

            tagger = WaifuDiffusionTimm(
                "waifu-diffusion-v1-4", threshold=0.8, use_rating=False
            )
            results = list(
                tagger.predict_pipe([test_image], threshold=0.8)
            )  # 明示的に閾値を指定
            assert len(results) == 1
            assert set(results[0]) == {"1girl", "blue_hair"}

    def test_predict_pipe_without_threshold(self, test_image, mock_interrogator):
        """バッチ処理のテスト - 閾値なしの場合を確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTaggerTimm",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                [
                    [
                        ("rating:safe", 0.99),
                        ("rating:questionable", 0.01),
                        ("rating:explicit", 0.00),
                        ("rating:general", 0.00),
                        ("1girl", 0.95),
                        ("blue_hair", 0.85),
                        ("school_uniform", 0.75),
                    ]
                ]
            ]
            mock_interrogator.apply_multi.return_value = mock_labels

            tagger = WaifuDiffusionTimm(
                "waifu-diffusion-v1-4", threshold=0.8, use_rating=False
            )
            results = list(tagger.predict_pipe([test_image]))  # 閾値を指定しない
            assert len(results) == 1
            assert set(results[0]) == {"1girl", "blue_hair", "school_uniform"}

    def test_predict_pipe_with_rating(self, test_image, mock_interrogator):
        """バッチ処理のテスト - レーティングタグを含むケースを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTaggerTimm",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                [
                    [
                        ("rating:safe", 0.99),
                        ("rating:questionable", 0.01),
                        ("rating:explicit", 0.00),
                        ("rating:general", 0.00),
                        ("1girl", 0.95),
                        ("blue_hair", 0.85),
                    ]
                ]
            ]
            mock_interrogator.apply_multi.return_value = mock_labels

            tagger = WaifuDiffusionTimm(
                "waifu-diffusion-v1-4", threshold=0.8, use_rating=True
            )
            results = list(tagger.predict_pipe([test_image], threshold=0.8))
            assert len(results) == 1
            assert set(results[0]) == {"rating:safe", "1girl", "blue_hair"}

    def test_predict_pipe_with_negative_threshold(self, test_image, mock_interrogator):
        """バッチ処理のテスト - 負の閾値でのデフォルト閾値使用を確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTaggerTimm",
            return_value=mock_interrogator,
        ):
            mock_labels = [
                [
                    [
                        ("rating:safe", 0.99),
                        ("rating:questionable", 0.01),
                        ("rating:explicit", 0.00),
                        ("rating:general", 0.00),
                        ("1girl", 0.95),
                        ("blue_hair", 0.85),
                        ("school_uniform", 0.75),
                    ]
                ]
            ]
            mock_interrogator.apply_multi.return_value = mock_labels

            default_threshold = 0.8
            tagger = WaifuDiffusionTimm(
                "waifu-diffusion-v1-4", threshold=default_threshold, use_rating=False
            )
            results = list(tagger.predict_pipe([test_image], threshold=-1))
            assert len(results) == 1
            assert set(results[0]) == {"1girl", "blue_hair"}

    def test_start_stop(self, mock_interrogator):
        """start/stopメソッドのテスト"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTaggerTimm",
            return_value=mock_interrogator,
        ):
            tagger = WaifuDiffusionTimm("waifu-diffusion-v1-4", threshold=0.5)
            tagger.start()
            mock_interrogator.load.assert_called_once()
            
            tagger.stop()
            mock_interrogator.unload.assert_called_once()


class TestZ3DE621:
    """Z3D_E621タガーのテスト"""

    def test_init(self):
        """初期化のテスト"""
        tagger = Z3D_E621()
        assert tagger.name() == "Z3D-E621-Convnext"

    def test_predict(self, test_image, mock_interrogator):
        """予測機能のテスト - 閾値フィルタリングを確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTagger",
            return_value=mock_interrogator,
        ):
            mock_labels = [("animal", 0.95), ("furry", 0.85), ("cute", 0.75)]
            mock_interrogator.apply.return_value = mock_labels

            tagger = Z3D_E621()
            result = tagger.predict(test_image, threshold=0.8)
            assert set(result) == {"animal", "furry"}

    def test_predict_without_threshold(self, test_image, mock_interrogator):
        """予測機能のテスト - 閾値なしの場合を確認"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTagger",
            return_value=mock_interrogator,
        ):
            mock_labels = [("animal", 0.95), ("furry", 0.85), ("cute", 0.75)]
            mock_interrogator.apply.return_value = mock_labels

            tagger = Z3D_E621()
            result = tagger.predict(test_image)  # 閾値を指定しない
            assert set(result) == {"animal", "furry", "cute"}

    def test_start_stop(self, mock_interrogator):
        """start/stopメソッドのテスト"""
        with patch(
            "scripts.dataset_tag_editor.taggers_builtin.WaifuDiffusionTagger",
            return_value=mock_interrogator,
        ):
            tagger = Z3D_E621()
            tagger.start()
            mock_interrogator.load.assert_called_once()
            
            tagger.stop()
            mock_interrogator.unload.assert_called_once()
