import unittest
import numpy as np
from pathlib import Path
import yaml
import sys
import os
import soundfile as sf

# プロジェクトルートをPythonパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.voice_clone import VoiceCloneManager

class TestVoiceClone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # テスト用の設定読み込み
        with open('config.yml', 'r') as f:
            cls.config = yaml.safe_load(f)
        
        # テスト用ディレクトリの作成
        Path("tests/data").mkdir(parents=True, exist_ok=True)
        
        # テスト用音声ファイルの生成
        dummy_audio = np.zeros(16000)  # 1秒の無音データ
        sf.write("tests/data/test_voice_sample.wav", dummy_audio, 16000)
        
        cls.voice_clone = VoiceCloneManager(cls.config)
        cls.test_audio = "tests/data/test_voice_sample.wav"
        cls.test_profile = "test_speaker"
        
    def test_model_loading(self):
        """モデル読み込みのテスト"""
        self.assertIsNotNone(self.voice_clone.model)
        
    def test_feature_extraction(self):
        """特徴抽出のテスト"""
        # テスト用の音声ファイルが存在することを確認
        self.assertTrue(Path(self.test_audio).exists())
        
        features = self.voice_clone.extract_voice_features(self.test_audio)
        
        self.assertIsNotNone(features)
        self.assertIn('audio_path', features)
        
        # 特徴ベクトルの次元数チェック（設定ファイルに依存しない形に）
        self.assertTrue(len(features['speaker_embedding'].shape) > 0)
        
    def test_profile_management(self):
        """プロファイル管理のテスト"""
        # テスト用特徴量
        test_features = {
            'speaker_embedding': np.random.rand(256),
            'f0_stats': np.random.rand(2),
            'energy_stats': np.random.rand(2)
        }
        
        # プロファイル保存
        success = self.voice_clone.save_voice_profile(
            self.test_profile,
            test_features
        )
        self.assertTrue(success)
        
        # プロファイルファイルの存在確認
        profile_path = Path(self.config['voice_clone']['profiles_dir'])
        self.assertTrue((profile_path / f"{self.test_profile}.npz").exists())
        
    def test_voice_conversion(self):
        """音声変換機能のテスト"""
        # テスト用プロファイルの作成
        test_features = {
            'speaker_embedding': np.zeros(256),
            'f0_stats': {'mean': 0.0, 'std': 0.0},
            'mel_spec_mean': np.zeros(80),
            'sample_count': 1
        }
        self.voice_clone.save_voice_profile("test_conversion", test_features)
        
        # テストケース
        test_cases = [
            {
                'name': "基本的な変換",
                'text': "こんにちは",
                'profile': "test_conversion",
                'should_succeed': True
            },
            {
                'name': "空のテキスト",
                'text': "",
                'profile': "test_conversion",
                'should_succeed': False
            },
            {
                'name': "存在しないプロファイル",
                'text': "テスト",
                'profile': "non_existent_profile",
                'should_succeed': False
            },
            {
                'name': "長いテキスト",
                'text': "これは少し長めのテキストです。音声合成の動作を確認します。",
                'profile': "test_conversion",
                'should_succeed': True
            }
        ]
        
        for case in test_cases:
            with self.subTest(case['name']):
                # 音声変換の実行
                result = self.voice_clone.convert_voice(case['text'], case['profile'])
                
                if case['should_succeed']:
                    self.assertIsNotNone(result, f"音声変換が失敗: {case['name']}")
                    self.assertIsInstance(result, np.ndarray, "戻り値が numpy 配列ではありません")
                    self.assertTrue(len(result) > 0, "音声データが空です")
                    
                    # サンプリングレートの確認
                    self.assertEqual(
                        self.voice_clone.sample_rate, 
                        22050,  # Tacotron2 のデフォルト
                        "不正なサンプリングレート"
                    )
                else:
                    self.assertIsNone(result, f"エラーケースで音声が生成されました: {case['name']}")

    def test_audio_playback(self):
        """音声再生機能のテスト"""
        # テスト用の音声データ
        test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))  # 1秒の440Hz正弦波
        
        try:
            # 音声再生
            self.voice_clone.play_audio(test_audio)
            success = True
        except Exception as e:
            success = False
            self.fail(f"音声再生でエラー発生: {str(e)}")
        
        self.assertTrue(success, "音声再生に失敗しました")

    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 存在しない音声ファイル
        features = self.voice_clone.extract_voice_features("nonexistent.wav")
        self.assertIsNone(features)
        
        # 存在しないプロファイル
        audio = self.voice_clone.convert_voice("テスト", "nonexistent_profile")
        self.assertIsNone(audio)
        
    def test_voice_profile_creation(self):
        """音声プロファイル作成のテスト"""
        # テスト用の音声ファイルを複数生成
        test_files = []
        for i in range(3):
            file_path = f"tests/data/test_voice_{i}.wav"
            dummy_audio = np.random.randn(22050)  # 1秒のランダム音声
            sf.write(file_path, dummy_audio, 22050)
            test_files.append(file_path)
        
        # プロファイル作成
        success = self.voice_clone.create_voice_profile(
            "test_multi_sample",
            test_files
        )
        
        self.assertTrue(success)
        self.assertIn("test_multi_sample", self.voice_clone.voice_profiles)
        
        # プロファイルの内容確認
        profile = self.voice_clone.voice_profiles["test_multi_sample"]
        self.assertIn('f0_stats', profile)
        self.assertIn('mel_spec_mean', profile)
        self.assertEqual(profile['sample_count'], 3)

    @classmethod
    def tearDownClass(cls):
        """テスト終了時の後処理"""
        # テストで作成したファイルの削除
        test_files = [
            "tests/data/test_voice_sample.wav",
            "tests/data/test_voice_0.wav",
            "tests/data/test_voice_1.wav",
            "tests/data/test_voice_2.wav"
        ]
        for file in test_files:
            try:
                Path(file).unlink(missing_ok=True)
            except Exception as e:
                print(f"警告: テストファイルの削除に失敗 {file}: {e}")
            
        # プロファイルディレクトリのクリーンアップ
        profile_dir = Path(cls.config['voice_clone']['profiles_dir'])
        if profile_dir.exists():
            for profile in profile_dir.glob("test_*.npz"):
                try:
                    profile.unlink()
                except Exception as e:
                    print(f"警告: プロファイルの削除に失敗 {profile}: {e}")

if __name__ == '__main__':
    unittest.main() 