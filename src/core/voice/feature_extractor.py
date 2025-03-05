import numpy as np
import torch
import librosa
import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Union
import warnings
import gc
import subprocess
import sys
from ..audio.processor import AudioProcessor

class FeatureExtractor:
    """
    音声特徴量抽出を担当するクラス
    
    音声ファイルまたは音声データから話者埋め込み、F0特徴量、メルスペクトログラムなどを抽出します。
    """
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # デバイス設定
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # オーディオ処理ユーティリティ
        self.audio_processor = AudioProcessor(config)
        
        # デフォルトサンプリングレート
        self.sample_rate = 22050
        if config:
            self.sample_rate = config.get('voice_clone', 'sample_rate', self.sample_rate)
            
        # SpeechBrainモデルは必要時に初期化
        self._speaker_embedding_model = None
        
    def extract_features(self, audio_input) -> Optional[Dict]:
        """
        音声ファイルまたは音声データから特徴量を抽出する
        
        Args:
            audio_input: 音声ファイルのパス（文字列）または前処理済み音声データ（numpy配列）
            
        Returns:
            抽出された特徴量の辞書、失敗した場合はNone
        """
        try:
            # 音声データ（numpy配列）かファイルパス（文字列）かを判定
            is_audio_data = isinstance(audio_input, np.ndarray)
            
            if not is_audio_data:
                # ファイルパスの場合
                if not Path(audio_input).exists():
                    self.logger.error(f"ファイルが存在しません: {audio_input}")
                    return None
                    
                # 音声読み込み・前処理
                audio_path = audio_input
                wav = self.preprocess_audio(audio_path)
                if wav is None:
                    self.logger.error(f"音声の前処理に失敗しました: {audio_path}")
                    return None
                
                file_info = os.path.basename(audio_path)
            else:
                # 既に前処理された音声データの場合
                wav = audio_input
                file_info = "前処理済み音声データ"
            
            # 話者埋め込み抽出（VoxCelebモデル使用）
            self.logger.info(f"話者埋め込みを抽出中: {file_info}")
            speaker_embedding = self._extract_speaker_embedding(wav)
            
            if speaker_embedding is None:
                self.logger.error(f"話者埋め込みの抽出に失敗: {file_info}")
                return None
            
            # サンプリングレート
            sr = self.sample_rate
            
            # F0（基本周波数）抽出
            self.logger.debug(f"F0特徴量を抽出中: {file_info}")
            f0, voiced_flag, _ = librosa.pyin(
                wav, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=1024,
                win_length=512,
                hop_length=256
            )
            
            # 有声区間の検出
            if np.sum(voiced_flag) < len(voiced_flag) * 0.1:  # 有声区間が10%未満の場合
                self.logger.warning(f"有声区間が少なすぎます: {np.sum(voiced_flag)}/{len(voiced_flag)}")
                # フォールバック: すべての区間を有声とみなす
                voiced_flag = np.ones_like(voiced_flag, dtype=bool)
            
            # 有声区間のF0のみを使用
            f0_voiced = f0[voiced_flag]
            
            # F0統計量
            f0_mean = np.mean(f0_voiced) if len(f0_voiced) > 0 else 0
            f0_std = np.std(f0_voiced) if len(f0_voiced) > 1 else 0
            
            # メルスペクトログラムの抽出
            self.logger.debug(f"メルスペクトログラムを抽出中: {file_info}")
            mel_spec = librosa.feature.melspectrogram(
                y=wav, 
                sr=sr,
                n_fft=1024,
                hop_length=256,
                win_length=512,
                n_mels=80
            )
            
            # 対数変換
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # 特徴量を辞書にまとめる
            features = {
                'speaker_embedding': speaker_embedding,
                'f0': f0_voiced,
                'f0_stats': {
                    'mean': float(f0_mean),
                    'std': float(f0_std)
                },
                'voiced_flag': voiced_flag,
                'mel_spec': log_mel_spec
            }
            
            self.logger.info(f"特徴抽出完了: スピーカー埋め込みサイズ {speaker_embedding.shape}, F0サンプル数 {len(f0_voiced)}")
            return features
            
        except Exception as e:
            self.logger.error(f"特徴抽出エラー: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
            
    def preprocess_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """
        音声ファイルを前処理する
        
        Args:
            audio_path: 音声ファイルのパス
            
        Returns:
            前処理された音声データ（numpy配列）、失敗した場合はNone
        """
        try:
            self.logger.info(f"音声ファイルを前処理中: {Path(audio_path).name}")
            
            # 音声読み込み
            wav, orig_sr = librosa.load(audio_path, sr=None, mono=True)
            
            # サンプリングレート変換（必要な場合）
            if orig_sr != self.sample_rate:
                wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=self.sample_rate)
                self.logger.debug(f"サンプリングレート変換: {orig_sr}Hz → {self.sample_rate}Hz")
            
            # 無音部分のトリミング
            wav, _ = librosa.effects.trim(wav, top_db=20)
            
            # ボリューム正規化
            wav = librosa.util.normalize(wav)
            
            self.logger.info(f"音声前処理完了: 長さ {len(wav) / self.sample_rate:.2f}秒, 最大振幅 {np.max(np.abs(wav)):.2f}")
            return wav
            
        except Exception as e:
            self.logger.error(f"音声前処理エラー: {str(e)}")
            return None
    
    def _extract_speaker_embedding(self, wav: np.ndarray) -> Optional[np.ndarray]:
        """
        音声データから話者埋め込みを抽出する
        
        Args:
            wav: 音声データ（numpy配列）
            
        Returns:
            話者埋め込みベクトル（numpy配列）、失敗した場合はNone
        """
        try:
            start_time = time.time()
            self.logger.info("話者埋め込み抽出を開始します")
            
            # SpeechBrainのキャッシュディレクトリを確認
            cache_dir = self._ensure_cache_dir()
            
            # SpeechBrainリポジトリを確認
            repo_path = self._ensure_speechbrain_repo(cache_dir)
            if repo_path is None:
                return self._fallback_speaker_embedding(wav)
            
            try:
                # PyTorchの警告を抑制
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                
                    # 信頼できるリポジトリから事前学習済みモデルを読み込む
                    self.logger.info("埋め込みモデルを読み込んでいます...")
                    classifier = torch.hub.load(
                        str(repo_path) if repo_path else 'speechbrain/spkrec-ecapa-voxceleb', 
                        'ecapa_tdnn',
                        source='local' if repo_path else 'github',
                        trust_repo=True
                    )
                    
                    # GPUが利用可能ならGPUに転送
                    if torch.cuda.is_available():
                        classifier = classifier.to(self.device)
                    
                    # 音声データをテンソルに変換
                    tensor_wav = torch.from_numpy(wav).float()
                    if torch.cuda.is_available():
                        tensor_wav = tensor_wav.to(self.device)
                    
                    # バッチ次元を追加
                    tensor_wav = tensor_wav.unsqueeze(0)
                    
                    # 埋め込みの抽出
                    with torch.no_grad():
                        try:
                            embeddings = classifier.encode_batch(tensor_wav)
                            embeddings_np = embeddings.cpu().numpy().squeeze()
                        except Exception as e:
                            self.logger.error(f"埋め込み抽出中にエラー: {str(e)}")
                            return self._fallback_speaker_embedding(wav)
                    
                    # 正規化
                    norm = np.linalg.norm(embeddings_np)
                    if norm > 0:
                        embeddings_np = embeddings_np / norm
                    
                    # メモリ解放
                    del classifier
                    del tensor_wav
                    del embeddings
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    elapsed_time = time.time() - start_time
                    self.logger.info(f"話者埋め込み抽出完了: {elapsed_time:.2f}秒")
                    
                    return embeddings_np
            except Exception as e:
                self.logger.error(f"SpeechBrainモデルの初期化に失敗: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                return self._fallback_speaker_embedding(wav)
        
        except Exception as e:
            self.logger.error(f"話者埋め込み抽出でエラー発生: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return self._fallback_speaker_embedding(wav)
    
    def _fallback_speaker_embedding(self, wav: np.ndarray) -> np.ndarray:
        """
        埋め込み抽出に失敗した場合のフォールバック
        
        Args:
            wav: 音声データ（numpy配列）
            
        Returns:
            ランダムな埋め込みベクトル（192次元）
        """
        self.logger.warning("話者埋め込み抽出に失敗したため、ランダムな埋め込みを使用します")
        
        # 正規化されたランダムベクトルを生成（192次元）
        np.random.seed(int(time.time()))
        random_embedding = np.random.randn(192)
        random_embedding = random_embedding / np.linalg.norm(random_embedding)
        
        return random_embedding
        
    def _ensure_cache_dir(self) -> Path:
        """キャッシュディレクトリを確保する"""
        cache_dir = Path.home() / ".cache" / "speechbrain"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
        
    def _ensure_speechbrain_repo(self, cache_dir: Path) -> Optional[Path]:
        """SpeechBrainリポジトリを確保する"""
        repo_path = cache_dir / "speechbrain"
        
        if repo_path.exists() and (repo_path / ".git").exists():
            self.logger.info(f"既存のSpeechBrainリポジトリを使用: {repo_path}")
            return repo_path
            
        self.logger.info("SpeechBrainリポジトリをクローンしています...")
        try:
            subprocess.run(
                ["git", "clone", "https://github.com/speechbrain/speechbrain.git", str(repo_path)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.logger.info(f"SpeechBrainリポジトリのクローンに成功: {repo_path}")
            return repo_path
        except subprocess.CalledProcessError as e:
            self.logger.error(f"SpeechBrainリポジトリのクローンに失敗: {e}")
            self.logger.error(f"エラー出力: {e.stderr.decode('utf-8')}")
            return None
        except Exception as e:
            self.logger.error(f"SpeechBrainリポジトリのクローン中に例外が発生: {e}")
            return None

    def aggregate_features(self, features_list: List[Dict]) -> Dict:
        """
        複数の音声から抽出した特徴量を集約する
        
        Args:
            features_list: 特徴量の辞書のリスト
            
        Returns:
            集約された特徴量の辞書
        """
        if not features_list:
            self.logger.error("特徴量リストが空です")
            return {}
            
        self.logger.info(f"{len(features_list)}個のサンプルから特徴量を集約します")
        
        try:
            # 1. 話者埋め込みの集約（重み付き平均）
            embeddings = []
            weights = []
            
            for features in features_list:
                if 'speaker_embedding' in features and features['speaker_embedding'] is not None:
                    embeddings.append(features['speaker_embedding'])
                    
                    # F0統計量を重みとして使用（高いF0の特徴量を優先）
                    f0_mean = features.get('f0_stats', {}).get('mean', 0)
                    weight = max(f0_mean, 1.0)  # 最低でも重み1を確保
                    weights.append(weight)
                    
            if not embeddings:
                self.logger.error("有効な話者埋め込みがありません")
                return {}
                
            # 重み付き平均を計算
            weights = np.array(weights) / np.sum(weights)
            embeddings = np.array(embeddings)
            speaker_embedding = np.sum(embeddings * weights[:, np.newaxis], axis=0)
            
            # 正規化
            norm = np.linalg.norm(speaker_embedding)
            if norm > 0:
                speaker_embedding = speaker_embedding / norm
                
            self.logger.info(f"話者埋め込みを集約しました: {len(embeddings)}サンプル")
            
            # 2. F0統計量の集約
            f0_means = []
            f0_stds = []
            
            for features in features_list:
                f0_stats = features.get('f0_stats', {})
                if 'mean' in f0_stats and 'std' in f0_stats:
                    f0_mean = f0_stats['mean']
                    f0_std = f0_stats['std']
                    
                    if f0_mean > 0:  # 有効なF0のみ使用
                        f0_means.append(f0_mean)
                        f0_stds.append(f0_std)
                        
            if not f0_means:
                self.logger.warning("有効なF0統計量がありません。デフォルト値を使用します")
                f0_mean = 110.0  # デフォルト値（男性の平均的なF0）
                f0_std = 20.0    # デフォルト値
            else:
                f0_mean = np.mean(f0_means)
                f0_std = np.mean(f0_stds)
                
            self.logger.info(f"F0統計量を集約しました: 平均={f0_mean:.1f}Hz, 標準偏差={f0_std:.1f}")
            
            # 3. メルスペクトログラムの集約
            mel_specs = []
            
            for features in features_list:
                if 'mel_spec' in features and features['mel_spec'] is not None:
                    mel_specs.append(features['mel_spec'])
                    
            if mel_specs:
                # 各メルスペクトログラムの形状を確認し、平均化するための標準サイズを決定
                mel_freq_bins = mel_specs[0].shape[0]  # 周波数ビン数（通常は80）
                # 各スペクトログラムから平均的な特性を抽出するために、平均を計算
                # 時間軸ではなく周波数軸の特性を見る
                mel_spec_features = np.array([np.mean(spec, axis=1) for spec in mel_specs])
                mel_spec_mean = np.mean(mel_spec_features, axis=0)
                
                # メルスペクトログラムの標準的な長さを設定（例：100フレーム）
                std_length = 100
                mel_spec_template = np.zeros((mel_freq_bins, std_length))
                
                # 平均的な特性からテンプレートを生成
                for i in range(mel_freq_bins):
                    mel_spec_template[i, :] = mel_spec_mean[i]
                
                mel_spec_mean = mel_spec_template
                
                self.logger.info(f"メルスペクトログラムを集約しました: {len(mel_specs)}サンプル")
            else:
                self.logger.warning("有効なメルスペクトログラムがありません。ゼロ行列を使用します")
                mel_spec_mean = np.zeros((80, 100))  # デフォルト値
                
            # 4. F0サンプルの収集（最大1000サンプル）
            f0_samples = []
            max_samples = 1000
            
            for features in features_list:
                if 'f0' in features and len(features['f0']) > 0:
                    f0_samples.extend(list(features['f0']))
                    
                    # サンプル数を制限
                    if len(f0_samples) > max_samples:
                        f0_samples = f0_samples[:max_samples]
                        break
            
            # 5. 集約結果の辞書を構築
            result = {
                'speaker_embedding': speaker_embedding,
                'f0_stats': {
                    'mean': float(f0_mean),
                    'std': float(f0_std)
                },
                'mel_spec_mean': mel_spec_mean,
                'sample_count': len(features_list),
                'created_at': time.time()
            }
            
            if f0_samples:
                result['f0_samples'] = np.array(f0_samples)
                
            self.logger.info("特徴量の集約が完了しました")
            return result
            
        except Exception as e:
            self.logger.error(f"特徴量の集約中にエラーが発生: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {} 