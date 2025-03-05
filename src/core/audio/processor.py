import numpy as np
import librosa
import soundfile as sf
import logging
import os
from scipy import signal
from pathlib import Path
import tempfile
import io
import warnings
import time
from typing import Optional, Tuple, Union, Dict


class AudioProcessor:
    """
    音声データの読み込み、分析、変換、保存などの処理を行うクラス
    """
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 非推奨警告の抑制
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # デフォルト設定
        self.sample_rate = 22050
        self.temp_dir = Path.home() / ".kaiwa_chan" / "temp"
        
        # 設定から値を取得
        if config:
            self.sample_rate = config.get("audio", "sample_rate", self.sample_rate)
            custom_temp = config.get('audio', 'temp_dir', None)
            if custom_temp:
                self.temp_dir = Path(custom_temp)
        
        # 一時ディレクトリを作成
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def load_audio(self, audio_path: Union[str, Path], 
                   target_sr: Optional[int] = None) -> Tuple[Optional[np.ndarray], int]:
        """
        音声ファイルを読み込む
        
        Args:
            audio_path: 音声ファイルのパス
            target_sr: 目標サンプリングレート（Noneの場合は元のレートを保持）
            
        Returns:
            (音声データ（numpy配列）, サンプリングレート)のタプル、失敗した場合は(None, 0)
        """
        try:
            self.logger.debug(f"音声ファイル読み込み中: {Path(audio_path).name}")
            
            # 既定のサンプリングレートがない場合は設定値を使用
            if target_sr is None:
                target_sr = self.sample_rate
            
            # 音声読み込み
            wav, orig_sr = librosa.load(audio_path, sr=None, mono=True)
            
            # サンプリングレート変換（必要な場合）
            if orig_sr != target_sr:
                wav = librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
                self.logger.debug(f"サンプリングレート変換: {orig_sr}Hz → {target_sr}Hz")
            
            self.logger.debug(f"音声読み込み完了: 長さ {len(wav) / target_sr:.2f}秒")
            return wav, target_sr
            
        except Exception as e:
            self.logger.error(f"音声ファイル読み込みエラー: {e}")
            return None, 0
    
    def save_audio(self, wav: np.ndarray, file_path: Union[str, Path], 
                   sample_rate: Optional[int] = None) -> bool:
        """
        音声データをファイルに保存する
        
        Args:
            wav: 音声データ（numpy配列）
            file_path: 保存先のパス
            sample_rate: サンプリングレート（Noneの場合は設定値を使用）
            
        Returns:
            保存に成功した場合はTrue、失敗した場合はFalse
        """
        try:
            # サンプリングレートが指定されていない場合は設定値を使用
            if sample_rate is None:
                sample_rate = self.sample_rate
            
            # 保存先ディレクトリを作成
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # 音声データを保存
            sf.write(file_path, wav, sample_rate)
            
            self.logger.debug(f"音声ファイル保存完了: {Path(file_path).name}")
            return True
            
        except Exception as e:
            self.logger.error(f"音声ファイル保存エラー: {e}")
            return False
    
    def normalize_volume(self, wav: np.ndarray, target_db: float = -23.0) -> np.ndarray:
        """
        音声のボリュームを正規化する
        
        Args:
            wav: 音声データ（numpy配列）
            target_db: 目標dBFS値
            
        Returns:
            正規化された音声データ
        """
        try:
            # 現在のdBFS値を計算
            if len(wav) == 0:
                return wav
                
            rms = np.sqrt(np.mean(wav**2))
            current_db = 20 * np.log10(rms) if rms > 0 else -100
            
            # ゲイン計算
            gain = 10**((target_db - current_db) / 20)
            
            # クリッピングを防ぐため、正規化後の最大値が1を超える場合はゲインを調整
            if gain * np.max(np.abs(wav)) > 1.0:
                gain = 0.99 / np.max(np.abs(wav))
            
            # ボリューム調整
            return wav * gain
            
        except Exception as e:
            self.logger.error(f"ボリューム正規化エラー: {e}")
            return wav  # エラー時は元の音声を返す
    
    def trim_silence(self, wav: np.ndarray, threshold_db: float = 20.0) -> np.ndarray:
        """
        音声の無音部分をトリミングする
        
        Args:
            wav: 音声データ（numpy配列）
            threshold_db: 無音とみなすdBのしきい値
            
        Returns:
            トリミングされた音声データ
        """
        try:
            # 無音部分のトリミング
            trimmed_wav, _ = librosa.effects.trim(wav, top_db=threshold_db)
            
            # トリミング前後の長さを比較
            before_length = len(wav)
            after_length = len(trimmed_wav)
            
            if before_length > after_length:
                self.logger.debug(f"無音トリミング: {before_length} → {after_length} サンプル " + 
                               f"({(before_length - after_length) / before_length * 100:.1f}% 削減)")
            
            return trimmed_wav
            
        except Exception as e:
            self.logger.error(f"無音トリミングエラー: {e}")
            return wav  # エラー時は元の音声を返す
    
    def split_audio(self, wav: np.ndarray, sr: int, 
                    min_segment_length: float = 0.5,
                    max_segment_length: float = 10.0,
                    silence_threshold_db: float = 30.0) -> list:
        """
        音声を無音部分で分割する
        
        Args:
            wav: 音声データ（numpy配列）
            sr: サンプリングレート
            min_segment_length: 最小セグメント長（秒）
            max_segment_length: 最大セグメント長（秒）
            silence_threshold_db: 無音とみなすdBのしきい値
            
        Returns:
            分割された音声データのリスト
        """
        try:
            # 無音区間の検出
            intervals = librosa.effects.split(wav, top_db=silence_threshold_db)
            
            segments = []
            min_samples = int(min_segment_length * sr)
            max_samples = int(max_segment_length * sr)
            
            current_segment = []
            current_length = 0
            
            for start, end in intervals:
                segment_length = end - start
                
                # 最小セグメント長より短い場合はスキップ
                if segment_length < min_samples:
                    continue
                    
                # 最大セグメント長より長い場合は分割
                if segment_length > max_samples:
                    # max_samples単位で分割
                    for i in range(start, end, max_samples):
                        seg_end = min(i + max_samples, end)
                        seg_wav = wav[i:seg_end]
                        segments.append(seg_wav)
                else:
                    # そのまま追加
                    seg_wav = wav[start:end]
                    segments.append(seg_wav)
            
            self.logger.debug(f"音声分割: {len(segments)}個のセグメントに分割")
            return segments
            
        except Exception as e:
            self.logger.error(f"音声分割エラー: {e}")
            # エラー時は元の音声を1つのセグメントとして返す
            return [wav] if len(wav) > 0 else []
    
    def create_temp_file(self, extension: str = "wav") -> Path:
        """
        一時ファイルのパスを生成する
        
        Args:
            extension: ファイルの拡張子
            
        Returns:
            一時ファイルのPath
        """
        timestamp = int(time.time() * 1000)
        filename = f"temp_{timestamp}.{extension}"
        return self.temp_dir / filename
    
    def clean_temp_files(self, max_age_hours: float = 24.0) -> int:
        """
        古い一時ファイルを削除する
        
        Args:
            max_age_hours: 保持する最大期間（時間）
            
        Returns:
            削除したファイル数
        """
        try:
            count = 0
            max_age_seconds = max_age_hours * 3600
            current_time = time.time()
            
            for file_path in self.temp_dir.glob("temp_*.*"):
                if not file_path.is_file():
                    continue
                    
                # ファイルの最終更新時刻を取得
                file_age = current_time - file_path.stat().st_mtime
                
                # 最大期間を超えた古いファイルを削除
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        count += 1
                    except Exception as e:
                        self.logger.error(f"一時ファイル削除エラー: {file_path} - {e}")
            
            if count > 0:
                self.logger.info(f"{count}個の古い一時ファイルを削除しました")
            
            return count
            
        except Exception as e:
            self.logger.error(f"一時ファイルのクリーンアップエラー: {e}")
            return 0
    
    def compute_mel_spectrogram(self, audio_data, sample_rate=None, n_fft=2048, 
                               hop_length=512, n_mels=80):
        """
        メルスペクトログラムを計算する
        
        Args:
            audio_data: 音声データ
            sample_rate: サンプリングレート
            n_fft: FFTのウィンドウサイズ
            hop_length: ホップ長
            n_mels: メルフィルタバンクの数
            
        Returns:
            np.ndarray: メルスペクトログラム
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data,
                sr=sample_rate,
                n_fft=n_fft,
                hop_length=hop_length,
                n_mels=n_mels
            )
            
            # デシベルスケールに変換
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            self.logger.debug(f"メルスペクトログラムを計算: shape={mel_spec_db.shape}")
            
            return mel_spec_db
            
        except Exception as e:
            self.logger.error(f"メルスペクトログラムの計算に失敗: {e}")
            raise
    
    def extract_pitch(self, audio_data, sample_rate=None, hop_length=512):
        """
        ピッチ（F0）を抽出する
        
        Args:
            audio_data: 音声データ
            sample_rate: サンプリングレート
            hop_length: ホップ長
            
        Returns:
            np.ndarray: ピッチ（F0）値の配列
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        try:
            # ピッチ抽出にはpyinを使用
            f0, voiced_flag, _ = librosa.pyin(
                audio_data,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sample_rate,
                hop_length=hop_length
            )
            
            # 無声部分を0に設定
            f0[~voiced_flag] = 0.0
            
            self.logger.debug(f"ピッチを抽出: 平均F0={np.mean(f0[voiced_flag]):.1f}Hz")
            
            return f0
            
        except Exception as e:
            self.logger.error(f"ピッチ抽出に失敗: {e}")
            raise
    
    def convert_sample_rate(self, audio_data, original_sr, target_sr):
        """
        サンプリングレートを変換する
        
        Args:
            audio_data: 音声データ
            original_sr: 元のサンプリングレート
            target_sr: 変換後のサンプリングレート
            
        Returns:
            np.ndarray: 変換された音声データ
        """
        try:
            # 同じサンプリングレートなら変換不要
            if original_sr == target_sr:
                return audio_data
                
            self.logger.debug(f"サンプリングレートを変換: {original_sr}Hz → {target_sr}Hz")
            
            # リサンプリング
            resampled_audio = librosa.resample(
                audio_data,
                orig_sr=original_sr,
                target_sr=target_sr
            )
            
            self.logger.debug(f"サンプリングレート変換完了: 元の長さ={len(audio_data)/original_sr:.2f}秒, 変換後の長さ={len(resampled_audio)/target_sr:.2f}秒")
            
            return resampled_audio
            
        except Exception as e:
            self.logger.error(f"サンプリングレート変換に失敗: {e}")
            raise
    
    def get_audio_info(self, audio_path):
        """
        音声ファイルの情報を取得する
        
        Args:
            audio_path: 音声ファイルのパス
            
        Returns:
            dict: 音声ファイルの情報
        """
        try:
            info = {}
            
            # librosaでファイル情報を取得
            audio, sr = librosa.load(audio_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            info['path'] = str(audio_path)
            info['sample_rate'] = sr
            info['duration'] = duration
            info['samples'] = len(audio)
            info['channels'] = 1  # librosaはモノラルに変換するため常に1
            
            # さらに詳細情報を取得（可能な場合）
            try:
                with sf.SoundFile(audio_path) as f:
                    info['channels'] = f.channels
                    info['format'] = f.format
                    info['subtype'] = f.subtype
            except:
                pass
                
            self.logger.debug(f"音声ファイル情報: {info}")
            
            return info
            
        except Exception as e:
            self.logger.error(f"音声ファイル情報の取得に失敗: {e}")
            raise
    
    def concatenate_audio(self, audio_list, sample_rates=None, target_sr=None):
        """
        複数の音声データを連結する
        
        Args:
            audio_list: 音声データのリスト
            sample_rates: 各音声データのサンプリングレートのリスト（Noneの場合は全て同じと仮定）
            target_sr: 出力のサンプリングレート（Noneの場合はデフォルト値を使用）
            
        Returns:
            tuple: (連結された音声データ, 出力サンプリングレート)
        """
        if target_sr is None:
            target_sr = self.sample_rate
            
        try:
            if len(audio_list) == 0:
                self.logger.warning("連結する音声データがありません")
                return np.array([]), target_sr
                
            # サンプリングレートの処理
            if sample_rates is None:
                sample_rates = [target_sr] * len(audio_list)
            elif len(sample_rates) != len(audio_list):
                self.logger.warning(f"音声データの数({len(audio_list)})とサンプリングレートの数({len(sample_rates)})が一致しません")
                sample_rates = [sample_rates[0]] * len(audio_list)
                
            # サンプリングレートを統一
            resampled_audio_list = []
            for i, (audio, sr) in enumerate(zip(audio_list, sample_rates)):
                resampled_audio = self.convert_sample_rate(audio, sr, target_sr)
                resampled_audio_list.append(resampled_audio)
                
            # 音声データを連結
            concatenated_audio = np.concatenate(resampled_audio_list)
            
            self.logger.debug(f"音声データを連結: {len(audio_list)}個のデータ, 合計長={len(concatenated_audio)/target_sr:.2f}秒")
            
            return concatenated_audio, target_sr
            
        except Exception as e:
            self.logger.error(f"音声データの連結に失敗: {e}")
            raise
    
    def apply_filters(self, audio_data, sample_rate=None, 
                     high_pass=None, low_pass=None, noise_reduce=False):
        """
        音声データにフィルタを適用する
        
        Args:
            audio_data: 音声データ
            sample_rate: サンプリングレート
            high_pass: ハイパスフィルタのカットオフ周波数（Hz）
            low_pass: ローパスフィルタのカットオフ周波数（Hz）
            noise_reduce: ノイズリダクションを適用するかどうか
            
        Returns:
            np.ndarray: フィルタ適用後の音声データ
        """
        if sample_rate is None:
            sample_rate = self.sample_rate
            
        filtered_audio = audio_data.copy()
        
        try:
            # ハイパスフィルタの適用
            if high_pass is not None and high_pass > 0:
                self.logger.debug(f"ハイパスフィルタを適用: カットオフ周波数={high_pass}Hz")
                filtered_audio = librosa.effects.high_pass_filter(
                    filtered_audio, sr=sample_rate, cutoff=high_pass
                )
                
            # ローパスフィルタの適用
            if low_pass is not None and low_pass < sample_rate / 2:
                self.logger.debug(f"ローパスフィルタを適用: カットオフ周波数={low_pass}Hz")
                filtered_audio = librosa.effects.low_pass_filter(
                    filtered_audio, sr=sample_rate, cutoff=low_pass
                )
                
            # ノイズリダクション（簡易実装）
            if noise_reduce:
                self.logger.debug("ノイズリダクションを適用")
                # 簡易的なスペクトラルサブトラクション
                S = np.abs(librosa.stft(filtered_audio))
                noise_floor = np.mean(S[:, :10], axis=1, keepdims=True) * 1.5
                S_reduced = S - noise_floor
                S_reduced = np.maximum(S_reduced, 0)
                phase = np.angle(librosa.stft(filtered_audio))
                filtered_audio = librosa.istft(S_reduced * np.exp(1j * phase))
                
            return filtered_audio
            
        except Exception as e:
            self.logger.error(f"フィルタの適用に失敗: {e}")
            return audio_data 