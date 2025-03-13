import whisper
import numpy as np
import sounddevice as sd
import logging
import time
from typing import Tuple, Optional, Dict, Union
import warnings
import torch

class SpeechRecognizer:
    """
    音声認識（STT）を管理するクラス
    
    音声データからテキストへの変換機能を提供します。
    WhisperモデルをベースにしたSTT機能を実装しています。
    """
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 警告の抑制
        self._suppress_warnings()
        
        # デフォルト設定
        self.model_size = "base"
        self.sample_rate = 16000
        self.max_duration = 30
        self.language = "ja"
        self.device = "cpu"
        
        # 設定から値を取得
        if config:
            self.model_size = config.get("stt", "model", self.model_size)
            self.sample_rate = config.get("audio", "sample_rate", self.sample_rate)
            self.max_duration = config.get("audio", "max_duration", self.max_duration)
            self.language = config.get("stt", "language", self.language)
            self.device = config.get("stt", "device", self.device)
        
        # モデルの遅延ロード
        self._model = None
        
    def _suppress_warnings(self):
        """警告とログの抑制"""
        # Whisperの警告を抑制
        logging.getLogger("whisper").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)
        
        # PyTorchの警告を抑制
        logging.getLogger("torch").setLevel(logging.ERROR)
        torch.set_warn_always(False)
        
    def load_model(self) -> bool:
        """
        Whisperモデルをロードする
        
        Returns:
            bool: モデルのロードに成功したかどうか
        """
        if self._model is not None:
            return True
            
        try:
            self.logger.info(f"Whisperモデル ({self.model_size}) をロード中...")
            start_time = time.time()
            self._model = whisper.load_model(self.model_size, device=self.device)
            load_time = time.time() - start_time
            self.logger.info(f"Whisperモデルをロードしました (所要時間: {load_time:.2f}秒)")
            return True
        except Exception as e:
            self.logger.error(f"Whisperモデルのロードに失敗: {e}")
            return False
    
    def record_audio(self, max_duration=None, silence_threshold=0.01, silence_time=2.0) -> Optional[np.ndarray]:
        """
        マイクから音声を録音する
        
        Args:
            max_duration: 最大録音時間（秒）、Noneの場合は設定値を使用
            silence_threshold: 無音判定の閾値
            silence_time: 無音判定する時間（秒）
            
        Returns:
            np.ndarray: 録音された音声データ、失敗時はNone
        """
        if max_duration is None:
            max_duration = self.max_duration
        
        # 設定ファイルからsilence_thresholdを取得（設定されていれば）
        config_threshold = self.config.get("audio", "silence_threshold", None)
        if config_threshold is not None:
            silence_threshold = float(config_threshold)
            self.logger.debug(f"設定ファイルから閾値を読み込みました: silence_threshold={silence_threshold}")
            
        try:
            self.logger.info("🎤 音声入力を開始します...")
            audio_chunks = []
            
            # 無音フレームのカウント用
            silent_frames = 0
            chunk_duration = 0.1  # 1チャンク当たりの秒数
            required_silent_frames = int(silence_time / chunk_duration)  # 無音判定に必要なフレーム数
            
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=int(self.sample_rate * chunk_duration)  # 0.1秒ごとにチェック
            )
            
            with stream:
                start_time = time.time()
                self.logger.debug("音声ストリームを開始しました") 
                
                has_detected_voice = False  # 有声検出フラグ
                
                while True:
                    audio_chunk, _ = stream.read(int(self.sample_rate * chunk_duration))
                    audio_chunks.append(audio_chunk)
                    
                    # 音声レベルをチェック
                    current_level = np.max(np.abs(audio_chunk))
                    if current_level < silence_threshold:
                        silent_frames += 1
                        if silent_frames >= required_silent_frames:
                            if has_detected_voice:  # 有声を検出した後の無音のみ終了条件とする
                                self.logger.debug(f"無音を {silence_time}秒 検出したため録音を終了します")
                                break
                    else:
                        # 音声が検出された場合
                        if not has_detected_voice and current_level >= silence_threshold * 2:  # より強い音声の場合は有声と判定
                            has_detected_voice = True
                            self.logger.debug(f"有声を検出しました（レベル: {current_level:.4f}）")
                        
                        # 無音カウントのリセット
                        silent_frames = 0
                        
                    # 最大録音時間のチェック
                    elapsed_time = time.time() - start_time
                    if elapsed_time >= max_duration:
                        self.logger.debug(f"最大録音時間 {max_duration}秒 に達したため録音を終了します")
                        break
            
            if not audio_chunks:
                self.logger.warning("録音データが空です")
                return None
                
            audio = np.concatenate(audio_chunks)
            recording_length = len(audio) / self.sample_rate
            self.logger.info(f"✅ 音声入力完了（長さ: {recording_length:.1f}秒）")
            
            # ノーマライズ
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                
            return audio.reshape(1, -1)
                
        except Exception as e:
            self.logger.error(f"録音エラー: {e}")
            return None
    
    def transcribe_file(self, audio_file: str) -> Optional[str]:
        """
        音声ファイルからテキストに変換する
        
        Args:
            audio_file: 音声ファイルのパス
            
        Returns:
            str: 認識されたテキスト、失敗時はNone
        """
        try:
            if not self.load_model():
                return None
                
            self.logger.info(f"音声ファイルを処理中: {audio_file}")
            
            result = self._model.transcribe(
                audio_file,
                language=self.language,
                fp16=False
            )
            
            recognized_text = result["text"].strip()
            self.logger.info(f"✅ 音声認識結果: {recognized_text}")
            return recognized_text
            
        except Exception as e:
            self.logger.error(f"音声ファイル認識エラー: {e}")
            return None
    
    def transcribe(self, audio: np.ndarray) -> Optional[str]:
        """
        音声データからテキストに変換する
        
        Args:
            audio: 音声データ（numpy配列）
            
        Returns:
            str: 認識されたテキスト、失敗時はNone
        """
        try:
            if not self.load_model():
                return None
                
            self.logger.info("🔄 音声認識を実行中...")
            
            # 音声データの形状を調整
            audio_data = audio.squeeze()
            
            # 音声認識を実行
            result = self._model.transcribe(
                audio_data,
                language=self.language,
                fp16=False
            )
            
            recognized_text = result["text"].strip()
            self.logger.info(f"✅ 音声認識結果: {recognized_text}")
            return recognized_text
            
        except Exception as e:
            self.logger.error(f"音声認識エラー: {e}")
            return None
    
    def transcribe_with_timestamps(self, audio: Union[np.ndarray, str]) -> Optional[Dict]:
        """
        音声データまたはファイルからテキストと時間情報を取得する
        
        Args:
            audio: 音声データ（numpy配列）または音声ファイルパス
            
        Returns:
            Dict: 認識結果（テキストとセグメント情報）、失敗時はNone
        """
        try:
            if not self.load_model():
                return None
                
            self.logger.info("🔄 タイムスタンプ付き音声認識を実行中...")
            
            # 音声認識を実行
            result = self._model.transcribe(
                audio,
                language=self.language,
                word_timestamps=True,
                fp16=False
            )
            
            self.logger.info(f"✅ タイムスタンプ付き音声認識完了: {len(result['segments'])}セグメント")
            return result
            
        except Exception as e:
            self.logger.error(f"タイムスタンプ付き音声認識エラー: {e}")
            return None
    
    def get_available_models(self) -> list:
        """
        利用可能なWhisperモデルの一覧を返す
        
        Returns:
            list: モデル名のリスト
        """
        return ["tiny", "base", "small", "medium", "large"] 