import whisper
import sounddevice as sd
import numpy as np
from scipy import signal as sig

class SpeechRecognizer:
    def __init__(self, config):
        self.model = whisper.load_model("base")
        self.sample_rate = config.config["audio"]["sample_rate"]
        self.duration = config.config["audio"]["duration"]
        
    def record_audio(self):
        try:
            # バッファサイズを明示的に計算
            num_samples = int(self.duration * self.sample_rate)
            
            # 録音
            recording = sd.rec(
                num_samples,
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocking=True  # 録音完了まで待機
            )
            
            # 音声データの前処理
            audio = np.squeeze(recording)  # 余分な次元を削除
            
            # パディングを追加して201サンプルにする
            if len(audio) < 201:
                padding = np.zeros(201 - len(audio))
                audio = np.concatenate([audio, padding])
            else:
                # リサンプリングして201サンプルにする
                audio = sig.resample(audio, 201)
            
            # 正規化
            audio = audio / (np.max(np.abs(audio)) + 1e-7)
            
            return audio.reshape(1, -1)  # [1, 201]の形状に整形
            
        except Exception as e:
            self.logger.error(f"録音エラー: {str(e)}")
            raise
        
    def transcribe(self, audio):
        try:
            # 音声データの形状を確認
            if audio.shape != (1, 201):
                raise ValueError(f"Invalid audio shape: {audio.shape}, expected: (1, 201)")
            
            result = self.model.transcribe(audio)
            return result["text"]
            
        except Exception as e:
            self.logger.error(f"音声認識エラー: {str(e)}")
            raise