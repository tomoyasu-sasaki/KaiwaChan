import whisper
import sounddevice as sd
import numpy as np
from scipy import signal as sig
from ..utils.logger import Logger

class SpeechRecognizer:
    def __init__(self, config):
        self.model = whisper.load_model("base")
        self.sample_rate = config.config["audio"]["sample_rate"]
        self.duration = config.config["audio"]["duration"]
        self.logger = Logger(config)
        
    def record_audio(self):
        try:
            # オーディオストリームの初期化
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=int(self.sample_rate * self.duration)
            )
            
            with stream:
                self.logger.info("録音を開始します...")
                audio_data, _ = stream.read(int(self.sample_rate * self.duration))
                self.logger.info("録音が完了しました")
                
                # 音声データの前処理
                audio = np.squeeze(audio_data)
                audio = audio / (np.max(np.abs(audio)) + 1e-7)
                
                return audio.reshape(1, -1)
                
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