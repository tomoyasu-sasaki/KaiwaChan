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
            self.logger.info("🎤 音声入力を開始します...")
            audio_chunks = []
            silence_threshold = 0.01
            silence_duration = 0
            max_silence = 2.0  # 2秒の無音で終了
            
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=int(self.sample_rate * 0.1)  # 0.1秒ごとにチェック
            )
            
            with stream:
                while True:
                    audio_chunk, _ = stream.read(int(self.sample_rate * 0.1))
                    audio_chunks.append(audio_chunk)
                    
                    # 音声レベルをチェック
                    if np.max(np.abs(audio_chunk)) < silence_threshold:
                        silence_duration += 0.1
                        if silence_duration >= max_silence:
                            break
                    else:
                        silence_duration = 0
                        
                    # 最大録音時間のチェック
                    if len(audio_chunks) * 0.1 >= self.duration:
                        break
                
            audio = np.concatenate(audio_chunks)
            self.logger.info(f"✅ 音声入力完了（長さ: {len(audio_chunks) * 0.1:.1f}秒）")
            return audio.reshape(1, -1)
                
        except Exception as e:
            self.logger.error(f"録音エラー: {str(e)}")
            raise
        
    def transcribe(self, audio):
        try:
            self.logger.info("🔄 音声認識を実行中...")
            audio_data = audio.squeeze()
            result = self.model.transcribe(
                audio_data,
                language="ja"
            )
            recognized_text = result["text"]
            self.logger.info(f"✅ 音声認識結果: {recognized_text}")
            return recognized_text
            
        except Exception as e:
            self.logger.error(f"音声認識エラー: {str(e)}")
            raise