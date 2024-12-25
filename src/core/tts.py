import requests
import json
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import requests.exceptions
from ..utils.logger import Logger
import sounddevice as sd

class TTSEngine:
    def __init__(self, config):
        self.config = config
        self.base_url = "http://localhost:50021"
        self.speaker_id = 1
        self.max_retries = 2    # リトライ回数を減らす
        self.retry_delay = 0.5  # 遅延を短縮
        self.logger = Logger(config)
        
        # キャッシュの初期化
        self.cache_dir = Path("cache/tts")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        
        self.sample_rate = 24000  # VOICEVOXのデフォルトサンプルレート
        
    def synthesize(self, text):
        # キャッシュチェック
        cache_key = f"{text}_{self.speaker_id}"
        if cache_key in self.cache:
            self.logger.info("✅ キャッシュから音声を取得")
            return self.cache[cache_key]
            
        self.logger.info(f"🔊 音声合成を開始: {text}")
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info("🔄 音声合成クエリを作成中...")
                params = {'text': text, 'speaker': self.speaker_id}
                query = requests.post(
                    f'{self.base_url}/audio_query',
                    params=params,
                    timeout=10
                )
                query.raise_for_status()
                
                self.logger.info("🔄 音声を生成中...")
                synthesis = requests.post(
                    f'{self.base_url}/synthesis',
                    headers={'Content-Type': 'application/json'},
                    params={'speaker': self.speaker_id},
                    data=json.dumps(query.json()),
                    timeout=30
                )
                synthesis.raise_for_status()
                
                audio_path = Path("temp_audio.wav")
                with open(audio_path, "wb") as f:
                    f.write(synthesis.content)
                
                # 音声を再生
                self._play_audio(audio_path)
                
                self.logger.info(f"✅ 音声合成・再生完了: {audio_path}")
                return str(audio_path)
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"⚠️ 試行 {attempt + 1}/{self.max_retries} 失敗: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error("❌ 音声合成に失敗しました")
                    raise Exception(f"VOICEVOX Engine Error: {str(e)}")
                time.sleep(self.retry_delay) 

    def _play_audio(self, audio_path):
        """音声ファイルを再生"""
        try:
            self.logger.info("🔊 音声を再生中...")
            data, samplerate = sf.read(audio_path)
            sd.play(data, samplerate)
            sd.wait()  # 再生完了まで待機
        except Exception as e:
            self.logger.error(f"音声再生エラー: {str(e)}")
            raise 