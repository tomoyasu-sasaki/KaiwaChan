import requests
import json
import soundfile as sf
import numpy as np
from pathlib import Path

class TTSEngine:
    def __init__(self, config):
        self.config = config
        self.base_url = "http://localhost:50021"
        self.speaker_id = 1  # デフォルトスピーカー
        
    def synthesize(self, text):
        try:
            # 音声合成用のクエリを作成
            params = {'text': text, 'speaker': self.speaker_id}
            query = requests.post(f'{self.base_url}/audio_query', params=params)
            query.raise_for_status()
            
            # 音声合成
            synthesis = requests.post(
                f'{self.base_url}/synthesis',
                headers={'Content-Type': 'application/json'},
                params={'speaker': self.speaker_id},
                data=json.dumps(query.json())
            )
            synthesis.raise_for_status()
            
            # 音声データを一時ファイルとして保存
            audio_path = Path("temp_audio.wav")
            with open(audio_path, "wb") as f:
                f.write(synthesis.content)
            
            return str(audio_path)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"VOICEVOX Engine Error: {str(e)}") 