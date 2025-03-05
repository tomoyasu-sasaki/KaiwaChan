import requests
import json
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import logging
import tempfile
import os
from typing import Optional, Dict, Union
import sounddevice as sd
import uuid

class TTSEngine:
    """
    テキスト音声合成（TTS）を管理するクラス
    
    テキストから音声データを生成する機能を提供します。
    VOICEVOXをバックエンドとして使用し、テキストから自然な音声を生成します。
    """
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # デフォルト設定
        self.base_url = "http://localhost:50021"
        self.speaker_id = 2
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30
        self.sample_rate = 24000  # VOICEVOXのデフォルトサンプルレート
        self.cache_size = 100
        self.cache_enabled = True
        
        # 設定から値を取得
        if config:
            voicevox_config = config.get("models", "voicevox", {})
            self.base_url = voicevox_config.get("url", self.base_url)
            self.speaker_id = config.get("character", "voice_id", self.speaker_id)
            self.max_retries = voicevox_config.get("max_retries", self.max_retries)
            self.retry_delay = voicevox_config.get("retry_delay", self.retry_delay)
            self.timeout = voicevox_config.get("timeout", self.timeout)
            self.cache_size = voicevox_config.get("cache_size", self.cache_size)
            self.cache_enabled = voicevox_config.get("cache_enabled", self.cache_enabled)
        
        # キャッシュの初期化
        self.cache_dir = Path(tempfile.gettempdir()) / "kaiwachan" / "tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        
    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> Optional[str]:
        """
        テキストから音声を合成する
        
        Args:
            text: 合成するテキスト
            speaker_id: 話者ID（指定しない場合はデフォルト値を使用）
            
        Returns:
            str: 生成された音声ファイルのパス、失敗時はNone
        """
        if not text or text.isspace():
            self.logger.warning("空のテキストが渡されました。音声合成をスキップします")
            return None
            
        # 話者IDの設定
        current_speaker = speaker_id if speaker_id is not None else self.speaker_id
        
        # キャッシュチェック
        if self.cache_enabled:
            cache_key = f"{text}_{current_speaker}"
            if cache_key in self.cache and os.path.exists(self.cache[cache_key]):
                self.logger.info("✅ キャッシュから音声を取得しました")
                return self.cache[cache_key]
        
        self.logger.info(f"🔊 音声合成を開始: {text[:30]}{'...' if len(text) > 30 else ''}")
        
        for attempt in range(self.max_retries):
            try:
                # 音声合成クエリを作成
                self.logger.debug(f"音声合成クエリを作成中... (試行 {attempt + 1}/{self.max_retries})")
                params = {'text': text, 'speaker': current_speaker}
                
                query = requests.post(
                    f'{self.base_url}/audio_query',
                    params=params,
                    timeout=self.timeout
                )
                query.raise_for_status()
                query_data = query.json()
                
                # カスタムパラメータを設定（例：話速、音高など）
                if self.config:
                    voice_params = self.config.get("character", "voice_params", {})
                    if voice_params:
                        for param, value in voice_params.items():
                            if param in query_data:
                                query_data[param] = value
                
                # 音声を生成
                self.logger.debug("音声を生成中...")
                synthesis = requests.post(
                    f'{self.base_url}/synthesis',
                    headers={'Content-Type': 'application/json'},
                    params={'speaker': current_speaker},
                    data=json.dumps(query_data),
                    timeout=self.timeout
                )
                synthesis.raise_for_status()
                
                # 一意のファイル名を生成
                file_name = f"tts_{uuid.uuid4().hex}.wav"
                audio_path = self.cache_dir / file_name
                
                # 音声データをファイルに保存
                with open(audio_path, "wb") as f:
                    f.write(synthesis.content)
                
                # キャッシュに追加
                if self.cache_enabled:
                    if len(self.cache) >= self.cache_size:
                        # 最も古いキャッシュエントリを削除
                        oldest_key = next(iter(self.cache))
                        oldest_file = self.cache.pop(oldest_key)
                        if os.path.exists(oldest_file):
                            try:
                                os.remove(oldest_file)
                            except Exception as e:
                                self.logger.warning(f"古いキャッシュファイルの削除に失敗: {e}")
                    
                    self.cache[cache_key] = str(audio_path)
                
                self.logger.info(f"✅ 音声合成完了: {audio_path}")
                return str(audio_path)
                
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"⚠️ 試行 {attempt + 1}/{self.max_retries} 失敗: {str(e)}")
                if attempt == self.max_retries - 1:
                    self.logger.error(f"❌ 音声合成に失敗しました: {str(e)}")
                    return None
                time.sleep(self.retry_delay)
            except Exception as e:
                self.logger.error(f"❌ 音声合成中に予期しないエラーが発生: {str(e)}")
                return None
    
    def synthesize_to_array(self, text: str, speaker_id: Optional[int] = None) -> Optional[np.ndarray]:
        """
        テキストから音声を合成し、numpy配列として返す
        
        Args:
            text: 合成するテキスト
            speaker_id: 話者ID（指定しない場合はデフォルト値を使用）
            
        Returns:
            np.ndarray: 生成された音声データ、失敗時はNone
        """
        audio_path = self.synthesize(text, speaker_id)
        if not audio_path:
            return None
            
        try:
            data, _ = sf.read(audio_path)
            return data
        except Exception as e:
            self.logger.error(f"❌ 音声ファイルの読み込みに失敗: {str(e)}")
            return None
    
    def play_audio(self, audio_data_or_path: Union[np.ndarray, str]) -> bool:
        """
        音声を再生する
        
        Args:
            audio_data_or_path: 再生する音声データまたは音声ファイルのパス
            
        Returns:
            bool: 再生に成功したかどうか
        """
        try:
            self.logger.info("🔊 音声を再生中...")
            
            # 入力が文字列（ファイルパス）の場合、ファイルから読み込む
            if isinstance(audio_data_or_path, str):
                data, samplerate = sf.read(audio_data_or_path)
            else:
                # numpy配列の場合はそのまま使用
                data = audio_data_or_path
                samplerate = self.sample_rate
            
            # 音声再生
            sd.play(data, samplerate)
            sd.wait()  # 再生完了まで待機
            
            self.logger.info("✅ 音声再生完了")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ 音声再生エラー: {str(e)}")
            return False
    
    def get_available_speakers(self) -> Dict:
        """
        利用可能な話者の一覧を取得する
        
        Returns:
            Dict: 話者情報の辞書（ID, 名前、スタイルなど）
        """
        try:
            response = requests.get(
                f'{self.base_url}/speakers',
                timeout=self.timeout
            )
            response.raise_for_status()
            
            speakers_data = response.json()
            speakers = {}
            
            for speaker in speakers_data:
                speaker_id = speaker.get("speaker_id")
                name = speaker.get("name")
                styles = speaker.get("styles", [])
                
                speakers[speaker_id] = {
                    "name": name,
                    "styles": [{"id": style.get("id"), "name": style.get("name")} for style in styles]
                }
            
            self.logger.info(f"✅ {len(speakers)} 人の話者情報を取得")
            return speakers
            
        except Exception as e:
            self.logger.error(f"❌ 話者情報の取得に失敗: {str(e)}")
            return {}
    
    def check_engine_status(self) -> bool:
        """
        TTSエンジンの状態を確認する
        
        Returns:
            bool: エンジンが利用可能かどうか
        """
        try:
            response = requests.get(
                f'{self.base_url}/version',
                timeout=5
            )
            response.raise_for_status()
            version = response.json().get("version")
            self.logger.info(f"✅ VOICEVOXエンジン利用可能 (バージョン: {version})")
            return True
        except Exception as e:
            self.logger.error(f"❌ VOICEVOXエンジンに接続できません: {str(e)}")
            return False 