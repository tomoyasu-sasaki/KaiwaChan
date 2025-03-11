import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
from ..tts import TTSEngine

class VoiceCloneManager:
    """音声合成を管理するクラス"""
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # TTS設定（デフォルト値）
        self.use_japanese_tts = True
        self.initialize_tts_on_demand = True
        if config:
            self.use_japanese_tts = config.get('voice_clone', 'use_japanese_tts', self.use_japanese_tts)
            self.initialize_tts_on_demand = config.get('voice_clone', 'initialize_tts_on_demand', self.initialize_tts_on_demand)
        
        # TTSエンジンは必要に応じて初期化
        self._tts_engine = None
        self._tts_initialized = False
        
        # 遅延初期化しない場合はすぐにTTSエンジンを初期化
        if not self.initialize_tts_on_demand:
            self._initialize_tts()
        
        self.logger.info("VoiceCloneManagerを初期化しました")
    
    def _initialize_tts(self) -> bool:
        """
        TTS（音声合成）エンジンを初期化する
        
        Returns:
            初期化に成功した場合はTrue、失敗した場合はFalse
        """
        if self._tts_initialized:
            return True
            
        try:
            self.logger.info("TTSエンジンを初期化しています...")
            self._tts_engine = TTSEngine(self.config)
            self._tts_initialized = True
            self.logger.info("TTSエンジンの初期化が完了しました")
            return True
        except Exception as e:
            self.logger.error(f"TTSエンジンの初期化に失敗しました: {e}")
            return False
    
    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> Optional[str]:
        """
        テキストから音声を合成する
        
        Args:
            text: 合成するテキスト
            speaker_id: 話者ID（Noneの場合はデフォルト）
            
        Returns:
            生成された音声ファイルのパス、失敗した場合はNone
        """
        if not self._tts_initialized and not self._initialize_tts():
            return None
            
        try:
            return self._tts_engine.synthesize(text, speaker_id)
        except Exception as e:
            self.logger.error(f"音声合成に失敗: {e}")
            return None
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        システム情報を取得する
        
        Returns:
            システム情報の辞書
        """
        info = {
            'tts_initialized': self._tts_initialized,
            'use_japanese_tts': self.use_japanese_tts,
        }
        
        if self._tts_initialized and self._tts_engine:
            info.update(self._tts_engine.get_system_info())
            
        return info 