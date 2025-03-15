"""
音声合成（TTS: Text-To-Speech）サブパッケージ
テキストから音声データへの変換機能を提供します
"""

from .tts_engine import TTSEngine

__all__ = [
    'TTSEngine'
]
