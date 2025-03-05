"""
KaiwaChanアプリケーションのコア機能モジュール
音声認識、対話生成、音声合成、声質変換などの機能を提供します
"""

# 各モジュールはサブディレクトリからインポートされます
# - audio: 音声処理関連
# - stt: 音声認識関連
# - tts: 音声合成関連
# - dialogue: 対話生成関連
# - voice: 音声合成・変換関連
# - animation: キャラクターアニメーション関連

# サブパッケージのインポート
from .stt import SpeechRecognizer
from .tts import TTSEngine
from .dialogue import DialogueEngine
from .voice import VoiceCloneManager
from .animation import CharacterAnimator, SpriteManager

# 公開するクラスやモジュールのリスト
__all__ = [
    'SpeechRecognizer',  # 音声認識
    'TTSEngine',         # 音声合成
    'DialogueEngine',    # 対話生成
    'VoiceCloneManager', # 声質変換
    'CharacterAnimator', # キャラクターアニメーション
    'SpriteManager'      # スプライト管理
]
