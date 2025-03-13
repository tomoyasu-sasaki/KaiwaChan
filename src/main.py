import sys
from pathlib import Path
import signal
sys.path.append(str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow
from src.core.stt import SpeechRecognizer
from src.core.dialogue import DialogueEngine
from src.core.tts import TTSEngine
from src.core.animation import CharacterAnimator
from src.config import get_settings
from src.utils.logger import Logger

def main():
    # Ctrl+C のシグナルハンドリングを有効化
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    # 設定の読み込み
    config = get_settings()
    
    # ロガーの初期化
    logger = Logger(config)
    logger.info("アプリケーションを起動します")
    
    try:
        # 音声認識エンジンの初期化
        speech_recognizer = SpeechRecognizer(config)
        
        # 対話エンジンの初期化
        dialogue_engine = DialogueEngine(config)
        
        # 音声合成エンジンの初期化
        tts_engine = TTSEngine(config)
        
        # キャラクターアニメーターの初期化（オプション）
        character_animator = CharacterAnimator(config)
        
        # アプリケーションの初期化
        app = QApplication(sys.argv)
        
        # メインウィンドウの作成
        window = MainWindow(
            config=config,
            logger=logger,
            speech_recognizer=speech_recognizer,
            dialogue_engine=dialogue_engine,
            tts_engine=tts_engine,
            character_animator=character_animator
        )
        window.show()
        
        # アプリケーションの実行
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"アプリケーションの初期化に失敗: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 