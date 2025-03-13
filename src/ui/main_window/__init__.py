from PyQt6.QtWidgets import QMainWindow, QStatusBar
from PyQt6.QtCore import Qt, QTimer

from .ui_handler import UIHandler
from .menu_handler import MenuHandler


class MainWindow(QMainWindow):
    """アプリケーションのメインウィンドウ"""
    
    def __init__(self, config, logger, speech_recognizer, dialogue_engine, tts_engine, character_animator=None):
        """
        メインウィンドウの初期化
        
        Args:
            config: アプリケーション設定
            logger: ロガーインスタンス
            speech_recognizer: 音声認識エンジン
            dialogue_engine: 対話生成エンジン
            tts_engine: 音声合成エンジン
            character_animator: キャラクターアニメーター（オプション）
        """
        super().__init__()
        
        # 依存コンポーネント
        self.config = config
        self.logger = logger
        self.speech_recognizer = speech_recognizer
        self.dialogue_engine = dialogue_engine
        self.tts_engine = tts_engine
        self.character_animator = character_animator
        
        # ウィンドウ設定
        self.setWindowTitle("会話ちゃん")
        self.setMinimumSize(800, 600)
        
        # ステータスバー
        self.status_label = QStatusBar()
        self.status_label.showMessage("準備完了")
        self.setStatusBar(self.status_label)
        
        # ハンドラの初期化
        self.ui_handler = UIHandler(self, config, logger)
        self.menu_handler = MenuHandler(self, config, logger)
        
        # アニメーション更新用タイマー
        if self.character_animator:
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self.update_animation)
            self.animation_timer.start(1000 // 30)  # 30 FPS
    
    def update_animation(self):
        """キャラクターアニメーションの更新"""
        if self.character_animator:
            self.character_animator.update()
        
    def closeEvent(self, event):
        """ウィンドウを閉じる際の処理"""
        # 音声処理スレッドが実行中の場合は停止
        if self.ui_handler.audio_thread and self.ui_handler.audio_thread.isRunning():
            self.ui_handler.audio_thread.terminate()
            self.ui_handler.audio_thread.wait()
            
        # キャラクターアニメーターが存在する場合は停止
        if self.character_animator:
            self.animation_timer.stop()
            self.character_animator.cleanup()
            
        event.accept() 