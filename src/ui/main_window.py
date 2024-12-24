from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QMessageBox, QTextEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from ..core.stt import SpeechRecognizer
from ..core.dialogue import DialogueEngine
from ..core.tts import TTSEngine
from ..utils.config import Config
from ..utils.logger import Logger
import signal
import sys
from pathlib import Path

class AudioProcessThread(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, speech_recognizer, dialogue_engine, tts_engine):
        super().__init__()
        self.speech_recognizer = speech_recognizer
        self.dialogue_engine = dialogue_engine
        self.tts_engine = tts_engine

    def run(self):
        try:
            audio = self.speech_recognizer.record_audio()
            text = self.speech_recognizer.transcribe(audio)
            response = self.dialogue_engine.generate_response(text)
            audio_path = self.tts_engine.synthesize(response)
            self.finished.emit(response)
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = Config()
        self.logger = Logger(self.config)
        
        # シグナルハンドラの設定
        signal.signal(signal.SIGINT, self._signal_handler)
        
        self.setWindowTitle("KaiwaChan")
        self.setMinimumSize(800, 600)
        
        try:
            self.speech_recognizer = SpeechRecognizer(self.config)
            self.dialogue_engine = DialogueEngine(self.config)
            self.tts_engine = TTSEngine(self.config)
            self.logger.info("初期化完了")
        except Exception as e:
            self.logger.error(f"初期化エラー: {str(e)}")
            self._show_error("初期化エラー", str(e))
            self.close()
        
        self._setup_ui()
        
    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        
        talk_button = QPushButton("Talk")
        talk_button.clicked.connect(self._handle_talk)
        
        layout.addWidget(self.text_display)
        layout.addWidget(talk_button)
        central_widget.setLayout(layout)
        
    def _handle_talk(self):
        self.thread = AudioProcessThread(
            self.speech_recognizer,
            self.dialogue_engine,
            self.tts_engine
        )
        self.thread.finished.connect(self._handle_response)
        self.thread.error.connect(self._handle_error)
        self.thread.start()
        
    def _handle_response(self, response):
        self.text_display.append(f"Response: {response}")
        self.logger.info(f"��答生成: {response}")
        
    def _handle_error(self, error_msg):
        self.logger.error(f"エラー発生: {error_msg}")
        self._show_error("処理エラー", error_msg)
        
    def _show_error(self, title, message):
        QMessageBox.critical(self, title, message)

    def closeEvent(self, event):
        """ウィンドウが閉じられる時の処理"""
        self.cleanup()
        event.accept()

    def cleanup(self):
        """リソースのクリーンアップ"""
        self.logger.info("システムをシャットダウンしています...")
        
        # 一時ファイルの削除
        temp_audio = Path("temp_audio.wav")
        if temp_audio.exists():
            temp_audio.unlink()
        
        # スレッドの終了待機
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        
        self.logger.info("シャットダウン完了")

    def _signal_handler(self, signum, frame):
        """Ctrl+C シグナルハンドラ"""
        self.logger.info("Ctrl+C が検出されました")
        self.close()
        QApplication.quit()