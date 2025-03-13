from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QLabel, QMessageBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
import pygame
import soundfile as sf

from ..components.message_display import MessageDisplay
from ..components.audio_process import AudioProcessThread


class UIHandler:
    """メインウィンドウのUI処理を管理するクラス"""
    
    def __init__(self, parent, config, logger):
        """
        UIハンドラの初期化
        
        Args:
            parent: 親ウィンドウ（MainWindow）
            config: アプリケーション設定
            logger: ロガーインスタンス
        """
        self.parent = parent
        self.config = config
        self.logger = logger
        
        # UIコンポーネント
        self.text_display = None
        self.talk_button = None
        self.status_label = None
        
        # 音声処理スレッド
        self.audio_thread = None
        
        # UIの初期化
        self.setup_ui()
        
    def setup_ui(self):
        """UIコンポーネントの設定と配置"""
        # メインウィジェットとレイアウト
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # ヘッダー部分
        header_layout = QHBoxLayout()
        
        # タイトルラベル
        title_label = QLabel("会話ちゃん")
        title_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        header_layout.addWidget(title_label)
        
        header_layout.addStretch()
        main_layout.addLayout(header_layout)
        
        # テキスト表示エリア
        self.text_display = MessageDisplay()
        main_layout.addWidget(self.text_display)
        
        # 操作ボタン
        button_layout = QHBoxLayout()
        
        self.talk_button = QPushButton("話す")
        self.talk_button.setMinimumHeight(50)
        self.talk_button.setFont(QFont("Arial", 12))
        self.talk_button.clicked.connect(self.handle_talk)
        button_layout.addWidget(self.talk_button)
        
        # クリアボタン
        clear_button = QPushButton("履歴クリア")
        clear_button.clicked.connect(self.clear_conversation)
        button_layout.addWidget(clear_button)
        
        main_layout.addLayout(button_layout)
        
        # メインウィジェットを設定
        self.parent.setCentralWidget(main_widget)
        
    def handle_talk(self):
        """会話ボタンクリック時の処理"""
        # UIの状態を変更
        self.talk_button.setEnabled(False)
        self.talk_button.setText("聞いています...")
        self.parent.statusBar().showMessage("音声入力中...")
        
        # 入力開始メッセージを表示
        self.text_display.add_system_message("音声入力を待機しています...")
        
        # スレッドの初期化と開始
        self.initialize_thread()
        
    def handle_response(self, response, audio_path):
        """対話応答の処理とテキスト音声合成"""
        # UIの状態を元に戻す
        self.talk_button.setEnabled(True)
        self.talk_button.setText("話す")
        self.parent.statusBar().showMessage("準備完了")
        
        # テキスト表示
        self.text_display.add_message("AI", response)
        self.logger.info(f"対話生成: {response}")
        
        try:
            # キャラクターの口パクアニメーションを開始
            if self.parent.character_animator:
                self.parent.character_animator.start_talking()
            
            # 音声ファイル再生
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # 音声の長さを取得して、その時間後に口パクを停止
            audio_data, samplerate = sf.read(audio_path)
            duration = len(audio_data) / samplerate * 1000  # ミリ秒に変換
            
            # 音声再生が完了したらタイマーでキャラクターの口パクアニメーションを停止
            def stop_animation():
                if self.parent.character_animator:
                    self.parent.character_animator.stop_talking()
            
            QTimer.singleShot(int(duration), stop_animation)
            
        except Exception as e:
            self.logger.error(f"音声再生エラー: {str(e)}")
            self.show_error("再生エラー", f"音声の再生に失敗しました: {str(e)}")
            
            # エラー時も口パクアニメーションを停止
            if self.parent.character_animator:
                self.parent.character_animator.stop_talking()
    
    def handle_error(self, error_msg):
        """エラー処理"""
        self.talk_button.setEnabled(True)
        self.talk_button.setText("話す")
        self.parent.statusBar().showMessage("エラーが発生しました")
        
        self.logger.error(error_msg)
        self.show_error("処理エラー", "音声処理中にエラーが発生しました。詳細はログを確認してください。")
        
    def show_error(self, title, message):
        """エラーダイアログ表示"""
        QMessageBox.critical(self.parent, title, message)
        
    def clear_conversation(self):
        """会話履歴をクリア"""
        self.text_display.clear_messages()
        
    def initialize_thread(self):
        """音声処理スレッドの初期化"""
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.terminate()
            self.audio_thread.wait()
            
        self.audio_thread = AudioProcessThread(
            self.parent.speech_recognizer,
            self.parent.dialogue_engine,
            self.parent.tts_engine,
            self.parent.character_animator
        )
        self.audio_thread.finished.connect(self.handle_response)
        self.audio_thread.error.connect(self.handle_error)
        self.audio_thread.start() 