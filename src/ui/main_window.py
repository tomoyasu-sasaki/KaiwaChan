import os
import sys
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                           QPushButton, QLabel, QComboBox, QMessageBox, QSizePolicy,
                           QMenuBar, QMenu, QStatusBar, QToolBar, QToolButton, QDialog, QFileDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QFont, QIcon, QAction
import tempfile
import pygame
import json
import logging
import glob

from ..core.stt import SpeechRecognizer
from ..core.dialogue import DialogueEngine
from ..core.voice import VoiceCloneManager
from ..core.animation import CharacterAnimator
from ..config import get_settings
from ..utils.logger import Logger
from .voice_profile_dialog import VoiceProfileDialog
from .components.message_display import MessageDisplay


class AudioProcessThread(QThread):
    """
    音声処理を行うスレッド
    
    会話の流れ：
    1. 音声入力の取得
    2. 音声認識（STT）
    3. 対話生成
    4. 音声合成（TTS）または音声変換
    5. キャラクターアニメーション
    """
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, speech_recognizer, dialogue_engine, voice_clone, profile_name, character_animator=None):
        super().__init__()
        self.speech_recognizer = speech_recognizer
        self.dialogue_engine = dialogue_engine
        self.voice_clone = voice_clone
        self.profile_name = profile_name
        self.character_animator = character_animator
        
    def run(self):
        """スレッドの実行：音声入力から応答生成までの一連の処理"""
        try:
            # 音声入力と認識
            audio = self.speech_recognizer.record_audio()
            text = self.speech_recognizer.transcribe(audio)
            
            # アニメーションを表示している場合は口パク開始
            if self.character_animator:
                self.character_animator.start_talking()
                
            # 応答生成
            response = self.dialogue_engine.generate_response(text)
            
            # 音声合成
            if self.profile_name:
                # プロファイルがある場合はボイスクローンで変換
                audio_data = self.voice_clone.synthesize_speech(response, self.profile_name)
                if audio_data is not None:
                    # 音声データを一時ファイルに保存（_handle_responseで再生できるように）
                    import soundfile as sf
                    temp_file = "temp_audio.wav"
                    sf.write(temp_file, audio_data, 22050)  # 正しいサンプリングレート（22050Hz）に修正
                    print(f"音声ファイルを保存: {temp_file}")
                else:
                    # ボイスクローン失敗時は通常の音声合成にフォールバック
                    audio_path = self.voice_clone.synthesize(response)
            else:
                # プロファイルがない場合は通常の音声合成を使用
                audio_path = self.voice_clone.synthesize(response)
            
            # 応答テキストをUI側に渡す
            self.finished.emit(response)
            
        except Exception as e:
            import traceback
            error_msg = f"エラー発生: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class AboutDialog(QDialog):
    """アプリケーション情報ダイアログ"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("会話ちゃんについて")
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # アプリケーション情報
        title_label = QLabel("会話ちゃん")
        title_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        version_label = QLabel("バージョン 1.0.0")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        desc_label = QLabel("ローカル環境で動作する音声対話可能なAIキャラクターチャットシステム")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 区切り線
        separator = QLabel("")
        separator.setStyleSheet("border-bottom: 1px solid #ddd; margin: 10px 0;")
        
        # 技術情報
        tech_label = QLabel(
            "使用技術:\n"
            "- PyQt6: GUIフレームワーク\n"
            "- Whisper: 音声認識\n"
            "- LLM: 対話生成\n"
            "- VOICEVOX: 音声合成\n"
            "- SpeechBrain: 話者認識"
        )
        tech_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # 閉じるボタン
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(self.accept)
        
        # レイアウトに追加
        layout.addWidget(title_label)
        layout.addWidget(version_label)
        layout.addWidget(desc_label)
        layout.addWidget(separator)
        layout.addWidget(tech_label)
        layout.addStretch()
        layout.addWidget(close_button)


class MainWindow(QMainWindow):
    """
    アプリケーションのメインウィンドウ
    
    ユーザーインターフェースと対話プロセスを管理する
    """
    def __init__(self):
        super().__init__()
        
        # 基本設定
        self.config = get_settings()
        self.logger = Logger(self.config)
        self.setWindowTitle("会話ちゃん")
        self.resize(800, 600)
        
        # コア機能の初期化
        self.initialize_core_components()
        
        # UIの設定 - 初期化順序を変更
        self.setup_status_bar()  # ステータスバーを先に初期化
        self.setup_ui()
        self.setup_menu()
        
        self.logger.info("アプリケーションが起動しました")
        
    def initialize_core_components(self):
        """コア機能コンポーネントの初期化"""
        try:
            # 音声認識エンジンの初期化
            self.speech_recognizer = SpeechRecognizer(self.config)
            
            # 対話エンジンの初期化
            self.dialogue_engine = DialogueEngine(self.config)
            
            # 音声変換エンジンの初期化
            self.voice_clone = VoiceCloneManager(self.config)
            
            # キャラクターアニメーションの初期化
            self.character_animator = CharacterAnimator(self.config)
            
            # オーディオ処理スレッド
            self.audio_thread = None
            
            self.logger.info("コア機能の初期化が完了しました")
        except Exception as e:
            self.logger.error(f"コア機能の初期化に失敗: {str(e)}")
            self._show_error("初期化エラー", f"アプリケーションの初期化に失敗しました: {str(e)}")
            sys.exit(1)
        
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
        
        # プロファイル選択
        profile_layout = QHBoxLayout()
        profile_label = QLabel("音声プロファイル:")
        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(150)
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        
        profile_button = QPushButton("プロファイル管理")
        profile_button.clicked.connect(self._show_profile_dialog)
        
        profile_layout.addWidget(profile_label)
        profile_layout.addWidget(self.profile_combo)
        profile_layout.addWidget(profile_button)
        
        header_layout.addLayout(profile_layout)
        main_layout.addLayout(header_layout)
        
        # テキスト表示エリア
        self.text_display = MessageDisplay()
        main_layout.addWidget(self.text_display)
        
        # 操作ボタン
        button_layout = QHBoxLayout()
        
        self.talk_button = QPushButton("話す")
        self.talk_button.setMinimumHeight(50)
        self.talk_button.setFont(QFont("Arial", 12))
        self.talk_button.clicked.connect(self._handle_talk)
        button_layout.addWidget(self.talk_button)
        
        # クリアボタン
        clear_button = QPushButton("履歴クリア")
        clear_button.clicked.connect(self._clear_conversation)
        button_layout.addWidget(clear_button)
        
        main_layout.addLayout(button_layout)
        
        # メインウィジェットを設定
        self.setCentralWidget(main_widget)
        
        # プロファイルリストの更新
        self._update_profile_list()
        
    def setup_menu(self):
        """メニューバーの設定"""
        menubar = self.menuBar()
        
        # ファイルメニュー
        file_menu = menubar.addMenu("ファイル")
        
        # エクスポート
        export_action = QAction("会話履歴を保存", self)
        export_action.triggered.connect(self._export_conversation)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # 終了
        exit_action = QAction("終了", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 編集メニュー
        edit_menu = menubar.addMenu("編集")
        
        # 会話履歴クリア
        clear_action = QAction("会話履歴クリア", self)
        clear_action.triggered.connect(self._clear_conversation)
        edit_menu.addAction(clear_action)
        
        # 設定メニュー
        settings_menu = menubar.addMenu("設定")
        
        # 音声プロファイル
        profile_action = QAction("音声プロファイル管理", self)
        profile_action.triggered.connect(self._show_profile_dialog)
        settings_menu.addAction(profile_action)
        
        # ヘルプメニュー
        help_menu = menubar.addMenu("ヘルプ")
        
        # アプリケーション情報
        about_action = QAction("会話ちゃんについて", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
        
    def setup_status_bar(self):
        """ステータスバーの設定"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 状態表示
        self.status_label = QLabel("準備完了")
        self.status_bar.addPermanentWidget(self.status_label)
        
    def _show_profile_dialog(self):
        """音声プロファイル管理ダイアログを表示"""
        dialog = VoiceProfileDialog(self.voice_clone, self)
        if dialog.exec():
            self._update_profile_list()
        
    def _update_profile_list(self):
        """プロファイルリストを更新"""
        self.profile_combo.clear()
        self.profile_combo.addItem("デフォルト")
        
        for profile_name in self.voice_clone.voice_profiles.keys():
            self.profile_combo.addItem(profile_name)
        
    def _on_profile_changed(self, profile_name):
        """プロファイル変更時の処理"""
        if profile_name and profile_name != "デフォルト":
            self.logger.info(f"音声プロファイル変更: {profile_name}")
            # 属性の存在確認を追加
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"プロファイル: {profile_name}")
        else:
            # 属性の存在確認を追加
            if hasattr(self, 'status_label'):
                self.status_label.setText("プロファイル: デフォルト")
        
    def _handle_talk(self):
        """会話ボタンクリック時の処理"""
        # UIの状態を変更
        self.talk_button.setEnabled(False)
        self.talk_button.setText("聞いています...")
        self.status_label.setText("音声入力中...")
        
        # 入力開始メッセージを表示
        self.text_display.add_system_message("音声入力を待機しています...")
        
        # スレッドの初期化と開始
        self.initialize_thread()
        
    def _handle_response(self, response):
        """対話応答の処理とテキスト音声合成"""
        # UIの状態を元に戻す
        self.talk_button.setEnabled(True)
        self.talk_button.setText("話す")
        self.status_label.setText("準備完了")
        
        # テキスト表示
        self.text_display.add_message("AI", response)
        self.logger.info(f"対話生成: {response}")
        
        try:
            # 音声ファイル再生
            self._play_latest_audio()
            
            # 音声再生が完了したらタイマーでキャラクターの口パクアニメーションを停止
            def stop_animation():
                # 口パクアニメーションの停止
                if hasattr(self, 'character_animator') and self.character_animator:
                    self.character_animator.stop_talking()
            
            # 音声の長さに基づいて口パクを止めるタイマーを設定（仮に5秒）
            QTimer.singleShot(5000, stop_animation)
            
        except Exception as e:
            self.logger.error(f"音声再生エラー: {str(e)}")
            self._show_error("再生エラー", f"音声の再生に失敗しました: {str(e)}")
            
            # エラー時も口パクアニメーションを停止
            if hasattr(self, 'character_animator') and self.character_animator:
                self.character_animator.stop_talking()
    
    def _play_latest_audio(self):
        """最新の音声ファイルを再生"""
        # 一般的なパターンとしてtemp_audio.wavかtemp_audio_tts.wavが使われる
        audio_paths = [
            Path("temp_audio.wav"), 
            Path("temp_audio_tts.wav"),
            # キャッシュディレクトリから最新のファイルを探す
            Path(tempfile.gettempdir()) / "kaiwachan" / "tts_cache" / "*.wav"
        ]
        audio_path = None
        
        # スレッドで生成された音声ファイルを優先して探す
        for path in audio_paths:
            if path.is_file():
                audio_path = str(path)
                break
            elif '*' in str(path):  # グロブパターンの場合
                files = sorted(glob.glob(str(path)), key=os.path.getctime, reverse=True)
                if files:
                    audio_path = files[0]
                    break
        
        if not audio_path:
            self.logger.warning("再生する音声ファイルが見つかりません")
            return
        
        self.logger.info(f"音声再生: {audio_path}")
        
        # キャラクターの口パクアニメーションを開始
        if hasattr(self, 'character_animator') and self.character_animator:
            self.character_animator.start_talking()
            
        # 音声再生
        if hasattr(self, 'audio_player'):
            self.audio_player.play_file(audio_path, blocking=False)
        else:
            # 既存のコードでの再生方法
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
    
    def _handle_error(self, error_msg):
        """エラー処理"""
        self.talk_button.setEnabled(True)
        self.talk_button.setText("話す")
        self.status_label.setText("エラーが発生しました")
        
        self.logger.error(error_msg)
        self._show_error("処理エラー", "音声処理中にエラーが発生しました。詳細はログを確認してください。")
        
    def _show_error(self, title, message):
        """エラーダイアログ表示"""
        QMessageBox.critical(self, title, message)
        
    def _clear_conversation(self):
        """会話履歴をクリア"""
        self.text_display.clear_messages()
        
    def _export_conversation(self):
        """会話履歴をファイルにエクスポート"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "会話履歴を保存", "", "テキストファイル (*.txt);;すべてのファイル (*)"
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.text_display.toPlainText())
                self.status_label.setText(f"会話履歴を保存しました: {file_path}")
                self.logger.info(f"会話履歴を保存: {file_path}")
            except Exception as e:
                self.logger.error(f"ファイル保存エラー: {str(e)}")
                self._show_error("保存エラー", f"ファイルの保存に失敗しました: {str(e)}")
    
    def _show_about(self):
        """アプリケーション情報ダイアログを表示"""
        dialog = AboutDialog(self)
        dialog.exec()
                
    def closeEvent(self, event):
        """アプリケーション終了時の処理"""
        self.cleanup()
        event.accept()
        
    def cleanup(self):
        """リソースのクリーンアップ"""
        try:
            # スレッドの停止
            if self.audio_thread and self.audio_thread.isRunning():
                self.audio_thread.terminate()
                self.audio_thread.wait()
                
            # キャラクターアニメーターのクリーンアップ
            if hasattr(self, 'character_animator') and self.character_animator:
                self.character_animator.cleanup()
                
            self.logger.info("アプリケーションのリソースをクリーンアップしました")
        except Exception as e:
            self.logger.error(f"クリーンアップエラー: {str(e)}")
        
    def initialize_thread(self):
        """音声処理スレッドの初期化"""
        if self.audio_thread and self.audio_thread.isRunning():
            self.audio_thread.terminate()
            self.audio_thread.wait()
            
        self.audio_thread = AudioProcessThread(
            self.speech_recognizer,
            self.dialogue_engine,
            self.voice_clone,
            self.profile_combo.currentText() if self.profile_combo.currentText() != "デフォルト" else None,
            self.character_animator
        )
        self.audio_thread.finished.connect(self._handle_response)
        self.audio_thread.error.connect(self._handle_error)
        self.audio_thread.start()
