import os
import sys
from pathlib import Path
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
                           QPushButton, QLabel, QMessageBox, QSizePolicy,
                           QMenuBar, QMenu, QStatusBar, QToolBar, QToolButton, QDialog, QFileDialog, QGroupBox, QInputDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QFont, QIcon, QAction
import tempfile
import pygame
import json
import yaml
import logging
import glob
import requests

from ..core.stt import SpeechRecognizer
from ..core.dialogue import DialogueEngine
from ..core.voice import VoiceCloneManager
from ..core.animation import CharacterAnimator
from ..config import get_settings
from ..utils.logger import Logger
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
    finished = pyqtSignal(str, str)  # (response_text, audio_path)
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
            audio_path = self.voice_clone.synthesize(response)
            
            if not audio_path:
                raise Exception("音声合成に失敗しました")
            
            # 応答テキストと音声ファイルのパスをUI側に渡す
            self.finished.emit(response, audio_path)
            
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


class VoiceModelsSettingsDialog(QDialog):
    """音声モデル変更ダイアログ"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("音声モデル変更")
        self.resize(600, 500)
        self.logger = Logger(get_settings())
        
        # レイアウトの設定
        layout = QVBoxLayout(self)
        
        # 現在の設定表示
        current_settings = QGroupBox("現在の設定")
        current_layout = QVBoxLayout()
        self.current_info = QTextEdit()
        self.current_info.setReadOnly(True)
        self.current_info.setMaximumHeight(100)
        current_layout.addWidget(self.current_info)
        current_settings.setLayout(current_layout)
        
        # エンジン選択
        engine_group = QGroupBox("音声合成エンジン")
        engine_layout = QVBoxLayout()
        
        # ラジオボタンの作成
        self.voicevox_radio = QPushButton("VOICEVOX")
        self.voicevox_radio.setCheckable(True)
        self.voicevox_radio.clicked.connect(lambda: self.switch_engine("voicevox"))
        
        self.parler_radio = QPushButton("Japanese Parler-TTS")
        self.parler_radio.setCheckable(True)
        self.parler_radio.clicked.connect(lambda: self.switch_engine("parler"))
        
        engine_layout.addWidget(self.voicevox_radio)
        engine_layout.addWidget(self.parler_radio)
        engine_group.setLayout(engine_layout)
        
        # VOICEVOX話者一覧
        self.voicevox_group = QGroupBox("VOICEVOX話者一覧")
        voicevox_layout = QVBoxLayout()
        self.speakers_list = QTextEdit()
        self.speakers_list.setReadOnly(True)
        voicevox_layout.addWidget(self.speakers_list)
        self.voicevox_group.setLayout(voicevox_layout)
        
        # Parler-TTS設定
        self.parler_group = QGroupBox("Parler-TTS設定")
        parler_layout = QVBoxLayout()
        
        # 話者の説明入力
        description_label = QLabel("話者の説明:")
        self.description_edit = QTextEdit()
        self.description_edit.setPlaceholderText("例: A female speaker with a slightly high-pitched voice...")
        self.description_edit.setMaximumHeight(100)
        
        parler_layout.addWidget(description_label)
        parler_layout.addWidget(self.description_edit)
        self.parler_group.setLayout(parler_layout)
        
        # ボタン
        button_layout = QHBoxLayout()
        self.select_button = QPushButton("設定を適用")
        self.select_button.clicked.connect(self.apply_settings)
        close_button = QPushButton("閉じる")
        close_button.clicked.connect(self.accept)
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(close_button)
        
        # レイアウトに追加
        layout.addWidget(current_settings)
        layout.addWidget(engine_group)
        layout.addWidget(self.voicevox_group)
        layout.addWidget(self.parler_group)
        layout.addLayout(button_layout)
        
        # 初期状態の設定
        self.load_current_settings()
        self.load_voicevox_speakers()
        
    def load_current_settings(self):
        """現在の設定を読み込んで表示"""
        try:
            with open("config.yml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            models_config = config.get("models", {})
            voicevox_config = models_config.get("voicevox", {})
            parler_config = models_config.get("parler", {})
            
            # 現在のエンジンを取得
            current_engine = models_config.get("current_engine", "voicevox")
            
            # エンジンに応じた情報を表示
            if current_engine == "voicevox":
                current_info = (
                    f"音声合成エンジン: VOICEVOX\n"
                    f"話者ID: {voicevox_config.get('speaker_id', '未設定')}"
                )
                self.voicevox_radio.setChecked(True)
                self.voicevox_group.show()
                self.parler_group.hide()
            else:
                current_info = (
                    f"音声合成エンジン: Japanese Parler-TTS\n"
                    f"話者の説明: {parler_config.get('description', '未設定')}"
                )
                self.parler_radio.setChecked(True)
                self.voicevox_group.hide()
                self.parler_group.show()
                
                # 説明文を設定
                self.description_edit.setText(parler_config.get("description", ""))
            
            self.current_info.setText(current_info)
            
        except Exception as e:
            self.logger.error(f"設定の読み込みに失敗: {str(e)}")
            self.current_info.setText("設定の読み込みに失敗しました。デフォルト設定を使用します。")
            
    def switch_engine(self, engine):
        """エンジンの切り替え"""
        if engine == "voicevox":
            self.voicevox_group.show()
            self.parler_group.hide()
            self.voicevox_radio.setChecked(True)
            self.parler_radio.setChecked(False)
        else:
            self.voicevox_group.hide()
            self.parler_group.show()
            self.voicevox_radio.setChecked(False)
            self.parler_radio.setChecked(True)
            
    def apply_settings(self):
        """設定を適用"""
        try:
            # 設定ファイルの読み込み
            with open("config.yml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            
            if "models" not in config:
                config["models"] = {}
            
            # 現在選択されているエンジンを確認
            current_engine = "voicevox" if self.voicevox_radio.isChecked() else "parler"
            config["models"]["current_engine"] = current_engine
            
            if current_engine == "voicevox":
                # VOICEVOXの設定
                speaker_id, ok = QInputDialog.getInt(
                    self,
                    "話者ID選択",
                    "話者IDを入力してください（上記一覧から選択）:",
                    value=1,
                    min=1,
                    max=999
                )
                
                if ok:
                    if "voicevox" not in config["models"]:
                        config["models"]["voicevox"] = {}
                    config["models"]["voicevox"]["speaker_id"] = speaker_id
                else:
                    return
            else:
                # Parler-TTSの設定
                description = self.description_edit.toPlainText().strip()
                if not description:
                    QMessageBox.warning(
                        self,
                        "入力エラー",
                        "話者の説明を入力してください。"
                    )
                    return
                
                if "parler" not in config["models"]:
                    config["models"]["parler"] = {}
                config["models"]["parler"]["description"] = description
            
            # 設定を保存
            with open("config.yml", "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 表示を更新
            self.load_current_settings()
            
            # 成功メッセージ
            QMessageBox.information(
                self,
                "設定完了",
                "音声設定を更新しました。\n変更を適用するにはアプリケーションの再起動が必要です。"
            )
            
        except Exception as e:
            self.logger.error(f"設定の保存に失敗: {str(e)}")
            QMessageBox.critical(
                self,
                "エラー",
                f"設定の保存に失敗しました: {str(e)}"
            )
            
    def load_voicevox_speakers(self):
        """VOICEVOXの話者一覧を取得して表示"""
        try:
            # VOICEVOXのAPIエンドポイント
            response = requests.get("http://localhost:50021/speakers")
            if response.status_code == 200:
                speakers = response.json()
                
                # 話者情報を整形
                speakers_info = []
                for speaker in speakers:
                    styles = speaker.get('styles', [])
                    for style in styles:
                        speakers_info.append(f"■ {speaker.get('name', '不明')} - {style.get('name', '不明')} (ID: {style.get('id', '不明')})")
                
                if speakers_info:
                    self.speakers_list.setText("\n".join(speakers_info))
                else:
                    self.speakers_list.setText("利用可能な話者が見つかりませんでした")
            else:
                self.speakers_list.setText(f"VOICEVOXサーバーからの応答エラー: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            self.speakers_list.setText("VOICEVOXサーバーに接続できません。\nVOICEVOXが起動しているか確認してください。")
        except Exception as e:
            self.speakers_list.setText(f"話者一覧の取得に失敗しました: {str(e)}")


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
            
            # アニメーションタイマーの設定
            self.animation_timer = QTimer(self)
            self.animation_timer.timeout.connect(self._update_animation)
            self.animation_timer.start(33)  # 約30FPS (1000ms / 30)
            
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
        voice_models_settings = QAction("音声モデル変更", self)
        voice_models_settings.triggered.connect(self._voice_models_settings)
        edit_menu.addAction(voice_models_settings)
        
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
        
    def _handle_response(self, response, audio_path):
        """対話応答の処理とテキスト音声合成"""
        # UIの状態を元に戻す
        self.talk_button.setEnabled(True)
        self.talk_button.setText("話す")
        self.status_label.setText("準備完了")
        
        # テキスト表示
        self.text_display.add_message("AI", response)
        self.logger.info(f"対話生成: {response}")
        
        try:
            # キャラクターの口パクアニメーションを開始
            if hasattr(self, 'character_animator') and self.character_animator:
                self.character_animator.start_talking()
            
            # 音声ファイル再生
            pygame.mixer.init()
            pygame.mixer.music.load(audio_path)
            pygame.mixer.music.play()
            
            # 音声の長さを取得して、その時間後に口パクを停止
            import soundfile as sf
            audio_data, samplerate = sf.read(audio_path)
            duration = len(audio_data) / samplerate * 1000  # ミリ秒に変換
            
            # 音声再生が完了したらタイマーでキャラクターの口パクアニメーションを停止
            def stop_animation():
                if hasattr(self, 'character_animator') and self.character_animator:
                    self.character_animator.stop_talking()
            
            QTimer.singleShot(int(duration), stop_animation)
            
        except Exception as e:
            self.logger.error(f"音声再生エラー: {str(e)}")
            self._show_error("再生エラー", f"音声の再生に失敗しました: {str(e)}")
            
            # エラー時も口パクアニメーションを停止
            if hasattr(self, 'character_animator') and self.character_animator:
                self.character_animator.stop_talking()
    
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
        
    def _voice_models_settings(self):
        """音声モデル変更ダイアログを表示"""
        dialog = VoiceModelsSettingsDialog(self)
        dialog.exec()
        
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
            # アニメーションタイマーの停止
            if hasattr(self, 'animation_timer') and self.animation_timer:
                self.animation_timer.stop()
            
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
            None,  # プロファイル機能を削除したのでNoneを指定
            self.character_animator
        )
        self.audio_thread.finished.connect(lambda response, audio_path: self._handle_response(response, audio_path))
        self.audio_thread.error.connect(self._handle_error)
        self.audio_thread.start()

    def _update_animation(self):
        """アニメーションの定期更新"""
        if hasattr(self, 'character_animator') and self.character_animator:
            try:
                self.character_animator.update()
            except Exception as e:
                self.logger.error(f"アニメーション更新エラー: {e}")
                # 一時的なエラーなら次のフレームで回復する可能性があるのでタイマーは停止しない
