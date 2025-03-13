from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTextEdit,
                           QPushButton, QLabel, QMessageBox, QGroupBox, QInputDialog)
from PyQt6.QtCore import Qt
import yaml
import requests
from ....utils.logger import Logger
from ....config import get_settings


class VoiceModelsSettingsDialog(QDialog):
    """音声モデル変更ダイアログ"""
    def __init__(self, parent=None, config=None):
        """
        音声モデル設定ダイアログの初期化
        
        Args:
            parent: 親ウィンドウ
            config: アプリケーション設定
        """
        super().__init__(parent)
        self.setWindowTitle("音声モデル変更")
        self.resize(600, 500)
        self.logger = Logger(get_settings())
        self.settings = get_settings()
        
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
            models_config = self.settings.get_app_config("models")
            if not models_config:
                self.current_info.setText("設定が読み込めません。デフォルト設定を使用します。")
                return
                
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
            self.current_info.setText(f"設定の読み込みに失敗しました: {str(e)}")
            
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
            # 現在選択されているエンジンを確認
            current_engine = "voicevox" if self.voicevox_radio.isChecked() else "parler"
            
            # エンジンの設定を更新
            self.settings.set_app_config("models", "current_engine", current_engine)
            
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
                    self.settings.set_app_config("models", "voicevox", {
                        "speaker_id": speaker_id,
                        "url": "http://localhost:50021",
                        "cache_enabled": True,
                        "cache_size": 100,
                        "max_retries": 3,
                        "retry_delay": 1.0,
                        "timeout": 30
                    })
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
                
                self.settings.set_app_config("models", "parler", {
                    "description": description
                })
            
            # 設定を保存
            self.settings.save_all()
            
            # 表示を更新
            self.load_current_settings()
            
            # 成功メッセージ
            QMessageBox.information(
                self,
                "設定完了",
                "音声設定を更新しました。\n変更を適用するにはアプリケーションの再起動が必要です。"
            )
            
        except Exception as e:
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