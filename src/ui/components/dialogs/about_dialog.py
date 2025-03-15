from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


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
        separator.setStyleSheet("border-bottom: 1px solid #ddd; margin: 5px 0;")
        
        # 技術情報
        tech_label = QLabel(
            "使用技術:\n"
            "- PyQt6: GUIフレームワーク\n"
            "- Whisper: 音声認識\n"
            "- LLM: 対話生成\n"
            "- VOICEVOX: 音声合成"
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