from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QProgressBar
from PyQt6.QtCore import Qt, pyqtSignal

class DetailedProgressWindow(QDialog):
    """詳細な進捗表示ウィンドウ"""
    canceled = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("処理の進捗")
        self.setMinimumSize(400, 200)
        
        layout = QVBoxLayout()
        
        # 現在の処理内容
        self.current_task = QLabel("準備中...")
        layout.addWidget(self.current_task)
        
        # 全体の進捗バー
        self.total_progress = QProgressBar()
        layout.addWidget(self.total_progress)
        
        # 現在のファイル処理の進捗バー
        self.file_progress = QProgressBar()
        layout.addWidget(self.file_progress)
        
        # ログ表示エリア
        self.log_display = QLabel()
        self.log_display.setWordWrap(True)
        layout.addWidget(self.log_display)
        
        self.setLayout(layout)
        
    def update_progress(self, current, total, message):
        """進捗の更新"""
        self.current_task.setText(message)
        self.total_progress.setValue(int(current / total * 100))
        
    def add_log(self, message):
        """ログメッセージの追加"""
        current_text = self.log_display.text()
        self.log_display.setText(f"{message}\n{current_text}") 