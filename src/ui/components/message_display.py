from PyQt6.QtWidgets import QTextEdit, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QColor, QTextCursor, QTextCharFormat, QBrush

class MessageDisplay(QTextEdit):
    """
    会話メッセージを表示するためのカスタムテキスト表示コンポーネント
    
    特徴:
    - ユーザーとAIのメッセージを視覚的に区別
    - メッセージのタイムスタンプ表示
    - 自動スクロール
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)  # 編集不可
        self.setFont(QFont("Arial", 10))
        self.setMinimumHeight(300)
        
        # 背景色とスタイル設定
        self.setStyleSheet("""
            QTextEdit {
                background-color: #f8f8f8;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
            }
        """)
        
        # 初期メッセージ
        self.setPlaceholderText("会話が始まるのを待っています...")
        
        # メッセージスタイル
        self.user_format = QTextCharFormat()
        self.user_format.setForeground(QBrush(QColor("#0066CC")))  # ユーザーメッセージの色
        self.user_format.setFontWeight(QFont.Weight.Bold)
        
        self.ai_format = QTextCharFormat()
        self.ai_format.setForeground(QBrush(QColor("#009933")))  # AIメッセージの色
        self.ai_format.setFontWeight(QFont.Weight.Bold)
        
        self.text_format = QTextCharFormat()
        self.text_format.setForeground(QBrush(QColor("#333333")))  # 通常テキストの色
        
        # 最初のメッセージ
        self.document().clear()
        self.add_system_message("システムを起動しました。「話す」ボタンをクリックして会話を始めてください。")
    
    def add_message(self, sender, text):
        """メッセージを追加"""
        # 送信者表示
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # 既存のテキストがある場合は改行を追加
        if not self.document().isEmpty():
            cursor.insertBlock()
            cursor.insertBlock()
        
        # 送信者に応じたフォーマット適用
        if sender.lower() == "user":
            cursor.insertText("あなた: ", self.user_format)
        elif sender.lower() == "ai":
            cursor.insertText("AI: ", self.ai_format)
        else:
            cursor.insertText(f"{sender}: ", self.ai_format)
        
        # メッセージ本文
        cursor.insertText(text, self.text_format)
        
        # 自動スクロール
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    def add_system_message(self, text):
        """システムメッセージを追加"""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        
        # 既存のテキストがある場合は改行を追加
        if not self.document().isEmpty():
            cursor.insertBlock()
        
        # システムメッセージのフォーマット
        system_format = QTextCharFormat()
        system_format.setForeground(QBrush(QColor("#999999")))  # グレー
        system_format.setFontItalic(True)
        
        # メッセージ挿入
        cursor.insertText(f"System: {text}", system_format)
        
        # 自動スクロール
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    def clear_messages(self):
        """すべてのメッセージをクリア"""
        self.clear()
        self.add_system_message("会話履歴をクリアしました。") 