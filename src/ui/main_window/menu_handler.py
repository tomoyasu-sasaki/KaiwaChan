from PyQt6.QtWidgets import QMenuBar, QMenu, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QAction

from ..components.dialogs.about_dialog import AboutDialog
from ..components.dialogs.voice_models_dialog import VoiceModelsSettingsDialog


class MenuHandler:
    """メインウィンドウのメニュー処理を管理するクラス"""
    
    def __init__(self, parent, config, logger):
        """
        メニューハンドラの初期化
        
        Args:
            parent: 親ウィンドウ（MainWindow）
            config: アプリケーション設定
            logger: ロガーインスタンス
        """
        self.parent = parent
        self.config = config
        self.logger = logger
        
        # メニューバーの初期化
        self.setup_menubar()
        
    def setup_menubar(self):
        """メニューバーの設定"""
        menubar = QMenuBar(self.parent)
        
        # ファイルメニュー
        file_menu = QMenu("ファイル(&F)", self.parent)
        exit_action = QAction("終了(&Q)", self.parent)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.parent.close)
        file_menu.addAction(exit_action)
        menubar.addMenu(file_menu)
        
        # 設定メニュー
        settings_menu = QMenu("設定(&S)", self.parent)
        voice_settings_action = QAction("音声モデル設定(&V)", self.parent)
        voice_settings_action.triggered.connect(self.show_voice_models_settings)
        settings_menu.addAction(voice_settings_action)
        menubar.addMenu(settings_menu)
        
        # ヘルプメニュー
        help_menu = QMenu("ヘルプ(&H)", self.parent)
        about_action = QAction("バージョン情報(&A)", self.parent)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)
        menubar.addMenu(help_menu)
        
        self.parent.setMenuBar(menubar)
        
    def show_voice_models_settings(self):
        """音声モデル設定ダイアログを表示"""
        try:
            dialog = VoiceModelsSettingsDialog(self.parent, self.config)
            dialog.exec()
        except Exception as e:
            self.logger.error(f"音声モデル設定ダイアログ表示エラー: {str(e)}")
            QMessageBox.critical(
                self.parent,
                "エラー",
                "音声モデル設定ダイアログの表示に失敗しました。詳細はログを確認してください。"
            )
            
    def show_about_dialog(self):
        """バージョン情報ダイアログを表示"""
        try:
            dialog = AboutDialog(self.parent)
            dialog.exec()
        except Exception as e:
            self.logger.error(f"バージョン情報ダイアログ表示エラー: {str(e)}")
            QMessageBox.critical(
                self.parent,
                "エラー",
                "バージョン情報ダイアログの表示に失敗しました。詳細はログを確認してください。"
            ) 