"""
KaiwaChan ユーティリティパッケージ
アプリケーション全体で使用される共通ユーティリティ機能を提供します
"""

# ロガー
from .logger import Logger, LoggerManager, setup_logging

# ファイル操作
from .file_manager import FileManager

# エラー処理
from .error_handler import ErrorHandler

# モデル管理
from .model_downloader import ModelDownloader


# 便利な関数
def get_version():
    """アプリケーションのバージョンを取得する"""
    return "1.0.0"

__all__ = [
    'Logger', 'LoggerManager', 'setup_logging',
    'FileManager',
    'ErrorHandler',
    'ModelDownloader',
    'get_version'
]
