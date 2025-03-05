import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import os
import sys
import traceback


class LoggerManager:
    """
    ロギング機能を管理するクラス
    アプリケーション全体で一貫したロギングを提供する
    """
    
    # シングルトンインスタンス
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """シングルトンインスタンスを取得する"""
        if cls._instance is None:
            cls._instance = LoggerManager()
        return cls._instance
    
    def __init__(self):
        """コンストラクタ（初期化は setup_logging メソッドで行う）"""
        self.root_logger = logging.getLogger()
        self.logger = logging.getLogger("KaiwaChan")
        self.is_setup = False
        self.log_file = None
        self.console_handler = None
        self.file_handler = None
    
    def setup_logging(self, config=None, log_level=logging.INFO, log_to_console=True, log_to_file=True):
        """
        ロギングをセットアップする
        
        Args:
            config: 設定オブジェクト（Noneの場合はデフォルト設定を使用）
            log_level: ログレベル
            log_to_console: コンソールにログを出力するかどうか
            log_to_file: ファイルにログを出力するかどうか
        """
        if self.is_setup:
            return
        
        # デフォルト設定
        log_dir = Path("logs")
        log_format = '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
        log_date_format = '%Y-%m-%d %H:%M:%S'
        max_log_files = 10
        max_log_size_mb = 5
        
        # 設定を読み込む
        if config:
            # 新しいSettingsManagerの場合
            if hasattr(config, 'get_app_config'):
                log_dir = Path(config.get_app_config('paths', 'logs') or 'logs')
                log_level = config.get_app_config('logging', 'level') or log_level
                log_format = config.get_app_config('logging', 'format') or log_format
                log_date_format = config.get_app_config('logging', 'date_format') or log_date_format
                max_log_files = config.get_app_config('logging', 'max_files') or max_log_files
                max_log_size_mb = config.get_app_config('logging', 'max_size_mb') or max_log_size_mb
            # 古いConfigクラスの場合（後方互換性）
            elif hasattr(config, 'get'):
                log_dir = Path(config.get('paths', 'logs', 'logs'))
                log_level = config.get('logging', 'level', log_level)
                log_format = config.get('logging', 'format', log_format)
                log_date_format = config.get('logging', 'date_format', log_date_format)
                max_log_files = config.get('logging', 'max_files', max_log_files)
                max_log_size_mb = config.get('logging', 'max_size_mb', max_log_size_mb)
        
        # ログディレクトリを作成
        os.makedirs(log_dir, exist_ok=True)
        
        # ログファイルのパス
        today = datetime.now().strftime('%Y-%m-%d')
        self.log_file = log_dir / f"kaiwachan_{today}.log"
        
        # フォーマッタを作成
        formatter = logging.Formatter(log_format, log_date_format)
        
        # ルートロガーを設定
        self.root_logger.setLevel(log_level)
        
        # 既存のハンドラをクリア
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        
        # コンソールハンドラを追加
        if log_to_console:
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setFormatter(formatter)
            self.console_handler.setLevel(log_level)
            self.root_logger.addHandler(self.console_handler)
        
        # ファイルハンドラを追加
        if log_to_file:
            # 通常のファイルハンドラ
            self.file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            self.file_handler.setFormatter(formatter)
            self.file_handler.setLevel(log_level)
            self.root_logger.addHandler(self.file_handler)
            
            # ローテーティングファイルハンドラ
            rotating_handler = logging.handlers.RotatingFileHandler(
                log_dir / "kaiwa_chan.log",
                maxBytes=max_log_size_mb * 1024 * 1024,
                backupCount=max_log_files,
                encoding='utf-8'
            )
            rotating_handler.setFormatter(formatter)
            rotating_handler.setLevel(log_level)
            self.root_logger.addHandler(rotating_handler)
        
        # アプリケーションロガーを設定
        self.logger = logging.getLogger("KaiwaChan")
        self.logger.setLevel(log_level)
        
        # 設定完了
        self.is_setup = True
        self.logger.info(f"ロギングを設定しました: レベル={logging.getLevelName(log_level)}, ファイル={self.log_file}")
    
    def get_logger(self, name):
        """
        指定された名前のロガーを取得する
        
        Args:
            name: ロガー名
            
        Returns:
            Logger: ロガーオブジェクト
        """
        if not self.is_setup:
            self.setup_logging()
            
        return logging.getLogger(name)
    
    def set_level(self, level):
        """
        ログレベルを設定する
        
        Args:
            level: ログレベル
        """
        self.root_logger.setLevel(level)
        self.logger.setLevel(level)
        
        for handler in self.root_logger.handlers:
            handler.setLevel(level)
            
        self.logger.info(f"ログレベルを変更しました: {logging.getLevelName(level)}")
    
    def log_exception(self, exctype, value, tb):
        """
        例外をログに記録する
        
        Args:
            exctype: 例外の型
            value: 例外の値
            tb: トレースバック
        """
        error_message = ''.join(traceback.format_exception(exctype, value, tb))
        self.logger.critical(f"未処理の例外が発生しました:\n{error_message}")


# シングルトンインスタンス
logger_manager = LoggerManager.get_instance()


class Logger:
    """
    ロガーのラッパークラス（後方互換性のため）
    """
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        logger_manager.setup_logging(config)
        self.logger = logger_manager.logger
    
    def info(self, message):
        """情報メッセージをログに記録する"""
        self.logger.info(message)
    
    def error(self, message):
        """エラーメッセージをログに記録する"""
        self.logger.error(message)
    
    def warning(self, message):
        """警告メッセージをログに記録する"""
        self.logger.warning(message)
    
    def debug(self, message):
        """デバッグメッセージをログに記録する"""
        self.logger.debug(message)
    
    def critical(self, message):
        """クリティカルメッセージをログに記録する"""
        self.logger.critical(message)


def setup_logging(config=None):
    """
    ロギングをセットアップする（グローバル関数）
    
    Args:
        config: 設定オブジェクト
    """
    logger_manager.setup_logging(config)
    
    # グローバル例外ハンドラを設定
    sys.excepthook = logger_manager.log_exception
    
    return logger_manager 