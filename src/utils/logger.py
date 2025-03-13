import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import os
import sys
import traceback
from types import TracebackType
from typing import Optional, Any, Dict, Union, Type, Callable
from dataclasses import dataclass
from enum import Enum, auto


class LogLevel(Enum):
    """ログレベルを定義する列挙型"""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


@dataclass
class LogConfig:
    """ロギング設定を保持するデータクラス"""
    log_dir: Path = Path("logs")
    log_level: LogLevel = LogLevel.INFO
    log_format: str = '%(asctime)s [%(levelname)s] [%(name)s] %(message)s'
    date_format: str = '%Y-%m-%d %H:%M:%S'
    max_files: int = 10
    max_size_mb: int = 5
    log_to_console: bool = True
    log_to_file: bool = True
    app_name: str = "KaiwaChan"


class LoggerManager:
    """
    ロギング機能を管理するクラス
    
    アプリケーション全体で一貫したロギングを提供し、以下の機能をサポート:
    - コンソールおよびファイルへのログ出力
    - ログローテーション
    - 柔軟な設定管理
    - グローバル例外ハンドリング
    
    Attributes:
        is_setup (bool): ロギングが設定済みかどうか
        log_file (Optional[Path]): 現在のログファイルパス
        config (LogConfig): 現在のロギング設定
    """
    
    _instance: Optional['LoggerManager'] = None
    
    def __init__(self) -> None:
        """シングルトンインスタンスの初期化"""
        self.root_logger: logging.Logger = logging.getLogger()
        self.logger: logging.Logger = logging.getLogger("KaiwaChan")
        self.is_setup: bool = False
        self.log_file: Optional[Path] = None
        self.console_handler: Optional[logging.StreamHandler] = None
        self.file_handler: Optional[logging.FileHandler] = None
        self.config: LogConfig = LogConfig()
    
    @classmethod
    def get_instance(cls) -> 'LoggerManager':
        """
        シングルトンインスタンスを取得する
        
        Returns:
            LoggerManager: シングルトンインスタンス
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _create_log_directory(self) -> None:
        """ログディレクトリを作成する"""
        try:
            os.makedirs(self.config.log_dir, exist_ok=True)
        except OSError as e:
            sys.stderr.write(f"ログディレクトリの作成に失敗しました: {e}\n")
            raise
    
    def _setup_handlers(self, formatter: logging.Formatter) -> None:
        """
        ログハンドラを設定する
        
        Args:
            formatter: ログフォーマッタ
        """
        # 既存のハンドラをクリア
        for handler in self.root_logger.handlers[:]:
            self.root_logger.removeHandler(handler)
        
        # コンソールハンドラを設定
        if self.config.log_to_console:
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setFormatter(formatter)
            self.console_handler.setLevel(self.config.log_level.value)
            self.root_logger.addHandler(self.console_handler)
        
        # ファイルハンドラを設定
        if self.config.log_to_file:
            # 日付別ログファイル
            today = datetime.now().strftime('%Y-%m-%d')
            self.log_file = self.config.log_dir / f"{self.config.app_name.lower()}_{today}.log"
            
            self.file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
            self.file_handler.setFormatter(formatter)
            self.file_handler.setLevel(self.config.log_level.value)
            self.root_logger.addHandler(self.file_handler)
            
            # ローテーティングログファイル
            rotating_handler = logging.handlers.RotatingFileHandler(
                self.config.log_dir / f"{self.config.app_name.lower()}.log",
                maxBytes=self.config.max_size_mb * 1024 * 1024,
                backupCount=self.config.max_files,
                encoding='utf-8'
            )
            rotating_handler.setFormatter(formatter)
            rotating_handler.setLevel(self.config.log_level.value)
            self.root_logger.addHandler(rotating_handler)
    
    def setup_logging(self, config: Optional[Union[LogConfig, Dict[str, Any]]] = None) -> None:
        """
        ロギングをセットアップする
        
        Args:
            config: ロギング設定（LogConfigオブジェクトまたは設定辞書）
        
        Raises:
            ValueError: 無効な設定値が指定された場合
        """
        if self.is_setup:
            return
            
        # 設定を更新
        if isinstance(config, dict):
            # 辞書から設定を更新
            for key, value in config.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        elif isinstance(config, LogConfig):
            self.config = config
            
        try:
            # ログディレクトリを作成
            self._create_log_directory()
            
            # ルートロガーを設定
            self.root_logger.setLevel(self.config.log_level.value)
            
            # フォーマッタを作成
            formatter = logging.Formatter(
                self.config.log_format,
                self.config.date_format
            )
            
            # ハンドラを設定
            self._setup_handlers(formatter)
            
            # アプリケーションロガーを設定
            self.logger = logging.getLogger(self.config.app_name)
            self.logger.setLevel(self.config.log_level.value)
            
            # 設定完了
            self.is_setup = True
            self.logger.info(
                f"ロギングを設定しました: "
                f"レベル={self.config.log_level.name}, "
                f"ファイル={self.log_file}"
            )
            
        except Exception as e:
            sys.stderr.write(f"ロギングの設定に失敗しました: {e}\n")
            raise
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        指定された名前のロガーを取得する
        
        Args:
            name: ロガー名
            
        Returns:
            logging.Logger: ロガーオブジェクト
        """
        if not self.is_setup:
            self.setup_logging()
        return logging.getLogger(name)
    
    def set_level(self, level: Union[LogLevel, int]) -> None:
        """
        ログレベルを設定する
        
        Args:
            level: 新しいログレベル（LogLevelまたはint）
        """
        if isinstance(level, LogLevel):
            level = level.value
            
        self.root_logger.setLevel(level)
        self.logger.setLevel(level)
        
        for handler in self.root_logger.handlers:
            handler.setLevel(level)
            
        self.logger.info(f"ログレベルを変更しました: {logging.getLevelName(level)}")
    
    def log_exception(
        self,
        exctype: Type[BaseException],
        value: BaseException,
        tb: Optional[TracebackType]
    ) -> None:
        """
        例外をログに記録する
        
        Args:
            exctype: 例外の型
            value: 例外の値
            tb: トレースバック
        """
        error_message = ''.join(traceback.format_exception(exctype, value, tb))
        self.logger.critical(f"未処理の例外が発生しました:\n{error_message}")


class Logger:
    """
    ロガーのラッパークラス
    
    LoggerManagerの機能を簡単に利用するためのインターフェースを提供します。
    """
    
    def __init__(self, config: Optional[Union[LogConfig, Dict[str, Any]]] = None) -> None:
        """
        コンストラクタ
        
        Args:
            config: ロギング設定
        """
        self._manager = LoggerManager.get_instance()
        self._manager.setup_logging(config)
        self._logger = self._manager.logger
    
    def log(self, level: LogLevel, message: str) -> None:
        """
        指定されたレベルでメッセージをログに記録する
        
        Args:
            level: ログレベル
            message: ログメッセージ
        """
        self._logger.log(level.value, message)
    
    def debug(self, message: str) -> None:
        """デバッグメッセージをログに記録する"""
        self.log(LogLevel.DEBUG, message)
    
    def info(self, message: str) -> None:
        """情報メッセージをログに記録する"""
        self.log(LogLevel.INFO, message)
    
    def warning(self, message: str) -> None:
        """警告メッセージをログに記録する"""
        self.log(LogLevel.WARNING, message)
    
    def error(self, message: str) -> None:
        """エラーメッセージをログに記録する"""
        self.log(LogLevel.ERROR, message)
    
    def critical(self, message: str) -> None:
        """クリティカルメッセージをログに記録する"""
        self.log(LogLevel.CRITICAL, message)


def setup_logging(
    config: Optional[Union[LogConfig, Dict[str, Any]]] = None
) -> LoggerManager:
    """
    ロギングをセットアップする（グローバル関数）
    
    Args:
        config: ロギング設定
        
    Returns:
        LoggerManager: ロガーマネージャーインスタンス
    """
    manager = LoggerManager.get_instance()
    manager.setup_logging(config)
    
    # グローバル例外ハンドラを設定
    sys.excepthook = manager.log_exception
    
    return manager 