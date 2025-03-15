import logging
import traceback
import sys
import time
from types import TracebackType
from functools import wraps
from typing import Optional, Any, Callable, TypeVar, Type, Union, Tuple, Dict
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


T = TypeVar('T')
F = TypeVar('F', bound=Callable[..., Any])


class ErrorLevel(Enum):
    """エラーレベルを定義する列挙型"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


@dataclass
class ErrorConfig:
    """エラー処理の設定を保持するデータクラス"""
    max_retries: int = 3
    retry_delay: float = 1.0
    log_traceback: bool = True
    raise_on_error: bool = True
    error_log_path: Optional[Path] = None


class ErrorHandlerException(Exception):
    """ErrorHandlerの基本例外クラス"""
    pass


class RetryExhaustedException(ErrorHandlerException):
    """リトライ回数が上限に達した場合の例外"""
    pass


class ErrorHandler:
    """
    エラー処理を管理するユーティリティクラス
    
    以下の機能を提供:
    - 構造化されたエラーハンドリング
    - 自動リトライ機能
    - エラーログ記録
    - UIへのエラー通知
    - グローバル例外ハンドリング
    
    Attributes:
        logger (logging.Logger): ロガーインスタンス
        config (ErrorConfig): エラー処理の設定
    """
    
    _instance: Optional['ErrorHandler'] = None
    
    def __init__(self, logger: Optional[logging.Logger] = None, config: Optional[ErrorConfig] = None) -> None:
        """
        コンストラクタ
        
        Args:
            logger: ロガーインスタンス（Noneの場合は新規作成）
            config: エラー処理の設定
        """
        self.logger = logger or logging.getLogger(__name__)
        self.config = config or ErrorConfig()
    
    @classmethod
    def get_instance(cls) -> 'ErrorHandler':
        """
        シングルトンインスタンスを取得する
        
        Returns:
            ErrorHandler: シングルトンインスタンス
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _format_error(self, error: Exception, message: Optional[str] = None) -> str:
        """
        エラーメッセージをフォーマットする
        
        Args:
            error: 例外オブジェクト
            message: 追加のエラーメッセージ
            
        Returns:
            str: フォーマットされたエラーメッセージ
        """
        error_type = type(error).__name__
        error_message = str(error)
        base_message = message or "エラーが発生しました"
        
        return f"{base_message}: [{error_type}] {error_message}"
    
    def _log_error(
        self,
        error: Exception,
        level: ErrorLevel,
        message: Optional[str] = None
    ) -> None:
        """
        エラーをログに記録する
        
        Args:
            error: 例外オブジェクト
            level: エラーレベル
            message: 追加のエラーメッセージ
        """
        error_message = self._format_error(error, message)
        
        if self.config.log_traceback:
            error_message += f"\n{traceback.format_exc()}"
        
        log_method = getattr(self.logger, level.name.lower())
        log_method(error_message)
        
        if self.config.error_log_path:
            try:
                with open(self.config.error_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{error_message}\n")
            except IOError as e:
                self.logger.error(f"エラーログの書き込みに失敗しました: {e}")
    
    def handle_error(
        self,
        error: Exception,
        message: Optional[str] = None,
        level: ErrorLevel = ErrorLevel.ERROR,
        raise_error: Optional[bool] = None
    ) -> str:
        """
        例外を処理する
        
        Args:
            error: 例外オブジェクト
            message: エラーメッセージ
            level: エラーレベル
            raise_error: 例外を再発生させるかどうか
            
        Returns:
            str: エラーメッセージ
            
        Raises:
            Exception: raise_errorがTrueの場合、元の例外を再発生
        """
        error_message = self._format_error(error, message)
        self._log_error(error, level, message)
        
        should_raise = raise_error if raise_error is not None else self.config.raise_on_error
        if should_raise:
            raise error
        
        return error_message
    
    def silent_error(
        self,
        default_value: Any = None,
        error_level: ErrorLevel = ErrorLevel.ERROR
    ) -> Callable[[F], F]:
        """
        エラーを無視するデコレータ
        
        Args:
            default_value: エラー発生時に返す値
            error_level: エラーログのレベル
            
        Returns:
            Callable: 関数デコレータ
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self._log_error(
                        e,
                        error_level,
                        f"Silent error in {func.__name__}"
                    )
                    return default_value
            return wrapper  # type: ignore
        return decorator
    
    def retry(
        self,
        max_attempts: Optional[int] = None,
        retry_delay: Optional[float] = None,
        exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = Exception,
        error_level: ErrorLevel = ErrorLevel.WARNING
    ) -> Callable[[F], F]:
        """
        リトライするデコレータ
        
        Args:
            max_attempts: 最大試行回数
            retry_delay: リトライ前の待機時間（秒）
            exceptions: キャッチする例外のタプル
            error_level: エラーログのレベル
            
        Returns:
            Callable: 関数デコレータ
        """
        max_retry = max_attempts or self.config.max_retries
        delay = retry_delay or self.config.retry_delay
        
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                last_error: Optional[Exception] = None
                
                for attempt in range(1, max_retry + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_error = e
                        self._log_error(
                            e,
                            error_level,
                            f"Attempt {attempt}/{max_retry} failed in {func.__name__}"
                        )
                        
                        if attempt < max_retry:
                            if delay > 0:
                                time.sleep(delay)
                
                if last_error:
                    raise RetryExhaustedException(
                        f"All {max_retry} attempts failed for {func.__name__}"
                    ) from last_error
                return None
            return wrapper  # type: ignore
        return decorator
    
    def notify_error(
        self,
        ui_callback: Optional[Callable[[str], None]] = None,
        error_level: ErrorLevel = ErrorLevel.ERROR
    ) -> Callable[[F], F]:
        """
        エラーを通知するデコレータ
        
        Args:
            ui_callback: UIに通知するコールバック関数
            error_level: エラーログのレベル
            
        Returns:
            Callable: 関数デコレータ
        """
        def decorator(func: F) -> F:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_message = self._format_error(
                        e,
                        f"Error in {func.__name__}"
                    )
                    self._log_error(e, error_level)
                    
                    if ui_callback:
                        try:
                            ui_callback(error_message)
                        except Exception as callback_error:
                            self._log_error(
                                callback_error,
                                ErrorLevel.ERROR,
                                "UI callback failed"
                            )
                    
                    raise
            return wrapper  # type: ignore
        return decorator
    
    @staticmethod
    def setup_global_exception_handler(
        logger: Optional[logging.Logger] = None
    ) -> None:
        """
        グローバル例外ハンドラを設定する
        
        Args:
            logger: ロガーインスタンス
        """
        handler = ErrorHandler.get_instance()
        log = logger or handler.logger
        
        def global_exception_handler(
            exctype: Type[BaseException],
            value: BaseException,
            tb: Optional[TracebackType]
        ) -> None:
            """未ハンドルの例外をキャッチするハンドラ"""
            error_message = ''.join(
                traceback.format_exception(exctype, value, tb)
            )
            log.critical(f"未ハンドルの例外が発生しました:\n{error_message}")
            
            # 標準のエラーハンドラを呼び出し
            sys.__excepthook__(exctype, value, tb)
        
        # グローバル例外ハンドラを設定
        sys.excepthook = global_exception_handler
        log.info("グローバル例外ハンドラを設定しました")


# シングルトンインスタンス
error_handler = ErrorHandler.get_instance()


def create_error_handler(
    logger: Optional[logging.Logger] = None,
    config: Optional[Union[ErrorConfig, Dict[str, Any]]] = None
) -> ErrorHandler:
    """
    新しいErrorHandlerインスタンスを作成する
    
    Args:
        logger: ロガーインスタンス
        config: エラー処理の設定
        
    Returns:
        ErrorHandler: 新しいErrorHandlerインスタンス
    """
    if isinstance(config, dict):
        config_obj = ErrorConfig()
        for key, value in config.items():
            if hasattr(config_obj, key):
                setattr(config_obj, key, value)
        config = config_obj
    
    return ErrorHandler(logger, config) 