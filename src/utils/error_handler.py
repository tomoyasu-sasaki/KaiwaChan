import logging
import traceback
import sys
from functools import wraps


class ErrorHandler:
    """
    エラー処理を管理するユーティリティクラス
    """
    
    def __init__(self, logger=None):
        """
        コンストラクタ
        
        Args:
            logger: ロガーオブジェクト（Noneの場合は新しく作成）
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def handle_exception(self, e, message=None):
        """
        例外を処理する
        
        Args:
            e: 例外オブジェクト
            message: エラーメッセージ
            
        Returns:
            str: エラーメッセージ
        """
        error_message = message or "エラーが発生しました"
        error_details = f"{error_message}: {str(e)}"
        
        # スタックトレースをログに記録
        self.logger.error(error_details)
        self.logger.debug(traceback.format_exc())
        
        return error_details
    
    def silent_error(self, default_value=None):
        """
        エラーを無視するデコレータ
        
        Args:
            default_value: エラー発生時に返す値
            
        Returns:
            関数デコレータ
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    self.logger.error(f"Silent error in {func.__name__}: {str(e)}")
                    self.logger.debug(traceback.format_exc())
                    return default_value
            return wrapper
        return decorator
    
    def retry(self, max_attempts=3, retry_delay=0, exceptions=(Exception,)):
        """
        リトライするデコレータ
        
        Args:
            max_attempts: 最大試行回数
            retry_delay: リトライ前の待機時間（秒）
            exceptions: キャッチする例外のタプル
            
        Returns:
            関数デコレータ
        """
        import time
        
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                last_exception = None
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        last_exception = e
                        self.logger.warning(f"Attempt {attempt}/{max_attempts} failed: {str(e)}")
                        
                        if attempt < max_attempts:
                            if retry_delay > 0:
                                time.sleep(retry_delay)
                
                # 全ての試行が失敗した場合
                self.logger.error(f"All {max_attempts} attempts failed for {func.__name__}")
                raise last_exception
            return wrapper
        return decorator
    
    def notify_error(self, ui_callback=None):
        """
        エラーを通知するデコレータ
        
        Args:
            ui_callback: UIに通知するコールバック関数
            
        Returns:
            関数デコレータ
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_message = f"エラーが発生しました: {str(e)}"
                    self.logger.error(error_message)
                    self.logger.debug(traceback.format_exc())
                    
                    # UIコールバックが指定されている場合は呼び出し
                    if ui_callback:
                        ui_callback(error_message)
                    
                    # 元の例外を再発生
                    raise
            return wrapper
        return decorator
    
    @staticmethod
    def setup_global_exception_handler():
        """
        グローバル例外ハンドラを設定する
        """
        logger = logging.getLogger(__name__)
        
        def global_exception_handler(exctype, value, tb):
            """未ハンドルの例外をキャッチするハンドラ"""
            error_message = ''.join(traceback.format_exception(exctype, value, tb))
            logger.critical(f"未ハンドルの例外が発生しました:\n{error_message}")
            
            # 標準のエラーハンドラを呼び出し
            sys.__excepthook__(exctype, value, tb)
        
        # グローバル例外ハンドラを設定
        sys.excepthook = global_exception_handler
        
        logger.info("グローバル例外ハンドラを設定しました")


# シングルトンインスタンス
error_handler = ErrorHandler() 