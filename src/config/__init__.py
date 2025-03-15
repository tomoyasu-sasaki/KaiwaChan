"""
KaiwaChan 設定パッケージ
アプリケーションの設定管理機能を提供します
"""

from .app_config import AppConfig
from .user_settings import UserSettings
from .settings_manager import SettingsManager

# 推奨：シングルトンインスタンスを取得するヘルパー関数
def get_settings() -> SettingsManager:
    """
    設定マネージャーのシングルトンインスタンスを取得する
    
    Returns:
        SettingsManagerインスタンス
    """
    return SettingsManager.get_instance()

__all__ = [
    'AppConfig',       # 後方互換性のため維持（非推奨）
    'UserSettings',    # 後方互換性のため維持（非推奨）
    'SettingsManager', # 新しい統合設定マネージャー
    'get_settings'     # 設定インスタンスを取得するヘルパー関数
]
