import json
from pathlib import Path
import logging
import os


class UserSettings:
    """
    ユーザー固有の設定を管理するクラス
    ユーザー設定はJSON形式で保存される
    """
    
    def __init__(self, settings_path=None):
        """
        コンストラクタ
        
        Args:
            settings_path: 設定ファイルのパス（Noneの場合はデフォルトのパスを使用）
        """
        self.logger = logging.getLogger(__name__)
        
        # デフォルトパスの設定
        if settings_path is None:
            app_data_dir = self._get_app_data_dir()
            settings_path = Path(app_data_dir) / 'settings.json'
        
        self.settings_path = Path(settings_path)
        self.settings = {}
        
        # ディレクトリが存在しない場合は作成
        os.makedirs(self.settings_path.parent, exist_ok=True)
        
        # 設定ファイルの読み込み
        self._load_settings()
    
    def _get_app_data_dir(self):
        """アプリケーションデータディレクトリを取得する"""
        home = Path.home()
        if os.name == 'nt':  # Windows
            return home / 'AppData' / 'Local' / 'KaiwaChan'
        elif os.name == 'posix':  # macOS/Linux
            return home / '.kaiwachan'
        else:
            return Path('.')  # その他のOS
    
    def _load_settings(self):
        """設定ファイルから設定を読み込む"""
        if not self.settings_path.exists():
            self.settings = self._get_default_settings()
            self._save_settings()
            return
            
        try:
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                self.settings = json.load(f)
            self.logger.info(f"ユーザー設定を読み込みました: {self.settings_path}")
        except Exception as e:
            self.logger.error(f"ユーザー設定の読み込みに失敗しました: {e}")
            self.settings = self._get_default_settings()
    
    def _save_settings(self):
        """設定をファイルに保存する"""
        try:
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            self.logger.info(f"ユーザー設定を保存しました: {self.settings_path}")
            return True
        except Exception as e:
            self.logger.error(f"ユーザー設定の保存に失敗しました: {e}")
            return False
    
    def _get_default_settings(self):
        """デフォルトの設定を取得する"""
        return {
            'general': {
                'language': 'ja',
                'theme': 'light',
                'first_run': True
            },
            'voice': {
                'current_profile': 'default',
                'volume': 0.8,
                'speed': 1.0
            },
            'ui': {
                'font_size': 12,
                'window_size': {
                    'width': 800,
                    'height': 600
                },
                'window_position': {
                    'x': 100,
                    'y': 100
                },
                'show_character': True
            }
        }
    
    def get(self, section, key=None, default=None):
        """
        設定値を取得する
        
        Args:
            section: 設定セクション名
            key: 設定キー名（Noneの場合はセクション全体を返す）
            default: デフォルト値
            
        Returns:
            設定値またはデフォルト値
        """
        if section not in self.settings:
            return default
            
        if key is None:
            return self.settings.get(section, default)
        
        return self.settings.get(section, {}).get(key, default)
    
    def set(self, section, key, value):
        """
        設定値を設定する
        
        Args:
            section: 設定セクション名
            key: 設定キー名
            value: 設定値
        """
        if section not in self.settings:
            self.settings[section] = {}
            
        self.settings[section][key] = value
        self._save_settings()
    
    def save(self):
        """設定をファイルに保存する"""
        return self._save_settings() 