import yaml
from pathlib import Path
import logging


class AppConfig:
    """
    アプリケーション全体の設定を管理するクラス
    設定ファイル（config.yml）から設定を読み込む
    """
    
    def __init__(self, config_path=None):
        """
        コンストラクタ
        
        Args:
            config_path: 設定ファイルのパス（Noneの場合はデフォルトのパスを使用）
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or Path('config.yml')
        self.config_data = {}
        self._load_config()
        
    def _load_config(self):
        """設定ファイルから設定を読み込む"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            self.logger.info(f"設定ファイルを読み込みました: {self.config_path}")
        except Exception as e:
            self.logger.error(f"設定ファイルの読み込みに失敗しました: {e}")
            self.config_data = {}
    
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
        if section not in self.config_data:
            return default
            
        if key is None:
            return self.config_data.get(section, default)
        
        return self.config_data.get(section, {}).get(key, default)
    
    def set(self, section, key, value):
        """
        設定値を設定する
        
        Args:
            section: 設定セクション名
            key: 設定キー名
            value: 設定値
        """
        if section not in self.config_data:
            self.config_data[section] = {}
            
        self.config_data[section][key] = value
    
    def save(self):
        """設定をファイルに保存する"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config_data, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"設定ファイルを保存しました: {self.config_path}")
            return True
        except Exception as e:
            self.logger.error(f"設定ファイルの保存に失敗しました: {e}")
            return False 