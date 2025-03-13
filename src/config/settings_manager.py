import yaml
import json
import logging
from pathlib import Path
import os
from typing import Any, Dict, Optional, Union, List


class SettingsManager:
    """
    設定管理システム
    
    アプリケーション全体の設定とユーザー設定を一元管理するクラス
    シングルトンパターンを使用してアプリケーション全体で同じ設定インスタンスを使用する
    """
    
    # シングルトンインスタンス
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """シングルトンインスタンスを取得する"""
        if cls._instance is None:
            cls._instance = SettingsManager()
        return cls._instance
    
    def __init__(self):
        """コンストラクタ - このクラスは get_instance() メソッドを使用して取得することを推奨"""
        self.logger = logging.getLogger(__name__)
        
        # プロジェクトのルートディレクトリを取得
        self.root_dir = Path(__file__).parent.parent.parent
        
        # アプリケーション設定（config.yml）
        self.app_config_path = self.root_dir / "config.yml"
        self.app_config = {}
        
        # ユーザー設定（settings.json）
        self.user_settings_path = self._get_user_settings_path()
        self.user_settings = {}
        
        # 内部キャッシュ設定（メモリ内のみ）
        self.runtime_settings = {}
        
        # デフォルト設定
        self.default_app_config = self._get_default_app_config()
        self.default_user_settings = self._get_default_user_settings()
        
        # 設定の読み込み
        self._load_app_config()
        self._load_user_settings()
        
        self.logger.info("設定マネージャーを初期化しました")
    
    def _get_user_settings_path(self) -> Path:
        """ユーザー設定ファイルのパスを取得する"""
        if os.name == 'nt':  # Windows
            app_data_dir = Path.home() / 'AppData' / 'Local' / 'KaiwaChan'
        elif os.name == 'posix':  # macOS/Linux
            app_data_dir = Path.home() / '.kaiwachan'
        else:
            app_data_dir = self.root_dir
        
        # ディレクトリが存在しない場合は作成
        app_data_dir.mkdir(parents=True, exist_ok=True)
        
        return app_data_dir / 'settings.json'
    
    def _get_default_app_config(self) -> Dict[str, Any]:
        """デフォルトのアプリケーション設定を取得する"""
        return {
            "models": {
                "whisper": "base",
                "llm": {
                    "path": str(self.root_dir / "models" / "granite-3.1-8b-instruct-Q4_K_M.gguf"),
                    "n_threads": 8,
                    "n_batch": 512,
                    "max_tokens": 128
                },
                "voicevox": {
                    "engine_path": "",
                    "cache_size": 1000,
                    "timeout": 5
                }
            },
            "audio": {
                "sample_rate": 16000,
                "duration": 5,
                "channels": 1,
                "device": None
            },
            "voice_clone": {
                "profiles_dir": str(self.root_dir / "profiles"),
                "sample_rate": 22050,
                "chunk_size": 1024,
                "feature_dim": 256
            },
            "animation": {
                "window_width": 400,
                "window_height": 600,
                "fps": 30,
                "background_color": [255, 255, 255],
                "default_character": "default",
                "assets_dir": str(self.root_dir / "assets" / "images"),
                "default_width": 300,
                "default_height": 400
            },
            "paths": {
                "logs": str(self.root_dir / "logs"),
                "models": str(self.root_dir / "models"),
                "cache": str(self.root_dir / "cache")
            }
        }
    
    def _get_default_user_settings(self) -> Dict[str, Any]:
        """デフォルトのユーザー設定を取得する"""
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
                'show_character': True,
                'show_logs': False
            },
            'behavior': {
                'auto_listen': True,
                'character_name': 'カイワちゃん',
                'prompt_template': "あなたは{character_name}という名前の会話AIです。ユーザーと楽しく会話してください。"
            }
        }
    
    def _load_app_config(self) -> None:
        """アプリケーション設定を読み込む"""
        try:
            if self.app_config_path.exists():
                with open(self.app_config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
                    # デフォルト設定と結合
                    self.app_config = self._deep_merge(self.default_app_config.copy(), loaded_config)
                    self.logger.info(f"アプリケーション設定を読み込みました: {self.app_config_path}")
            else:
                # 設定ファイルが存在しない場合はデフォルト設定を保存
                self.app_config = self.default_app_config.copy()
                self._save_app_config()
                self.logger.info(f"デフォルトのアプリケーション設定を作成しました: {self.app_config_path}")
        except Exception as e:
            self.logger.error(f"アプリケーション設定の読み込みに失敗しました: {e}")
            self.app_config = self.default_app_config.copy()
    
    def _load_user_settings(self) -> None:
        """ユーザー設定を読み込む"""
        try:
            if self.user_settings_path.exists():
                with open(self.user_settings_path, 'r', encoding='utf-8') as f:
                    self.user_settings = json.load(f)
                    # デフォルト設定と結合して不足している設定項目を補完
                    for section, values in self.default_user_settings.items():
                        if section not in self.user_settings:
                            self.user_settings[section] = values
                        else:
                            for key, value in values.items():
                                if key not in self.user_settings[section]:
                                    self.user_settings[section][key] = value
                    
                    self.logger.info(f"ユーザー設定を読み込みました: {self.user_settings_path}")
            else:
                # 設定ファイルが存在しない場合はデフォルト設定を保存
                self.user_settings = self.default_user_settings.copy()
                self._save_user_settings()
                self.logger.info(f"デフォルトのユーザー設定を作成しました: {self.user_settings_path}")
        except Exception as e:
            self.logger.error(f"ユーザー設定の読み込みに失敗しました: {e}")
            self.user_settings = self.default_user_settings.copy()
    
    def _save_app_config(self) -> bool:
        """アプリケーション設定をファイルに保存する"""
        try:
            with open(self.app_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.app_config, f, default_flow_style=False, allow_unicode=True)
            self.logger.info(f"アプリケーション設定を保存しました: {self.app_config_path}")
            return True
        except Exception as e:
            self.logger.error(f"アプリケーション設定の保存に失敗しました: {e}")
            return False
    
    def _save_user_settings(self) -> bool:
        """ユーザー設定をファイルに保存する"""
        try:
            with open(self.user_settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_settings, f, indent=2, ensure_ascii=False)
            self.logger.info(f"ユーザー設定を保存しました: {self.user_settings_path}")
            return True
        except Exception as e:
            self.logger.error(f"ユーザー設定の保存に失敗しました: {e}")
            return False
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        2つの辞書を再帰的に結合する
        
        Args:
            base: ベースとなる辞書
            override: 上書きする辞書
            
        Returns:
            結合された辞書
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base
    
    def get_app_config(self, section: Optional[str] = None, key: Optional[str] = None, default: Any = None) -> Any:
        """
        アプリケーション設定を取得する
        
        Args:
            section: 設定セクション名（Noneの場合は全体を返す）
            key: 設定キー名（Noneの場合はセクション全体を返す）
            default: デフォルト値
            
        Returns:
            設定値またはデフォルト値
        """
        if section is None:
            return self.app_config
            
        if section not in self.app_config:
            return default
            
        if key is None:
            return self.app_config.get(section, default)
        
        return self.app_config.get(section, {}).get(key, default)
    
    def set_app_config(self, section: str, key: str, value: Any, save: bool = True) -> None:
        """
        アプリケーション設定を設定する
        
        Args:
            section: 設定セクション名
            key: 設定キー名
            value: 設定値
            save: 設定を保存するかどうか
        """
        if section not in self.app_config:
            self.app_config[section] = {}
            
        self.app_config[section][key] = value
        
        if save:
            self._save_app_config()
    
    def get_user_setting(self, section: Optional[str] = None, key: Optional[str] = None, default: Any = None) -> Any:
        """
        ユーザー設定を取得する
        
        Args:
            section: 設定セクション名（Noneの場合は全体を返す）
            key: 設定キー名（Noneの場合はセクション全体を返す）
            default: デフォルト値
            
        Returns:
            設定値またはデフォルト値
        """
        if section is None:
            return self.user_settings
            
        if section not in self.user_settings:
            return default
            
        if key is None:
            return self.user_settings.get(section, default)
        
        return self.user_settings.get(section, {}).get(key, default)
    
    def set_user_setting(self, section: str, key: str, value: Any, save: bool = True) -> None:
        """
        ユーザー設定を設定する
        
        Args:
            section: 設定セクション名
            key: 設定キー名
            value: 設定値
            save: 設定を保存するかどうか
        """
        if section not in self.user_settings:
            self.user_settings[section] = {}
            
        self.user_settings[section][key] = value
        
        if save:
            self._save_user_settings()
    
    def get_runtime_setting(self, key: str, default: Any = None) -> Any:
        """
        実行時設定を取得する（メモリ内のみ、保存されない）
        
        Args:
            key: 設定キー
            default: デフォルト値
            
        Returns:
            設定値またはデフォルト値
        """
        return self.runtime_settings.get(key, default)
    
    def set_runtime_setting(self, key: str, value: Any) -> None:
        """
        実行時設定を設定する（メモリ内のみ、保存されない）
        
        Args:
            key: 設定キー
            value: 設定値
        """
        self.runtime_settings[key] = value
    
    def save_all(self) -> bool:
        """すべての設定を保存する"""
        app_saved = self._save_app_config()
        user_saved = self._save_user_settings()
        return app_saved and user_saved
    
    def reset_app_config(self) -> None:
        """アプリケーション設定をデフォルトに戻す"""
        self.app_config = self.default_app_config.copy()
        self._save_app_config()
        self.logger.info("アプリケーション設定をデフォルトにリセットしました")
    
    def reset_user_settings(self) -> None:
        """ユーザー設定をデフォルトに戻す"""
        self.user_settings = self.default_user_settings.copy()
        self._save_user_settings()
        self.logger.info("ユーザー設定をデフォルトにリセットしました")
    
    def get_dot_path(self, key_path: str, default: Any = None) -> Any:
        """
        ドット区切りのパスで設定値を取得する（簡易アクセス用）
        例: settings.get_dot_path("audio.sample_rate")
        
        Args:
            key_path: ドット区切りの設定パス（"セクション.キー"形式）
            default: デフォルト値
            
        Returns:
            設定値またはデフォルト値
        """
        parts = key_path.split('.')
        
        if len(parts) == 1:
            # セクションのみの場合
            return self.get_app_config(parts[0], None, default)
        
        # まずアプリケーション設定から検索
        value = self.get_app_config(parts[0], parts[1], None)
        if value is not None:
            return value
            
        # 次にユーザー設定から検索
        value = self.get_user_setting(parts[0], parts[1], None)
        if value is not None:
            return value
            
        # 最後に実行時設定から検索
        if parts[0] in self.runtime_settings:
            if len(parts) == 2:
                return self.runtime_settings.get(parts[0], {}).get(parts[1], default)
        
        return default
        
    def get(self, section: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        互換性のあるget()メソッド - 古いConfigクラスのインターフェースに合わせたメソッド
        
        Args:
            section: 設定セクション名
            key: 設定キー名（Noneの場合はセクション全体を返す）
            default: デフォルト値
            
        Returns:
            設定値またはデフォルト値
        """
        return self.get_app_config(section, key, default)
    
    def get_model_path(self, model_name: str) -> Path:
        """
        モデルファイルのパスを取得する
        
        Args:
            model_name: モデル名（"llm", "whisper"など）
            
        Returns:
            Path: モデルファイルのパス
        """
        models_dir = Path(self.get_app_config("paths", "models"))
        
        # モデル名に応じてパスを返す
        if model_name == "llm":
            model_config = self.get_app_config("models", "llm", {})
            model_path = model_config.get("path")
            if model_path:
                return Path(model_path)
            
        elif model_name == "whisper":
            model_file = self.get_app_config("models", "whisper", {}).get("file", "whisper-small.en")
            return models_dir / "whisper" / model_file
        
        # デフォルトはモデル名をそのまま使用
        return models_dir / model_name 