import yaml
import os
from pathlib import Path

class Config:
    def __init__(self):
        # プロジェクトのルートディレクトリを取得
        self.root_dir = Path(__file__).parent.parent.parent
        self.config_path = self.root_dir / "config.yml"
        
        self.default_config = {
            "models": {
                "whisper": "base",
                "llama": str(self.root_dir / "models" / "llama-2-7b-chat.gguf"),
                "voicevox_engine_path": ""
            },
            "audio": {
                "sample_rate": 16000,
                "duration": 5
            },
            "paths": {
                "logs": str(self.root_dir / "logs"),
                "models": str(self.root_dir / "models")
            }
        }
        self.config = self._load_config()

    def _load_config(self):
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        self._save_config(self.default_config)
        return self.default_config

    def _save_config(self, config):
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)

    def get_model_path(self, model_name):
        """モデルの絶対パスを取得"""
        return Path(self.config["models"][model_name]) 