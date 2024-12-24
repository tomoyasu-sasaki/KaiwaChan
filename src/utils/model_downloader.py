import requests
import os
from pathlib import Path
from tqdm import tqdm

class ModelDownloader:
    def __init__(self, config):
        self.config = config
        self.models_dir = Path(config.config["paths"]["models"])
        self.models_dir.mkdir(exist_ok=True)

    def download_llama(self):
        """
        1. https://huggingface.co/lmstudio-community/granite-3.1-8b-instruct-GGUF にアクセス
        2. Files and versionsタブを開く
        3. granite-3.1-8b-instruct-Q4_K_M.ggufをダウンロード
        4. modelsディレクトリに配置
        """
        model_path = self.models_dir / "granite-3.1-8b-instruct-Q4_K_M.gguf"
        print(f"Graniteモデルを {model_path} に配置してください")

    def setup_voicevox(self):
        """
        VOICEVOXのセットアップ手順を表示
        """
        print("VOICEVOXのセットアップ手順:")
        print("1. https://voicevox.hiroshiba.jp/からダウンロード")
        print("2. インストールして起動")
        print("3. config.ymlのvoicevox_engine_pathを設定") 