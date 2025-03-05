import requests
import os
from pathlib import Path
from tqdm import tqdm
import logging
import hashlib
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor
import shutil
import time
import platform


class ModelDownloader:
    """
    AIモデルのダウンロードと管理を行うクラス
    """
    
    # サポートするモデルの定義
    MODELS = {
        "llama": {
            "name": "Granite 3.1 8B Instruct",
            "files": [
                {
                    "url": "https://huggingface.co/lmstudio-community/granite-3.1-8b-instruct-GGUF/resolve/main/granite-3.1-8b-instruct-Q4_K_M.gguf",
                    "filename": "granite-3.1-8b-instruct-Q4_K_M.gguf",
                    "size": 4903691116,  # 約4.9GB
                    "md5": None,  # 不明な場合はNone
                    "description": "Granite 3.1 8BモデルのQ4_K_M量子化版"
                },
                {
                    "url": "https://huggingface.co/lmstudio-community/granite-3.1-8b-instruct-GGUF/resolve/main/granite-3.1-8b-instruct-Q5_K_M.gguf",
                    "filename": "granite-3.1-8b-instruct-Q5_K_M.gguf",
                    "size": 5910713612,  # 約5.9GB
                    "md5": None,
                    "description": "Granite 3.1 8BモデルのQ5_K_M量子化版（より高品質）"
                }
            ],
            "required": True,
            "description": "メインの対話生成モデル"
        },
        "whisper": {
            "name": "Whisper (medium)",
            "files": [
                {
                    "url": None,  # PyTorchが自動的にダウンロードする
                    "filename": "medium.pt",
                    "size": 1500000000,  # 約1.5GB
                    "md5": None,
                    "description": "OpenAI Whisper中サイズモデル"
                }
            ],
            "required": True,
            "description": "音声認識モデル"
        },
        "tts": {
            "name": "TTS Kokoro Model",
            "files": [
                {
                    "url": None,  # ライブラリが自動的にダウンロードする
                    "filename": "tts_models/ja/kokoro/tacotron2-DDC",
                    "size": 300000000,  # 約300MB
                    "md5": None,
                    "description": "日本語音声合成モデル"
                }
            ],
            "required": False,
            "description": "テキスト音声合成モデル（VOICEVOXの代替）"
        }
    }
    
    def __init__(self, config):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.models_dir = Path(config.get('paths', 'models', 'models'))
        self.cache_dir = Path(config.get('paths', 'cache', 'cache'))
        self.voicevox_path = config.get('voicevox', 'engine_path', '')
        
        # ディレクトリを作成
        self.models_dir.mkdir(exist_ok=True)
        self.cache_dir.mkdir(exist_ok=True)
        
        # ダウンロード設定
        self.chunk_size = 8192  # 一度にダウンロードするバイト数
        self.max_retries = 3    # ダウンロードの最大リトライ回数
        self.timeout = 30       # HTTPリクエストのタイムアウト（秒）
        
        # 並列ダウンロード用のスレッドプール
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    def check_all_models(self):
        """
        すべてのモデルの存在を確認する
        
        Returns:
            dict: モデル名をキー、存在するかどうかを値とする辞書
        """
        results = {}
        
        for model_id, model_info in self.MODELS.items():
            results[model_id] = self.check_model(model_id)
            
        return results
    
    def check_model(self, model_id):
        """
        指定されたモデルの存在を確認する
        
        Args:
            model_id: モデルID
            
        Returns:
            bool: モデルが存在するかどうか
        """
        if model_id not in self.MODELS:
            self.logger.warning(f"未知のモデルID: {model_id}")
            return False
            
        model_info = self.MODELS[model_id]
        
        # すべてのファイルの存在を確認
        for file_info in model_info["files"]:
            if not file_info["url"]:
                # URLがない場合はライブラリが自動的にダウンロードするのでスキップ
                continue
                
            filename = file_info["filename"]
            file_path = self.models_dir / filename
            
            if not file_path.exists():
                return False
                
            # ファイルサイズを確認
            if file_path.stat().st_size < file_info["size"] * 0.9:  # 10%の誤差を許容
                return False
                
            # MD5を確認（設定されている場合）
            if file_info["md5"] and not self._verify_md5(file_path, file_info["md5"]):
                return False
                
        return True
    
    def download_model(self, model_id, force=False, callback=None):
        """
        指定されたモデルをダウンロードする
        
        Args:
            model_id: モデルID
            force: 既存のファイルを上書きするかどうか
            callback: 進捗コールバック関数 callback(current, total, message)
            
        Returns:
            bool: ダウンロードが成功したかどうか
        """
        if model_id not in self.MODELS:
            self.logger.error(f"未知のモデルID: {model_id}")
            return False
            
        model_info = self.MODELS[model_id]
        self.logger.info(f"{model_info['name']}のダウンロードを開始します...")
        
        if callback:
            callback(0, 100, f"{model_info['name']}のダウンロードを開始します...")
        
        # 各ファイルをダウンロード
        success = True
        for i, file_info in enumerate(model_info["files"]):
            if not file_info["url"]:
                # URLがない場合はスキップ
                self.logger.info(f"{file_info['filename']}は自動ダウンロードされます。")
                if callback:
                    callback((i+1) * 100 // len(model_info["files"]), 100, 
                            f"{file_info['filename']}は自動ダウンロードされます。")
                continue
                
            filename = file_info["filename"]
            file_path = self.models_dir / filename
            
            # ファイルが既に存在するかを確認
            if file_path.exists() and not force:
                if self._verify_file(file_path, file_info):
                    self.logger.info(f"{filename}は既に存在します。")
                    if callback:
                        callback((i+1) * 100 // len(model_info["files"]), 100, 
                                f"{filename}は既に存在します。")
                    continue
                else:
                    self.logger.warning(f"{filename}は不完全または破損しています。再ダウンロードします。")
            
            # ダウンロード実行
            if callback:
                callback(i * 100 // len(model_info["files"]), 100, 
                        f"{filename}のダウンロードを開始します...")
                
            file_success = self._download_file(
                file_info["url"], 
                file_path, 
                file_info["size"],
                callback=lambda current, total: callback(
                    i * 100 // len(model_info["files"]) + current // len(model_info["files"]),
                    100,
                    f"{filename}をダウンロード中: {current}/{total} MB"
                ) if callback else None
            )
            
            if not file_success:
                success = False
                
        if success:
            self.logger.info(f"{model_info['name']}のダウンロードが完了しました。")
            if callback:
                callback(100, 100, f"{model_info['name']}のダウンロードが完了しました。")
        else:
            self.logger.error(f"{model_info['name']}のダウンロードに失敗しました。")
            if callback:
                callback(100, 100, f"{model_info['name']}のダウンロードに失敗しました。")
                
        return success
    
    def download_required_models(self, force=False, callback=None):
        """
        必須モデルをすべてダウンロードする
        
        Args:
            force: 既存のファイルを上書きするかどうか
            callback: 進捗コールバック関数 callback(current, total, message)
            
        Returns:
            bool: すべてのダウンロードが成功したかどうか
        """
        required_models = [model_id for model_id, model_info in self.MODELS.items() 
                          if model_info["required"]]
        
        total_models = len(required_models)
        success = True
        
        for i, model_id in enumerate(required_models):
            if callback:
                callback(i * 100 // total_models, 100, 
                        f"モデル {i+1}/{total_models} をダウンロード中...")
                
            model_success = self.download_model(
                model_id, 
                force,
                callback=lambda current, total, message: callback(
                    i * 100 // total_models + current // total_models,
                    100,
                    message
                ) if callback else None
            )
            
            if not model_success:
                success = False
                
        if success:
            self.logger.info("すべての必須モデルがダウンロードされました。")
            if callback:
                callback(100, 100, "すべての必須モデルがダウンロードされました。")
        else:
            self.logger.error("一部のモデルのダウンロードに失敗しました。")
            if callback:
                callback(100, 100, "一部のモデルのダウンロードに失敗しました。")
                
        return success
    
    def setup_voicevox(self, callback=None):
        """
        VOICEVOXのセットアップを実行する
        
        Args:
            callback: 進捗コールバック関数 callback(current, total, message)
            
        Returns:
            bool: セットアップが成功したかどうか
        """
        messages = [
            "VOICEVOXのセットアップ手順:",
            "1. https://voicevox.hiroshiba.jp/ からダウンロード",
            "2. インストールして起動",
            "3. config.ymlのvoicevox_engine_pathを設定"
        ]
        
        for i, message in enumerate(messages):
            self.logger.info(message)
            if callback:
                callback(i * 100 // len(messages), 100, message)
        
        # VOICEVOXの設定をチェック
        if self.voicevox_path:
            voicevox_exists = Path(self.voicevox_path).exists()
            status = "存在します" if voicevox_exists else "存在しません"
            message = f"VOICEVOX: {self.voicevox_path} ({status})"
            self.logger.info(message)
            if callback:
                callback(100, 100, message)
            return voicevox_exists
        else:
            message = "VOICEVOXのパスが設定されていません。"
            self.logger.warning(message)
            if callback:
                callback(100, 100, message)
            return False
    
    def _download_file(self, url, file_path, expected_size=None, callback=None):
        """
        ファイルをダウンロードする
        
        Args:
            url: ダウンロードURL
            file_path: 保存先のパス
            expected_size: 期待されるファイルサイズ（バイト）
            callback: 進捗コールバック関数 callback(current_mb, total_mb)
            
        Returns:
            bool: ダウンロードが成功したかどうか
        """
        temp_path = file_path.with_suffix(file_path.suffix + '.download')
        
        try:
            # 一時ファイルが既に存在する場合は削除
            if temp_path.exists():
                temp_path.unlink()
            
            # ディレクトリが存在しない場合は作成
            os.makedirs(file_path.parent, exist_ok=True)
            
            # ダウンロード開始
            for retry in range(self.max_retries):
                try:
                    with requests.get(url, stream=True, timeout=self.timeout) as response:
                        response.raise_for_status()
                        
                        # ファイルサイズを取得
                        total_size = int(response.headers.get('content-length', 0))
                        total_mb = total_size / (1024 * 1024)
                        
                        if expected_size and total_size > 0 and expected_size > 0:
                            # サイズの差が10%以上ある場合は警告
                            if abs(total_size - expected_size) > expected_size * 0.1:
                                self.logger.warning(
                                    f"期待されるファイルサイズと実際のサイズが異なります: "
                                    f"期待={expected_size/1024/1024:.1f}MB, "
                                    f"実際={total_size/1024/1024:.1f}MB"
                                )
                        
                        # プログレスバー用の設定
                        if total_size > 0:
                            progress_bar = tqdm(
                                total=total_mb, 
                                unit='MB', 
                                unit_scale=True,
                                desc=f"Downloading {file_path.name}"
                            )
                        else:
                            progress_bar = None
                            
                        # ファイルに書き込み
                        downloaded_size = 0
                        with open(temp_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=self.chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)
                                    
                                    # 進捗を更新
                                    downloaded_mb = downloaded_size / (1024 * 1024)
                                    if progress_bar:
                                        progress_bar.update(len(chunk) / (1024 * 1024))
                                    if callback:
                                        callback(int(downloaded_mb), int(total_mb) if total_mb > 0 else 100)
                        
                        if progress_bar:
                            progress_bar.close()
                    
                    # ダウンロードが成功したら、一時ファイルを正式なファイルにリネーム
                    if file_path.exists():
                        file_path.unlink()
                    temp_path.rename(file_path)
                    
                    self.logger.info(f"{file_path.name}のダウンロードが完了しました。")
                    return True
                    
                except (requests.exceptions.RequestException, IOError) as e:
                    self.logger.error(f"ダウンロード中にエラーが発生しました: {e}")
                    if retry < self.max_retries - 1:
                        wait_time = 2 ** retry  # 指数バックオフ
                        self.logger.info(f"{wait_time}秒後にリトライします...")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"{self.max_retries}回のリトライ後も失敗しました。")
                        # 一時ファイルが存在する場合は削除
                        if temp_path.exists():
                            temp_path.unlink()
                        return False
            
            return False
                    
        except Exception as e:
            self.logger.error(f"予期しないエラーが発生しました: {e}")
            # 一時ファイルが存在する場合は削除
            if temp_path.exists():
                temp_path.unlink()
            return False
    
    def _verify_file(self, file_path, file_info):
        """
        ダウンロードしたファイルを検証する
        
        Args:
            file_path: ファイルのパス
            file_info: ファイル情報の辞書
            
        Returns:
            bool: ファイルが正しいかどうか
        """
        # ファイルが存在することを確認
        if not file_path.exists():
            return False
            
        # ファイルサイズを確認
        actual_size = file_path.stat().st_size
        expected_size = file_info["size"]
        
        if expected_size and actual_size < expected_size * 0.9:  # 10%の誤差を許容
            self.logger.warning(
                f"ファイルサイズが期待値より小さいです: {actual_size/1024/1024:.1f}MB < {expected_size/1024/1024:.1f}MB"
            )
            return False
            
        # MD5を確認（設定されている場合）
        if file_info["md5"] and not self._verify_md5(file_path, file_info["md5"]):
            self.logger.warning(f"MD5チェックサムが一致しません: {file_path.name}")
            return False
            
        return True
    
    def _verify_md5(self, file_path, expected_md5):
        """
        ファイルのMD5チェックサムを検証する
        
        Args:
            file_path: ファイルのパス
            expected_md5: 期待されるMD5ハッシュ
            
        Returns:
            bool: MD5が一致するかどうか
        """
        try:
            md5_hash = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    md5_hash.update(chunk)
            actual_md5 = md5_hash.hexdigest()
            return actual_md5.lower() == expected_md5.lower()
        except Exception as e:
            self.logger.error(f"MD5チェックサム計算中にエラーが発生しました: {e}")
            return False 