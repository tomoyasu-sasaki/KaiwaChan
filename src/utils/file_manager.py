import os
import shutil
from pathlib import Path
import logging
import json
import yaml
import tempfile


class FileManager:
    """
    ファイル操作を管理するユーティリティクラス
    """
    
    def __init__(self):
        """コンストラクタ"""
        self.logger = logging.getLogger(__name__)
    
    def ensure_dir(self, directory):
        """
        ディレクトリが存在することを確認し、存在しない場合は作成する
        
        Args:
            directory: 作成するディレクトリのパス
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            os.makedirs(directory, exist_ok=True)
            return True
        except Exception as e:
            self.logger.error(f"ディレクトリの作成に失敗しました: {e}")
            return False
    
    def save_json(self, data, file_path):
        """
        データをJSONファイルに保存する
        
        Args:
            data: 保存するデータ
            file_path: 保存先のファイルパス
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # ディレクトリが存在しない場合は作成
            self.ensure_dir(os.path.dirname(file_path))
            
            # 一時ファイルに書き込んでから移動する（書き込み中の読み込みを防ぐため）
            with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as temp_file:
                json.dump(data, temp_file, indent=2, ensure_ascii=False)
                temp_path = temp_file.name
            
            # 一時ファイルを目的のファイルに移動
            shutil.move(temp_path, file_path)
            
            self.logger.debug(f"JSONファイルを保存しました: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"JSONファイルの保存に失敗しました: {e}")
            return False
    
    def load_json(self, file_path, default=None):
        """
        JSONファイルからデータを読み込む
        
        Args:
            file_path: 読み込むファイルパス
            default: ファイルが存在しない場合や読み込みに失敗した場合のデフォルト値
            
        Returns:
            データまたはデフォルト値
        """
        if not os.path.exists(file_path):
            return default
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.debug(f"JSONファイルを読み込みました: {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"JSONファイルの読み込みに失敗しました: {e}")
            return default
    
    def save_yaml(self, data, file_path):
        """
        データをYAMLファイルに保存する
        
        Args:
            data: 保存するデータ
            file_path: 保存先のファイルパス
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # ディレクトリが存在しない場合は作成
            self.ensure_dir(os.path.dirname(file_path))
            
            # 一時ファイルに書き込んでから移動する（書き込み中の読み込みを防ぐため）
            with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as temp_file:
                yaml.dump(data, temp_file, default_flow_style=False, allow_unicode=True)
                temp_path = temp_file.name
            
            # 一時ファイルを目的のファイルに移動
            shutil.move(temp_path, file_path)
            
            self.logger.debug(f"YAMLファイルを保存しました: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"YAMLファイルの保存に失敗しました: {e}")
            return False
    
    def load_yaml(self, file_path, default=None):
        """
        YAMLファイルからデータを読み込む
        
        Args:
            file_path: 読み込むファイルパス
            default: ファイルが存在しない場合や読み込みに失敗した場合のデフォルト値
            
        Returns:
            データまたはデフォルト値
        """
        if not os.path.exists(file_path):
            return default
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            self.logger.debug(f"YAMLファイルを読み込みました: {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"YAMLファイルの読み込みに失敗しました: {e}")
            return default
    
    def save_text(self, text, file_path):
        """
        テキストをファイルに保存する
        
        Args:
            text: 保存するテキスト
            file_path: 保存先のファイルパス
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # ディレクトリが存在しない場合は作成
            self.ensure_dir(os.path.dirname(file_path))
            
            # 一時ファイルに書き込んでから移動する（書き込み中の読み込みを防ぐため）
            with tempfile.NamedTemporaryFile('w', delete=False, encoding='utf-8') as temp_file:
                temp_file.write(text)
                temp_path = temp_file.name
            
            # 一時ファイルを目的のファイルに移動
            shutil.move(temp_path, file_path)
            
            self.logger.debug(f"テキストファイルを保存しました: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"テキストファイルの保存に失敗しました: {e}")
            return False
    
    def load_text(self, file_path, default=""):
        """
        テキストファイルからテキストを読み込む
        
        Args:
            file_path: 読み込むファイルパス
            default: ファイルが存在しない場合や読み込みに失敗した場合のデフォルト値
            
        Returns:
            テキストまたはデフォルト値
        """
        if not os.path.exists(file_path):
            return default
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.logger.debug(f"テキストファイルを読み込みました: {file_path}")
            return text
        except Exception as e:
            self.logger.error(f"テキストファイルの読み込みに失敗しました: {e}")
            return default
    
    def copy_file(self, src, dst):
        """
        ファイルをコピーする
        
        Args:
            src: コピー元のファイルパス
            dst: コピー先のファイルパス
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # コピー先のディレクトリが存在しない場合は作成
            self.ensure_dir(os.path.dirname(dst))
            
            shutil.copy2(src, dst)
            self.logger.debug(f"ファイルをコピーしました: {src} -> {dst}")
            return True
        except Exception as e:
            self.logger.error(f"ファイルのコピーに失敗しました: {e}")
            return False
    
    def move_file(self, src, dst):
        """
        ファイルを移動する
        
        Args:
            src: 移動元のファイルパス
            dst: 移動先のファイルパス
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            # 移動先のディレクトリが存在しない場合は作成
            self.ensure_dir(os.path.dirname(dst))
            
            shutil.move(src, dst)
            self.logger.debug(f"ファイルを移動しました: {src} -> {dst}")
            return True
        except Exception as e:
            self.logger.error(f"ファイルの移動に失敗しました: {e}")
            return False
    
    def delete_file(self, file_path):
        """
        ファイルを削除する
        
        Args:
            file_path: 削除するファイルパス
            
        Returns:
            bool: 成功したかどうか
        """
        if not os.path.exists(file_path):
            return True
            
        try:
            os.remove(file_path)
            self.logger.debug(f"ファイルを削除しました: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"ファイルの削除に失敗しました: {e}")
            return False 