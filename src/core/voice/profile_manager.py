import os
import json
import time
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import shutil

class VoiceProfileManager:
    """
    音声プロファイル管理クラス
    
    ユーザーの音声プロファイル情報を管理し、保存・読み込み・削除などの操作を行います。
    """
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # プロファイルの格納ディレクトリ
        self.profiles_dir = Path.home() / ".kaiwa_chan" / "voice_profiles"
        if config:
            custom_dir = config.get('voice_clone', 'profiles_dir', None)
            if custom_dir:
                self.profiles_dir = Path(custom_dir)
        
        # プロファイル情報
        self.profiles = {}
        
        # プロファイルを読み込む
        self._load_existing_profiles()
    
    def _load_existing_profiles(self) -> None:
        """既存の音声プロファイルを読み込む"""
        try:
            # ディレクトリが存在しない場合は作成
            self.profiles_dir.mkdir(parents=True, exist_ok=True)
            
            profile_count = 0
            
            # ディレクトリ内のすべてのサブディレクトリを検索
            for profile_dir in self.profiles_dir.iterdir():
                if not profile_dir.is_dir():
                    continue
                
                profile_id = profile_dir.name
                profile_path = profile_dir / "profile.json"
                
                if not profile_path.exists():
                    self.logger.warning(f"プロファイルメタデータが見つかりません: {profile_id}")
                    continue
                
                try:
                    # プロファイル情報を読み込む
                    with open(profile_path, 'r', encoding='utf-8') as f:
                        profile_data = json.load(f)
                    
                    # 埋め込みデータを読み込む
                    embedding_path = profile_dir / "speaker_embedding.npy"
                    if embedding_path.exists():
                        profile_data['speaker_embedding'] = np.load(embedding_path)
                    
                    # F0統計情報を読み込む
                    f0_path = profile_dir / "f0_samples.npy"
                    if f0_path.exists():
                        profile_data['f0_samples'] = np.load(f0_path)
                    
                    # メルスペクトログラムを読み込む
                    mel_path = profile_dir / "mel_spec_mean.npy"
                    if mel_path.exists():
                        profile_data['mel_spec_mean'] = np.load(mel_path)
                    
                    # プロファイル情報を登録
                    self.profiles[profile_id] = profile_data
                    profile_count += 1
                    
                except Exception as e:
                    self.logger.error(f"プロファイル読み込みエラー: {profile_id} - {e}")
                    continue
            
            self.logger.info(f"{profile_count}個のプロファイルを読み込みました")
            
        except Exception as e:
            self.logger.error(f"プロファイル読み込み中にエラーが発生: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def get_profile_ids(self) -> List[str]:
        """
        利用可能なプロファイルIDのリストを取得
        
        Returns:
            プロファイルIDのリスト
        """
        return list(self.profiles.keys())
    
    def get_profile_names(self) -> Dict[str, str]:
        """
        プロファイルID → 名前のマッピングを取得
        
        Returns:
            プロファイルID → 名前の辞書
        """
        return {
            profile_id: profile.get('name', profile_id)
            for profile_id, profile in self.profiles.items()
        }
    
    def get_profile(self, profile_id: str) -> Optional[Dict]:
        """
        指定されたIDのプロファイルを取得
        
        Args:
            profile_id: プロファイルID
            
        Returns:
            プロファイル情報の辞書、存在しない場合はNone
        """
        return self.profiles.get(profile_id)
    
    def save_profile(self, profile_id: str, profile_data: Dict) -> bool:
        """
        プロファイルを保存する
        
        Args:
            profile_id: プロファイルID
            profile_data: プロファイル情報の辞書
            
        Returns:
            保存に成功した場合はTrue、失敗した場合はFalse
        """
        try:
            # プロファイルIDを正規化（ファイル名に使用できない文字を除去）
            safe_profile_id = self._sanitize_profile_id(profile_id)
            if not safe_profile_id:
                self.logger.error(f"無効なプロファイルID: {profile_id}")
                return False
            
            # プロファイルディレクトリを作成
            profile_dir = self.profiles_dir / safe_profile_id
            profile_dir.mkdir(parents=True, exist_ok=True)
            
            # メタデータコピーを作成（NumPy配列は別途保存するため除外）
            metadata = {k: v for k, v in profile_data.items() if not isinstance(v, np.ndarray)}
            
            # 作成日時がない場合は追加
            if 'created_at' not in metadata:
                metadata['created_at'] = time.time()
            
            # 更新日時を設定
            metadata['updated_at'] = time.time()
            
            # メタデータをJSON形式で保存
            with open(profile_dir / "profile.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            # 話者埋め込みを保存
            if 'speaker_embedding' in profile_data and profile_data['speaker_embedding'] is not None:
                np.save(profile_dir / "speaker_embedding.npy", profile_data['speaker_embedding'])
            
            # F0サンプルを保存
            if 'f0_samples' in profile_data and profile_data['f0_samples'] is not None:
                np.save(profile_dir / "f0_samples.npy", profile_data['f0_samples'])
            
            # メルスペクトログラムを保存
            if 'mel_spec_mean' in profile_data and profile_data['mel_spec_mean'] is not None:
                np.save(profile_dir / "mel_spec_mean.npy", profile_data['mel_spec_mean'])
            
            # プロファイル情報を更新
            self.profiles[safe_profile_id] = profile_data
            
            self.logger.info(f"プロファイルを保存しました: {safe_profile_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"プロファイル保存中にエラーが発生: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        プロファイルを削除する
        
        Args:
            profile_id: 削除するプロファイルのID
            
        Returns:
            削除に成功した場合はTrue、失敗した場合はFalse
        """
        try:
            if profile_id not in self.profiles:
                self.logger.warning(f"存在しないプロファイルは削除できません: {profile_id}")
                return False
            
            # プロファイルディレクトリを削除
            profile_dir = self.profiles_dir / profile_id
            if profile_dir.exists():
                shutil.rmtree(profile_dir)
            
            # プロファイル情報を削除
            del self.profiles[profile_id]
            
            self.logger.info(f"プロファイルを削除しました: {profile_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"プロファイル削除中にエラーが発生: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def rename_profile(self, profile_id: str, new_name: str) -> bool:
        """
        プロファイルの名前を変更する
        
        Args:
            profile_id: プロファイルID
            new_name: 新しい名前
            
        Returns:
            変更に成功した場合はTrue、失敗した場合はFalse
        """
        try:
            if profile_id not in self.profiles:
                self.logger.warning(f"存在しないプロファイルの名前は変更できません: {profile_id}")
                return False
            
            # プロファイル情報を更新
            self.profiles[profile_id]['name'] = new_name
            
            # プロファイルを保存
            return self.save_profile(profile_id, self.profiles[profile_id])
            
        except Exception as e:
            self.logger.error(f"プロファイル名変更中にエラーが発生: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def _sanitize_profile_id(self, profile_id: str) -> str:
        """
        プロファイルIDを正規化する（ファイル名に使用できる形式に変換）
        
        Args:
            profile_id: 元のプロファイルID
            
        Returns:
            正規化されたプロファイルID
        """
        # 空白をアンダースコアに、特殊文字を削除
        safe_id = profile_id.strip()
        safe_id = safe_id.replace(' ', '_')
        
        # ファイル名に使用できない文字を削除
        invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
        for char in invalid_chars:
            safe_id = safe_id.replace(char, '')
        
        # 空になった場合は現在のタイムスタンプを使用
        if not safe_id:
            safe_id = f"profile_{int(time.time())}"
        
        return safe_id 