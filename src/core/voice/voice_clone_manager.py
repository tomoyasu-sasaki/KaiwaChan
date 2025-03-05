import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import torch
import traceback
from TTS.api import TTS

from .feature_extractor import FeatureExtractor
from .profile_manager import VoiceProfileManager
from ..tts import TTSEngine

class VoiceCloneManager:
    """
    音声クローン管理クラス
    
    音声プロファイルの管理、特徴抽出、音声合成を統合的に管理します。
    """
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # TTS設定（デフォルト値）
        self.use_japanese_tts = True
        self.initialize_tts_on_demand = True
        if config:
            self.use_japanese_tts = config.get('voice_clone', 'use_japanese_tts', self.use_japanese_tts)
            self.initialize_tts_on_demand = config.get('voice_clone', 'initialize_tts_on_demand', self.initialize_tts_on_demand)
        
        # コンポーネント初期化
        self.feature_extractor = FeatureExtractor(config)
        self.profile_manager = VoiceProfileManager(config)
        
        # TTSエンジンは必要に応じて初期化
        self._tts_engine = None
        self._tts_initialized = False
        
        # 遅延初期化しない場合はすぐにTTSエンジンを初期化
        if not self.initialize_tts_on_demand:
            self._initialize_tts()
        
        self.logger.info("VoiceCloneManagerを初期化しました")
    
    def _initialize_tts(self) -> bool:
        """
        TTS（音声合成）エンジンを初期化する
        
        Returns:
            初期化に成功した場合はTrue、失敗した場合はFalse
        """
        if self._tts_initialized:
            return True
            
        try:
            self.logger.info("TTSエンジンを初期化しています...")
            self._tts_engine = TTSEngine(self.config)
            self._tts_initialized = True
            self.logger.info("TTSエンジンの初期化が完了しました")
            return True
        except Exception as e:
            self.logger.error(f"TTSエンジンの初期化に失敗しました: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_profile_ids(self) -> List[str]:
        """
        利用可能なプロファイルIDのリストを取得
        
        Returns:
            プロファイルIDのリスト
        """
        return self.profile_manager.get_profile_ids()
    
    def get_profile_names(self) -> Dict[str, str]:
        """
        プロファイルID → 名前のマッピングを取得
        
        Returns:
            プロファイルID → 名前の辞書
        """
        return self.profile_manager.get_profile_names()
    
    def get_profile(self, profile_id: str) -> Optional[Dict]:
        """
        指定されたIDのプロファイルを取得
        
        Args:
            profile_id: プロファイルID
            
        Returns:
            プロファイル情報の辞書、存在しない場合はNone
        """
        return self.profile_manager.get_profile(profile_id)
    
    def extract_voice_features(self, audio_input) -> Optional[Dict]:
        """
        音声ファイルから特徴量を抽出する
        
        Args:
            audio_input: 音声ファイルのパスまたは前処理済み音声データ
            
        Returns:
            抽出された特徴量の辞書、失敗した場合はNone
        """
        return self.feature_extractor.extract_features(audio_input)
    
    def create_profile(self, audio_files: List[str], profile_name: str, progress_callback=None) -> Optional[str]:
        """
        複数の音声ファイルから新しい音声プロファイルを作成する
        
        Args:
            audio_files: 音声ファイルのパスのリスト
            profile_name: プロファイルの名前
            progress_callback: 進捗状況を通知するコールバック関数
            
        Returns:
            作成されたプロファイルのID、失敗した場合はNone
        """
        if not audio_files:
            self.logger.error("音声ファイルが指定されていません")
            return None
            
        try:
            self.logger.info(f"音声プロファイル作成開始: {profile_name}")
            self.logger.info(f"音声ファイル数: {len(audio_files)}")
            
            # 特徴量抽出
            features_list = []
            total_steps = len(audio_files) * 3  # 3ステップ: 読み込み、特徴抽出、統合
            current_step = 0
            
            for i, audio_file in enumerate(audio_files):
                # 進捗状況を通知
                if progress_callback:
                    if not progress_callback(current_step, total_steps, f"音声ファイルを処理中: {Path(audio_file).name}"):
                        self.logger.info("ユーザーによってプロファイル作成がキャンセルされました")
                        return None
                    current_step += 1
                    
                self.logger.info(f"音声特徴量を抽出中: {Path(audio_file).name}")
                features = self.extract_voice_features(audio_file)
                
                if features:
                    features_list.append(features)
                    self.logger.info(f"特徴量抽出成功: {Path(audio_file).name}")
                else:
                    self.logger.warning(f"特徴量抽出失敗: {Path(audio_file).name} - スキップします")
                    
                # 特徴抽出完了の進捗を通知
                if progress_callback:
                    if not progress_callback(current_step, total_steps, f"特徴抽出完了: {Path(audio_file).name}"):
                        self.logger.info("ユーザーによってプロファイル作成がキャンセルされました")
                        return None
                    current_step += 1
            
            if not features_list:
                self.logger.error("有効な特徴量がありません。プロファイル作成を中止します")
                return None
                
            # 特徴量を集約
            if progress_callback:
                if not progress_callback(current_step, total_steps, f"特徴量を集約中... ({len(features_list)}個のサンプル)"):
                    self.logger.info("ユーザーによってプロファイル作成がキャンセルされました")
                    return None
                
            self.logger.info(f"{len(features_list)}個のサンプルから特徴量を集約します")
            aggregated_features = self.feature_extractor.aggregate_features(features_list)
            
            if not aggregated_features:
                self.logger.error("特徴量の集約に失敗しました")
                return None
                
            # プロファイル情報を作成
            profile_id = f"{profile_name}_{int(time.time())}"
            profile_data = {
                'name': profile_name,
                'created_at': time.time(),
                'sample_count': len(features_list),
                'sample_files': [Path(f).name for f in audio_files],
                **aggregated_features
            }
            
            # 最終進捗通知
            if progress_callback:
                if not progress_callback(total_steps - 1, total_steps, "プロファイルを保存中..."):
                    self.logger.info("ユーザーによってプロファイル作成がキャンセルされました")
                    return None
            
            # プロファイルを保存
            if self.profile_manager.save_profile(profile_id, profile_data):
                self.logger.info(f"プロファイルを保存しました: {profile_id}")
                
                # 完了通知
                if progress_callback:
                    progress_callback(total_steps, total_steps, "プロファイル作成が完了しました")
                    
                return profile_id
            else:
                self.logger.error(f"プロファイルの保存に失敗しました: {profile_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"プロファイル作成中にエラーが発生: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def delete_profile(self, profile_id: str) -> bool:
        """
        プロファイルを削除する
        
        Args:
            profile_id: 削除するプロファイルのID
            
        Returns:
            削除に成功した場合はTrue、失敗した場合はFalse
        """
        return self.profile_manager.delete_profile(profile_id)
    
    def rename_profile(self, profile_id: str, new_name: str) -> bool:
        """
        プロファイルの名前を変更する
        
        Args:
            profile_id: プロファイルID
            new_name: 新しい名前
            
        Returns:
            変更に成功した場合はTrue、失敗した場合はFalse
        """
        return self.profile_manager.rename_profile(profile_id, new_name)
    
    def synthesize_speech(self, text: str, profile_id: str = None, speed: float = 1.0) -> Optional[np.ndarray]:
        """
        テキストから音声を合成する
        
        Args:
            text: 合成するテキスト
            profile_id: 使用するプロファイルのID（Noneの場合はデフォルト音声）
            speed: 音声の速度（1.0が標準）
            
        Returns:
            合成された音声データ（numpy配列）、失敗した場合はNone
        """
        # TTSエンジンが初期化されていない場合は初期化
        if not self._tts_initialized and not self._initialize_tts():
            return None
            
        try:
            # プロファイル情報を取得（指定がない場合はNone）
            profile = None
            if profile_id:
                profile = self.profile_manager.get_profile(profile_id)
                if not profile:
                    self.logger.warning(f"指定されたプロファイルが見つかりません: {profile_id}")
            
            # テキストから音声を合成
            return self._tts_engine.synthesize_to_array(
                text, 
                speaker_id=None  # プロファイルベースの合成はまだ実装されていません
            )
            
        except Exception as e:
            self.logger.error(f"音声合成中にエラーが発生: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> Optional[str]:
        """
        テキストから音声を合成し、ファイルパスを返す
        MainWindowクラスとの互換性のためのメソッド
        
        Args:
            text: 合成するテキスト
            speaker_id: 話者ID（指定しない場合はデフォルト値を使用）
            
        Returns:
            str: 生成された音声ファイルのパス、失敗時はNone
        """
        # TTSエンジンが初期化されていない場合は初期化
        if not self._tts_initialized and not self._initialize_tts():
            return None
            
        try:
            # TTSエンジンのsynthesizeメソッドを呼び出す
            return self._tts_engine.synthesize(text, speaker_id)
        except Exception as e:
            self.logger.error(f"音声合成中にエラーが発生: {e}")
            self.logger.error(traceback.format_exc())
            return None
    
    def play_speech(self, text: str, profile_id: str = None, speed: float = 1.0) -> bool:
        """
        テキストを音声に変換して再生する
        
        Args:
            text: 合成するテキスト
            profile_id: 使用するプロファイルのID（Noneの場合はデフォルト音声）
            speed: 音声の速度（1.0が標準）
            
        Returns:
            再生に成功した場合はTrue、失敗した場合はFalse
        """
        # TTSエンジンが初期化されていない場合は初期化
        if not self._tts_initialized and not self._initialize_tts():
            return False
            
        try:
            # 音声を合成
            waveform = self.synthesize_speech(text, profile_id, speed)
            if waveform is None:
                return False
                
            # 音声を再生
            return self._tts_engine.play_audio(waveform)
            
        except Exception as e:
            self.logger.error(f"音声再生中にエラーが発生: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        システム情報を取得する
        
        Returns:
            システム情報の辞書
        """
        info = {
            'profile_count': len(self.get_profile_ids()),
            'tts_initialized': self._tts_initialized,
            'use_japanese_tts': self.use_japanese_tts,
            'device': str(torch.device('cuda' if torch.cuda.is_available() else 'cpu')),
            'has_cuda': torch.cuda.is_available(),
        }
        
        # GPUメモリ情報（CUDAが利用可能な場合のみ）
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            info['cuda_current_device'] = torch.cuda.current_device()
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            
            # 利用可能なGPUメモリ
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                info['cuda_memory'] = {
                    'free_mb': free_mem / (1024 * 1024),
                    'total_mb': total_mem / (1024 * 1024),
                    'used_mb': (total_mem - free_mem) / (1024 * 1024),
                    'usage_percent': (total_mem - free_mem) / total_mem * 100
                }
            except:
                info['cuda_memory'] = 'Unknown'
        
        return info 

    @property
    def voice_profiles(self):
        """
        音声プロファイル辞書へのアクセスを提供するプロパティ
        
        Returns:
            Dict: 音声プロファイル辞書
        """
        return self.profile_manager.profiles 