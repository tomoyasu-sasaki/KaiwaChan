import pygame
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, Any
import random
from .sprite_manager import SpriteManager

class CharacterAnimator:
    """
    キャラクターアニメーター
    
    キャラクターの表示、感情表現、口パク、アニメーションなどを管理します。
    スプライトマネージャーと連携して、キャラクターアニメーションを制御します。
    """
    
    def __init__(self, settings=None):
        """
        コンストラクタ
        
        Args:
            settings: 設定オブジェクト
                - 古いConfig、AppConfigオブジェクト
                - 新しいSettingsManagerオブジェクト
                - その他のconfig.getメソッドを持つオブジェクト
                - None（デフォルト設定を使用）
        """
        self.logger = logging.getLogger(__name__)
        self.settings = settings
        
        # Pygameの初期化（必要な場合）
        if not pygame.get_init():
            pygame.init()
        
        # スプライトマネージャーの初期化
        self.sprite_manager = SpriteManager(settings)
        
        # ウィンドウ設定
        self.width = 400
        self.height = 600
        self.background_color = (255, 255, 255)  # 白背景
        
        if settings:
            # 設定オブジェクトのタイプに応じて設定を取得
            if hasattr(settings, 'get_app_config'):
                # 新しいSettingsManagerクラス
                self.width = settings.get_app_config('animation', 'window_width', self.width)
                self.height = settings.get_app_config('animation', 'window_height', self.height)
                bg_color = settings.get_app_config('animation', 'background_color', None)
                if bg_color:
                    self.background_color = bg_color
            elif hasattr(settings, 'get'):
                # AppConfigやget()メソッドを持つその他の設定クラス
                self.width = settings.get('animation', 'window_width', self.width)
                self.height = settings.get('animation', 'window_height', self.height)
                bg_color = settings.get('animation', 'background_color', None)
                if bg_color:
                    self.background_color = bg_color
            elif hasattr(settings, 'config') and isinstance(settings.config, dict):
                # 古いConfigクラス
                if 'character' in settings.config and 'window' in settings.config['character']:
                    self.width = settings.config['character']['window'].get('width', self.width)
                    self.height = settings.config['character']['window'].get('height', self.height)
        
        # ウィンドウの初期化
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("KaiwaChan Character")
        
        # アニメーション状態
        self.current_character = "default"
        self.current_emotion = "neutral"
        self.is_talking = False
        self.mouth_state = "closed"
        self.blinking = False
        
        # アニメーションタイミング
        self.talk_start_time = 0
        self.last_mouth_change = 0
        self.mouth_change_interval = 0.1  # 100ms
        self.emotion_transition_start = 0
        self.emotion_transition_duration = 0.5  # 500ms
        self.last_blink_time = 0
        self.blink_interval = 4.0  # 平均4秒ごとに瞬き
        self.blink_duration = 0.15  # 瞬きの持続時間
        
        # フレーム管理
        self.clock = pygame.time.Clock()
        self.fps = 30
        
        # 設定から値を読み込み
        if settings:
            # 新しいSettingsManagerクラス
            if hasattr(settings, 'get_app_config'):
                self.fps = settings.get_app_config('animation', 'fps', self.fps)
                self.mouth_change_interval = settings.get_app_config('animation', 'mouth_change_interval', self.mouth_change_interval)
                self.blink_interval = settings.get_app_config('animation', 'blink_interval', self.blink_interval)
            # AppConfigやget()メソッドを持つその他の設定クラス
            elif hasattr(settings, 'get'):
                self.fps = settings.get('animation', 'fps', self.fps)
                self.mouth_change_interval = settings.get('animation', 'mouth_change_interval', self.mouth_change_interval)
                self.blink_interval = settings.get('animation', 'blink_interval', self.blink_interval)
            # 古いConfigクラス
            elif hasattr(settings, 'config') and isinstance(settings.config, dict):
                if 'character' in settings.config:
                    self.fps = settings.config['character'].get('fps', self.fps)
        
        # アニメーションループフラグ
        self.running = True
        
        # デフォルトキャラクターの読み込み
        self._load_default_character()
        
        self.logger.info("CharacterAnimatorを初期化しました")
    
    def _load_default_character(self):
        """デフォルトキャラクターの読み込み"""
        try:
            # 利用可能なキャラクターのリストを取得
            available_characters = self.sprite_manager.get_available_characters()
            
            if not available_characters:
                self.logger.warning("利用可能なキャラクターが見つかりません")
                return
            
            # 設定からデフォルトキャラクターを取得（存在しない場合は最初のキャラクター）
            default_character = "default"
            
            if self.settings:
                # 新しいSettingsManagerクラス
                if hasattr(self.settings, 'get_app_config'):
                    default_character = self.settings.get_app_config('animation', 'default_character', default_character)
                # AppConfigやget()メソッドを持つその他の設定クラス
                elif hasattr(self.settings, 'get'):
                    default_character = self.settings.get('animation', 'default_character', default_character)
                # 古いConfigクラス
                elif hasattr(self.settings, 'config') and isinstance(self.settings.config, dict):
                    if 'character' in self.settings.config:
                        default_character = self.settings.config['character'].get('default_id', default_character)
            
            # 指定されたキャラクターが存在しなければ最初のキャラクターを使用
            if default_character not in available_characters and available_characters:
                default_character = available_characters[0]
            
            # キャラクターの読み込み
            success = self.sprite_manager.load_character(default_character)
            if success:
                self.current_character = default_character
                self.logger.info(f"デフォルトキャラクター{default_character}を読み込みました")
            else:
                self.logger.error(f"デフォルトキャラクター{default_character}の読み込みに失敗しました")
        
        except Exception as e:
            self.logger.error(f"デフォルトキャラクターの読み込みエラー: {e}")
    
    def update(self):
        """
        キャラクターの状態を更新し、画面を再描画する
        メインスレッドから呼び出す必要があります
        """
        current_time = time.time()
        
        # 口パクの更新
        if self.is_talking:
            if current_time - self.last_mouth_change > self.mouth_change_interval:
                self.mouth_state = "open" if self.mouth_state == "closed" else "closed"
                self.last_mouth_change = current_time
        else:
            self.mouth_state = "closed"
        
        # 瞬きの更新
        if not self.blinking:
            # 瞬き間隔にランダム性を持たせる
            blink_threshold = self.blink_interval * (0.5 + random.random())
            if current_time - self.last_blink_time > blink_threshold:
                self.blinking = True
                self.last_blink_time = current_time
        else:
            if current_time - self.last_blink_time > self.blink_duration:
                self.blinking = False
        
        # 画面クリア
        self.screen.fill(self.background_color)
        
        # キャラクターの描画
        self._draw_character()
        
        # 画面更新
        pygame.display.flip()
    
    def _draw_character(self):
        """
        現在の状態に基づいてキャラクターを描画する
        """
        # 基本表情を取得
        base_sprite_name = self.current_emotion
        if self.blinking:
            base_sprite_name = f"{self.current_emotion}_blink"
            if not self.sprite_manager.get_sprite(self.current_character, base_sprite_name):
                base_sprite_name = "blink"  # フォールバック
                if not self.sprite_manager.get_sprite(self.current_character, base_sprite_name):
                    base_sprite_name = self.current_emotion  # さらにフォールバック
        
        base_sprite = self.sprite_manager.get_sprite(self.current_character, base_sprite_name)
        if not base_sprite:
            base_sprite = self.sprite_manager.get_sprite(self.current_character, "neutral")
            if not base_sprite:
                self.logger.error(f"基本表情スプライトが見つかりません: {self.current_character}/{base_sprite_name}")
                return
        
        # 口のスプライトを取得
        mouth_sprite_name = f"mouth_{self.mouth_state}"
        mouth_sprite = self.sprite_manager.get_sprite(self.current_character, mouth_sprite_name)
        
        # 合成画像を作成
        if mouth_sprite:
            composite = self.sprite_manager.create_composite_sprite(base_sprite, mouth_sprite)
        else:
            composite = base_sprite
        
        # 画面中央に描画
        image_rect = composite.get_rect(center=(self.width//2, self.height//2))
        self.screen.blit(composite, image_rect)
    
    def set_emotion(self, emotion: str):
        """
        キャラクターの感情を設定する
        
        Args:
            emotion: 感情名（"neutral", "happy", "sad"など）
        """
        try:
            # 指定された感情のスプライトが存在するか確認
            if self.sprite_manager.get_sprite(self.current_character, emotion):
                self.current_emotion = emotion
                self.emotion_transition_start = time.time()
                self.logger.info(f"感情を変更: {emotion}")
            else:
                self.logger.warning(f"感情{emotion}のスプライトが見つかりません")
        except Exception as e:
            self.logger.error(f"感情設定エラー: {e}")
    
    def set_emotion_from_data(self, emotion_data: Dict[str, Any]):
        """
        感情データに基づいて表情を設定
        
        Args:
            emotion_data: 感情データの辞書
        """
        try:
            emotion = emotion_data.get("emotion", "neutral")
            self.set_emotion(emotion)
        except Exception as e:
            self.logger.error(f"感情データからの感情設定エラー: {e}")
    
    def start_talking(self):
        """発話開始"""
        self.is_talking = True
        self.talk_start_time = time.time()
        self.last_mouth_change = time.time()
        self.logger.debug("発話開始")
    
    def stop_talking(self):
        """発話終了"""
        self.is_talking = False
        self.mouth_state = "closed"
        self.logger.debug("発話終了")
    
    def change_character(self, character_id: str) -> bool:
        """
        表示キャラクターを変更する
        
        Args:
            character_id: キャラクターID
            
        Returns:
            変更に成功した場合はTrue、失敗した場合はFalse
        """
        try:
            # 指定されたキャラクターが既に読み込まれているか確認
            if character_id in self.sprite_manager.sprites:
                self.current_character = character_id
                self.current_emotion = "neutral"  # 感情をリセット
                self.logger.info(f"キャラクターを変更: {character_id}")
                return True
            
            # キャラクターを読み込む
            success = self.sprite_manager.load_character(character_id)
            if success:
                self.current_character = character_id
                self.current_emotion = "neutral"  # 感情をリセット
                self.logger.info(f"キャラクターを変更: {character_id}")
                return True
            else:
                self.logger.error(f"キャラクター{character_id}の読み込みに失敗しました")
                return False
        
        except Exception as e:
            self.logger.error(f"キャラクター変更エラー: {e}")
            return False
    
    def cleanup(self):
        """
        終了処理
        """
        self.running = False
        pygame.quit()
        self.logger.info("CharacterAnimatorを終了しました") 