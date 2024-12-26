import pygame
import time
from pathlib import Path
import random
from ..utils.logger import Logger

class CharacterAnimation:
    def __init__(self, config):
        pygame.init()
        self.config = config
        self.logger = Logger(config)
        
        # ウィンドウ設定
        self.width = 400
        self.height = 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("KaiwaChan Character")
        
        # 画像の読み込み
        self.assets_dir = Path(self.config.root_dir) / "assets" / "images"
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        
        # 表情画像の読み込み
        self.expressions = {}
        self._load_expressions()
        
        # 現在の表情状態
        self.current_expression = "neutral"
        self.is_talking = False
        self.talk_start_time = 0
        self.mouth_state = "closed"
        self.mouth_change_interval = 0.1  # 100ms
        self.last_mouth_change = 0
        
        # 感情状態
        self.current_emotion = "neutral"
        self.emotion_transition_start = 0
        self.emotion_transition_duration = 0.5  # 500ms
        
        # フレーム管理
        self.clock = pygame.time.Clock()
        self.fps = 30
        
    def _load_expressions(self):
        """表情画像の読み込み"""
        expression_files = {
            "neutral": "neutral.png",
            "happy": "happy.png",
            "sad": "sad.png",
            "mouth_open": "mouth_open.png",
            "mouth_closed": "mouth_closed.png"
        }
        
        for name, filename in expression_files.items():
            path = self.assets_dir / filename
            if path.exists():
                self.expressions[name] = pygame.image.load(str(path))
                self.expressions[name] = pygame.transform.scale(
                    self.expressions[name], 
                    (300, 400)  # キャラクターのサイズ
                )
            else:
                self.logger.warning(f"画像が見つかりません: {path}")
                
    def update(self):
        current_time = time.time()
        
        # 口パクの更新
        if self.is_talking:
            if current_time - self.last_mouth_change > self.mouth_change_interval:
                self.mouth_state = "open" if self.mouth_state == "closed" else "closed"
                self.last_mouth_change = current_time
        else:
            self.mouth_state = "closed"
        
        # 画面の更新
        self.screen.fill((255, 255, 255))
        
        # 基本表情の描画
        base_expression = self.expressions.get(self.current_emotion)
        if base_expression:
            image_rect = base_expression.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(base_expression, image_rect)
        
        # 口の描画
        mouth_image = self.expressions.get(f"mouth_{self.mouth_state}")
        if mouth_image:
            mouth_rect = mouth_image.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(mouth_image, mouth_rect)
        
        pygame.display.flip()
        self.clock.tick(self.fps)
    
    def set_emotion(self, emotion_data):
        """感情データに基づいて表情を設定"""
        try:
            emotion = emotion_data.get("emotion", "neutral")
            if emotion in self.expressions:
                self.current_emotion = emotion
                self.emotion_transition_start = time.time()
                self.logger.info(f"感情を変更: {emotion}")
        except Exception as e:
            self.logger.error(f"感情設定エラー: {str(e)}")
    
    def start_talking(self):
        """発話開始"""
        self.is_talking = True
        self.talk_start_time = time.time()
        self.last_mouth_change = time.time()
    
    def stop_talking(self):
        """発話終了"""
        self.is_talking = False
        self.mouth_state = "closed"
        
    def cleanup(self):
        """終了処理"""
        pygame.quit() 