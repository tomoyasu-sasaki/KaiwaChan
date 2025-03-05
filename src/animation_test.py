#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
キャラクターアニメーションテストスクリプト
アニメーションモジュールの動作確認を行います
"""

import sys
import time
import pygame
import logging
import threading
from pathlib import Path

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

# 親ディレクトリへのパスを追加してインポートを有効化
sys.path.insert(0, str(Path(__file__).parent.parent))

# モジュールのインポート
from src.core.animation import CharacterAnimator, SpriteManager
from src.config import get_settings

def test_sequence(animator):
    """
    アニメーションのテストシーケンスを実行する（メインスレッドとは別スレッドで実行）
    
    Args:
        animator: CharacterAnimatorインスタンス
    """
    logger = logging.getLogger("AnimationTest")
    
    try:
        # テストシーケンスの実行
        logger.info("5秒後に表情を「happy」に変更します")
        time.sleep(5)
        animator.set_emotion("happy")
        
        logger.info("3秒後に発話を開始します")
        time.sleep(3)
        animator.start_talking()
        
        logger.info("5秒後に発話を終了します")
        time.sleep(5)
        animator.stop_talking()
        
        logger.info("3秒後に表情を「sad」に変更します")
        time.sleep(3)
        animator.set_emotion("sad")
        
        logger.info("3秒後に表情を「neutral」に戻します")
        time.sleep(3)
        animator.set_emotion("neutral")
        
        logger.info("テストが完了しました。閉じるには右上の×ボタンを押してください。")
        
    except Exception as e:
        logger.error(f"テストシーケンスでエラーが発生しました: {e}")

def main():
    """アニメーションテストのメイン関数"""
    logger = logging.getLogger("AnimationTest")
    logger.info("キャラクターアニメーションテストを開始します")
    
    # 設定マネージャーの取得
    settings = get_settings()
    
    # Pygameの初期化
    pygame.init()
    
    # アニメーションモジュールの初期化
    animator = CharacterAnimator(settings)
    
    # テストシーケンスを別スレッドで開始
    test_thread = threading.Thread(target=test_sequence, args=(animator,))
    test_thread.daemon = True
    test_thread.start()
    
    # メインスレッドでアニメーションループを実行
    running = True
    clock = pygame.time.Clock()
    
    try:
        while running:
            # イベント処理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # アニメーションの更新
            animator.update()
            
            # フレームレート制御
            clock.tick(30)
    
    except KeyboardInterrupt:
        logger.info("テストが中断されました")
    finally:
        # アニメーションの終了処理
        pygame.quit()
        logger.info("キャラクターアニメーションテストを終了します")

if __name__ == "__main__":
    main() 