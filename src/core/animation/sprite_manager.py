import pygame
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union
import os

class SpriteManager:
    """
    スプライト管理クラス
    
    キャラクターのスプライト（画像アセット）を読み込み、管理します。
    画像リソースの管理、スケーリング、変換などの機能を提供します。
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
        
        # アセットディレクトリの設定
        self.base_assets_dir = Path.cwd() / "assets"
        self.images_dir = self.base_assets_dir / "images"
        self.characters_dir = self.images_dir / "characters"
        
        # 設定から値を読み込み
        if settings:
            # 新しいSettingsManagerクラス
            if hasattr(settings, 'get_app_config'):
                custom_assets = settings.get_app_config('animation', 'assets_dir', None)
                if custom_assets:
                    self.base_assets_dir = Path(custom_assets)
            # AppConfigやget()メソッドを持つその他の設定クラス
            elif hasattr(settings, 'get'):
                custom_assets = settings.get('animation', 'assets_dir', None)
                if custom_assets:
                    self.base_assets_dir = Path(custom_assets)
            # 古いConfigクラス
            elif hasattr(settings, 'root_dir') and hasattr(settings, 'config'):
                self.base_assets_dir = Path(settings.root_dir) / "assets"
        
        # ディレクトリパスの更新
        self.images_dir = self.base_assets_dir / "images"
        self.characters_dir = self.images_dir / "characters"
                
        # ディレクトリが存在しない場合は作成
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        
        # キャラクター画像の辞書
        self.sprites: Dict[str, Dict[str, pygame.Surface]] = {}
        
        # デフォルトの画像サイズ
        self.default_size = (300, 400)
        
        # 設定から値を読み込み
        if settings:
            # 新しいSettingsManagerクラス
            if hasattr(settings, 'get_app_config'):
                width = settings.get_app_config('animation', 'default_width', None)
                height = settings.get_app_config('animation', 'default_height', None)
                if width and height:
                    self.default_size = (width, height)
            # AppConfigやget()メソッドを持つその他の設定クラス
            elif hasattr(settings, 'get'):
                width = settings.get('animation', 'default_width', None)
                height = settings.get('animation', 'default_height', None)
                if width and height:
                    self.default_size = (width, height)
            # 古いConfigクラス
            elif hasattr(settings, 'config') and isinstance(settings.config, dict):
                if 'character' in settings.config:
                    char_config = settings.config['character']
                    if 'window' in char_config:
                        width = char_config['window'].get('width', 300)
                        height = char_config['window'].get('height', 400)
                        # キャラクターのサイズはウィンドウより少し小さく
                        self.default_size = (int(width * 0.75), int(height * 0.75))
        
        self.logger.info(f"SpriteManagerを初期化しました: {self.base_assets_dir}")
    
    def load_character(self, character_id: str, directory: Optional[Path] = None) -> bool:
        """
        キャラクターの画像セットを読み込む
        
        Args:
            character_id: キャラクターID
            directory: 画像ディレクトリ（Noneの場合はデフォルトディレクトリを使用）
            
        Returns:
            読み込みに成功した場合はTrue、失敗した場合はFalse
        """
        try:
            # キャラクターディレクトリの決定
            if directory is None:
                directory = self.characters_dir / character_id
            
            # ディレクトリが存在しない場合はエラー
            if not directory.exists() or not directory.is_dir():
                self.logger.error(f"キャラクターディレクトリが見つかりません: {directory}")
                return False
            
            # 画像を格納する辞書を初期化
            self.sprites[character_id] = {}
            
            # 画像ファイルの読み込み
            image_count = 0
            for file_path in directory.glob("*.png"):
                sprite_name = file_path.stem
                try:
                    # 画像の読み込みとスケーリング
                    sprite = pygame.image.load(str(file_path))
                    sprite = pygame.transform.scale(sprite, self.default_size)
                    
                    # スプライト辞書に登録
                    self.sprites[character_id][sprite_name] = sprite
                    image_count += 1
                    
                except Exception as e:
                    self.logger.error(f"画像の読み込みに失敗: {file_path} - {e}")
            
            if image_count == 0:
                self.logger.warning(f"キャラクター{character_id}の画像が見つかりません")
                return False
                
            self.logger.info(f"キャラクター{character_id}の画像を{image_count}個読み込みました")
            return True
            
        except Exception as e:
            self.logger.error(f"キャラクター{character_id}の読み込みに失敗: {e}")
            return False
    
    def get_sprite(self, character_id: str, sprite_name: str) -> Optional[pygame.Surface]:
        """
        指定されたキャラクターの指定されたスプライトを取得
        
        Args:
            character_id: キャラクターID
            sprite_name: スプライト名
            
        Returns:
            スプライト（pygame.Surface）、見つからない場合はNone
        """
        try:
            if character_id not in self.sprites:
                self.logger.warning(f"キャラクター{character_id}は読み込まれていません")
                return None
                
            if sprite_name not in self.sprites[character_id]:
                self.logger.warning(f"スプライト{sprite_name}はキャラクター{character_id}に存在しません")
                return None
                
            return self.sprites[character_id][sprite_name]
            
        except Exception as e:
            self.logger.error(f"スプライト取得エラー: {character_id}/{sprite_name} - {e}")
            return None
    
    def get_sprite_names(self, character_id: str) -> List[str]:
        """
        指定されたキャラクターの利用可能なスプライト名のリストを取得
        
        Args:
            character_id: キャラクターID
            
        Returns:
            スプライト名のリスト
        """
        if character_id not in self.sprites:
            return []
            
        return list(self.sprites[character_id].keys())
    
    def create_composite_sprite(self, base_sprite: pygame.Surface, 
                              overlay_sprite: pygame.Surface) -> pygame.Surface:
        """
        2つのスプライトを合成する（ベーススプライトの上にオーバーレイを配置）
        
        Args:
            base_sprite: ベーススプライト
            overlay_sprite: オーバーレイスプライト
            
        Returns:
            合成されたスプライト
        """
        try:
            # ベーススプライトのコピーを作成
            composite = base_sprite.copy()
            
            # オーバーレイをベースに描画
            composite.blit(overlay_sprite, (0, 0))
            
            return composite
            
        except Exception as e:
            self.logger.error(f"スプライト合成エラー: {e}")
            return base_sprite  # エラー時はベーススプライトを返す
    
    def get_available_characters(self) -> List[str]:
        """
        利用可能なキャラクターIDのリストを取得
        
        Returns:
            キャラクターIDのリスト
        """
        available_characters = []
        
        # 事前に読み込まれたキャラクター
        available_characters.extend(list(self.sprites.keys()))
        
        # ディレクトリ内の未読み込みキャラクター
        if self.characters_dir.exists():
            for subdir in self.characters_dir.iterdir():
                if subdir.is_dir() and subdir.name not in available_characters:
                    if any(subdir.glob("*.png")):
                        available_characters.append(subdir.name)
        
        return available_characters
    
    def unload_character(self, character_id: str) -> bool:
        """
        キャラクターのスプライトをアンロードする（メモリから解放）
        
        Args:
            character_id: キャラクターID
            
        Returns:
            アンロードに成功した場合はTrue、失敗した場合はFalse
        """
        try:
            if character_id in self.sprites:
                del self.sprites[character_id]
                self.logger.info(f"キャラクター{character_id}をアンロードしました")
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"キャラクターのアンロードに失敗: {character_id} - {e}")
            return False
    
    def resize_sprite(self, sprite: pygame.Surface, size: Tuple[int, int]) -> pygame.Surface:
        """
        スプライトをリサイズする
        
        Args:
            sprite: リサイズするスプライト
            size: 新しいサイズ（幅, 高さ）
            
        Returns:
            リサイズされたスプライト
        """
        try:
            return pygame.transform.scale(sprite, size)
        except Exception as e:
            self.logger.error(f"スプライトのリサイズに失敗: {e}")
            return sprite  # エラー時は元のスプライトを返す 