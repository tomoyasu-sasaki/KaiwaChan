from llama_cpp import Llama
from pathlib import Path
import logging
import time
import json
from typing import List, Dict, Optional, Any
from src.config.settings_manager import SettingsManager
import os
import warnings

class DialogueEngine:
    """
    対話生成エンジン
    
    ユーザーの入力に対する応答を生成するクラスです。
    LLMを使用して自然な会話応答を生成します。
    """
    
    def __init__(self, settings_manager: SettingsManager):
        """
        初期化
        
        Args:
            settings_manager: 設定マネージャー
        """
        self.settings_manager = settings_manager
        self.logger = logging.getLogger(__name__)
        
        # 警告の抑制
        self._suppress_warnings()
        
        self.model = None
        self.conversation_history = []
        self.max_history = 10
        
        # モデルのパスを取得
        model_path = self.settings_manager.get_model_path("llm")
        if not model_path.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

        # モデルの設定を取得
        model_config = self.settings_manager.get_app_config("models", "llm", {})
        self.n_threads = model_config.get("n_threads", 8)
        self.n_batch = model_config.get("n_batch", 512)
        self.max_tokens = model_config.get("max_tokens", 128)
        
        # モデルの初期化
        self._initialize_model()
    
    def _suppress_warnings(self):
        """警告とログの抑制"""
        # LLaMAの警告を抑制
        logging.getLogger("llama_cpp").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # Metal関連の警告を抑制（bf16関連）
        os.environ["METAL_DEBUG_ERROR_MODE"] = "0"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    def _initialize_model(self) -> bool:
        """
        LLMモデルを初期化する
        
        Returns:
            bool: 初期化に成功したかどうか
        """
        try:
            # モデルのパスを取得
            model_path = self.settings_manager.get_model_path("llm")
            self.logger.info(f"LLMモデルをロード中: {model_path}")
            start_time = time.time()
            
            # モデルをロード
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=2048,
                n_gpu_layers=-1,  # 利用可能なすべてのGPUレイヤーを使用
                n_threads=self.n_threads,
                n_batch=self.n_batch,
                seed=42,
                verbose=False  # ログ出力を抑制
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"LLMモデルのロードが完了しました (所要時間: {load_time:.2f}秒)")
            return True
            
        except Exception as e:
            self.logger.error(f"LLMモデルの初期化に失敗: {e}")
            return False
    
    def generate_response(self, user_input: str, character_info: Optional[Dict] = None) -> str:
        """
        ユーザー入力に対する応答を生成する
        
        Args:
            user_input: ユーザーの入力テキスト
            character_info: キャラクター情報（オプション）
            
        Returns:
            生成された応答テキスト
        """
        try:
            
            self.logger.info(f"👤 ユーザー入力: {user_input}")
            
            # モデルが初期化されていない場合は初期化を試みる
            if self.model is None:
                if not self._initialize_model():
                    return "申し訳ありません。対話エンジンの初期化に失敗しました。"
            
            # 会話履歴が空の場合のみシステムプロンプトを生成
            messages = []
            if not self.conversation_history:
                system_prompt = self._generate_system_prompt(character_info)
                messages.append({"role": "system", "content": system_prompt})
                self.conversation_history.append({"role": "system", "content": system_prompt})
            else:
                # システムプロンプトを履歴から取得
                system_message = next((m for m in self.conversation_history if m["role"] == "system"), None)
                if system_message:
                    messages.append(system_message)
            
            self.logger.info("🤖 応答を生成中...")
            
            # 過去の会話を追加（システムプロンプト以外）
            for h in self.conversation_history:
                if h["role"] != "system":
                    messages.append(h)
            
            # 新しいユーザー入力を追加
            messages.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 応答の生成
            response = self.model.create_chat_completion(
                messages,
                max_tokens=self.max_tokens,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<|end_of_text|>", "<|start_of_role|>"]
            )
            
            # 応答の後処理
            assistant_response = response["choices"][0]["message"]["content"]
            
            # 会話履歴に追加
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # 会話履歴が長すぎる場合は古いものを削除（システムプロンプトは保持）
            if len(self.conversation_history) > self.max_history * 2 + 1:  # +1 はシステムプロンプト用
                system_message = next(m for m in self.conversation_history if m["role"] == "system")
                self.conversation_history = [system_message] + self.conversation_history[-(self.max_history * 2):]
            
            return assistant_response
            
        except Exception as e:
            self.logger.error(f"応答生成中にエラーが発生: {str(e)}")
            return "申し訳ありません。応答の生成中にエラーが発生しました。"
    
    def _generate_system_prompt(self, character_info: Optional[Dict] = None) -> str:
        """
        システムプロンプトを生成する
        
        Args:
            character_info: キャラクター情報（オプション）
            
        Returns:
            str: 生成されたシステムプロンプト
        """
        # キャラクター設定をconfig.ymlから取得
        default_character = self.settings_manager.get_app_config("behavior", "character", {})
        
        # デフォルト値の設定（設定ファイルの値を優先）
        character_name = default_character.get("name", "会話ちゃん")
        character_personality = default_character.get("personality", "お嬢様")
        character_speaking_style = default_character.get("speaking_style", "丁寧で優雅な日本語")
        
        # character_infoで上書き
        if character_info:
            character_name = character_info.get("name", character_name)
            character_personality = character_info.get("personality", character_personality)
            character_speaking_style = character_info.get("speaking_style", character_speaking_style)
        
        # プロンプトの生成
        system_prompt = f"""あなたは{character_name}という名前の{character_personality}なAIアシスタントです。
{character_speaking_style}で応答してください。
簡潔に回答し、対話が継続するように意識してください。必要以上の説明は避けてください。

以下の制約に従ってください：
1. 常に日本語で応答すること
2. 一度の応答は3文以内に収めること
3. 相手の発言に共感を示しながら会話を進めること
4. 質問されていない場合でも、会話を継続させるための質問を1つ含めること"""
        
        return system_prompt
    
    def _clean_response(self, text: str) -> str:
        """
        生成された応答をクリーンアップする
        
        Args:
            text: 生のレスポンステキスト
            
        Returns:
            str: クリーンアップされたテキスト
        """
        # 不要なプレフィックスを削除
        prefixes = ["Assistant:", "AI:", "回答:", "\n"]
        result = text
        
        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        
        # 質問や回答のパターンで分割し、最初の応答のみを取得
        cut_patterns = [
            "\nユーザー:", "\n質問:", "\n回答:",
            "User:", "Assistant:",
            "ユーザー:", "AI:",
            "質問:", "回答:"
        ]
        
        for pattern in cut_patterns:
            if pattern in result:
                result = result.split(pattern)[0].strip()
        
        return result
    
    def _update_history(self, user_input: str, assistant_response: str) -> None:
        """
        会話履歴を更新する
        
        Args:
            user_input: ユーザー入力
            assistant_response: アシスタントの応答
        """
        # 新しい対話ターンを追加
        self.conversation_history.append({"role": "user", "content": user_input})
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        
        # 履歴の長さを制限（システムプロンプトは保持）
        if len(self.conversation_history) > self.max_history * 2 + 1:
            system_message = next(m for m in self.conversation_history if m["role"] == "system")
            self.conversation_history = [system_message] + self.conversation_history[-(self.max_history * 2):]
    
    def clear_history(self) -> None:
        """会話履歴をクリアする"""
        self.conversation_history = []
        self.logger.info("会話履歴をクリアしました")
        
    def set_max_history(self, max_history: int) -> None:
        """
        保持する会話履歴の最大ターン数を設定する
        
        Args:
            max_history: 最大ターン数
        """
        if max_history < 1:
            max_history = 1
        self.max_history = max_history
        
        # 現在の履歴が長すぎる場合は切り詰める
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def save_conversation(self, file_path: str) -> bool:
        """
        会話履歴をJSONファイルに保存する
        
        Args:
            file_path: 保存先のファイルパス
            
        Returns:
            bool: 保存に成功したかどうか
        """
        try:
            if not self.conversation_history:
                self.logger.warning("保存する会話履歴がありません")
                return False
                
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
                
            self.logger.info(f"会話履歴を保存しました: {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"会話履歴の保存に失敗: {e}")
            return False
    
    def load_conversation(self, file_path: str) -> bool:
        """
        会話履歴をJSONファイルから読み込む
        
        Args:
            file_path: 読み込むファイルパス
            
        Returns:
            bool: 読み込みに成功したかどうか
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
                
            if isinstance(history, list):
                self.conversation_history = history
                
                # 履歴が長すぎる場合は切り詰める
                if len(self.conversation_history) > self.max_history:
                    self.conversation_history = self.conversation_history[-self.max_history:]
                    
                self.logger.info(f"会話履歴を読み込みました ({len(self.conversation_history)}ターン): {file_path}")
                return True
            else:
                self.logger.error("無効な会話履歴形式です")
                return False
                
        except Exception as e:
            self.logger.error(f"会話履歴の読み込みに失敗: {e}")
            return False 