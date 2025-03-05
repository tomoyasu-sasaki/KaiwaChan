from llama_cpp import Llama
from pathlib import Path
import logging
import time
import json
from typing import List, Dict, Optional, Any

class DialogueEngine:
    """
    対話生成エンジン
    
    ユーザーの入力に対する応答を生成するクラスです。
    LLMを使用して自然な会話応答を生成します。
    """
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.model = None
        self.conversation_history = []
        self.max_history = 10
        
        # モデルの初期化
        if config:
            self._initialize_model()
    
    def _initialize_model(self) -> bool:
        """
        LLMモデルを初期化する
        
        Returns:
            bool: 初期化に成功したかどうか
        """
        try:
            model_path = self.config.get_model_path("llama")
            if not model_path.exists():
                self.logger.error(f"モデルファイルが見つかりません: {model_path}")
                raise FileNotFoundError(
                    f"Graniteモデルが見つかりません: {model_path}\n"
                    "1. https://huggingface.co/lmstudio-community/granite-3.1-8b-instruct-GGUF からモデルをダウンロード\n"
                    f"2. {model_path}に配置してください"
                )
            
            self.logger.info(f"LLMモデルをロード中: {model_path}")
            start_time = time.time()
            
            # モデル設定を取得
            model_config = self.config.get_app_config("models", "llm", {})
            n_threads = model_config.get("n_threads", 8)
            n_batch = model_config.get("n_batch", 512)
            n_ctx = model_config.get("n_ctx", 2048)
            
            # モデルをロード
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=n_ctx,
                n_gpu_layers=-1,  # 利用可能なすべてのGPUレイヤーを使用
                n_threads=n_threads,
                n_batch=n_batch,
                seed=42
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
            
            # プロンプトの生成
            system_prompt = self._generate_system_prompt(character_info)
            
            self.logger.info("🤖 応答を生成中...")
            
            # モデル設定を取得
            llm_config = self.config.get_app_config("models", "llm", {})
            max_tokens = llm_config.get("max_tokens", 128)
            
            # 会話履歴の準備
            messages = [{"role": "system", "content": system_prompt}]
            
            # 過去の会話を追加
            for h in self.conversation_history:
                messages.append({"role": h["role"], "content": h["content"]})
            
            # 新しいユーザー入力を追加
            messages.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "user", "content": user_input})
            
            # 応答の生成
            response = self.model.create_chat_completion(
                messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.1,
                stop=["<|end_of_text|>", "<|start_of_role|>"]
            )
            
            # 応答の後処理
            assistant_response = response["choices"][0]["message"]["content"]
            
            # 会話履歴に追加
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            # 会話履歴が長すぎる場合は古いものを削除
            if len(self.conversation_history) > self.max_history * 2:
                # システムプロンプト以外の古い会話を削除
                self.conversation_history = self.conversation_history[-self.max_history * 2:]
            
            return assistant_response
            
        except Exception as e:
            self.logger.error(f"応答生成中にエラーが発生: {str(e)}")
            return "申し訳ありません。応答の生成中にエラーが発生しました。"
    
    def _generate_system_prompt(self, character_info: Optional[Dict] = None) -> str:
        """
        システムプロンプトを生成する
        
        Args:
            character_info: キャラクター情報
            
        Returns:
            str: 生成されたシステムプロンプト
        """
        # キャラクター設定
        character_name = "AI"
        character_personality = "親しみやすく丁寧"
        character_speaking_style = "簡潔で自然な日本語"
        
        if character_info:
            character_name = character_info.get("name", character_name)
            character_personality = character_info.get("personality", character_personality)
            character_speaking_style = character_info.get("speaking_style", character_speaking_style)
        
        # システムプロンプト
        system_prompt = f"""あなたは{character_name}という名前の{character_personality}なAIアシスタントです。
{character_speaking_style}で応答してください。
質問に対して簡潔に回答し、必要以上の説明は避けてください。"""
        
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
        self.conversation_history.append({
            "user": user_input,
            "assistant": assistant_response,
            "timestamp": time.time()
        })
        
        # 履歴の長さを制限
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
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