from llama_cpp import Llama
from pathlib import Path
import logging

class DialogueEngine:
    def __init__(self, config):
        model_path = config.get_model_path("llama")
        if not model_path.exists():
            raise FileNotFoundError(
                f"Graniteモデルが見つかりません: {model_path}\n"
                "1. https://huggingface.co/lmstudio-community/granite-3.1-8b-instruct-GGUF からモデルをダウンロード\n"
                f"2. {model_path}に配置してください"
            )
            
        model_config = config.config["models"]["llama"]
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            n_gpu_layers=-1,
            n_threads=model_config.get("n_threads", 8),
            n_batch=model_config.get("n_batch", 512),
            seed=42
        )
        self.context = []
        self.logger = logging.getLogger(__name__)
        
    def generate_response(self, user_input):
        self.logger.info(f"👤 ユーザー入力: {user_input}")
        self.logger.info("🤖 応答を生成中...")
        
        prompt = self._build_prompt(user_input)
        response = self.model(
            prompt,
            max_tokens=128,
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            stream=False
        )
        
        result = response["choices"][0]["text"].strip()
        # "Assistant:"などのプレフィックスを削除
        result = self._clean_response(result)
        
        self.logger.info(f"✅ 生成された応答: {result}")
        return result
        
    def _build_prompt(self, user_input):
        # システムプロンプトを追加
        system_prompt = """あなたは親しみやすいAIアシスタントです。
簡潔で自然な日本語で応答してください。質問の回答のみを出力してください。"""
        
        # 直接の応答のみを要求
        prompt = f"{system_prompt}\n\n質問: {user_input}\n回答:"
        return prompt
        
    def _clean_response(self, text):
        # 不要なプレフィックスを削除
        prefixes = ["Assistant:", "回答:", "\n"]
        result = text
        for prefix in prefixes:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        
        # 質問や回答のパターンで分割し、最初の応答のみを取得
        patterns = [
            "\n質問:", "\n回答:",
            "User:", "Assistant:",
            "質問:", "回答:"
        ]
        
        for pattern in patterns:
            if pattern in result:
                result = result.split(pattern)[0].strip()
        
        return result 