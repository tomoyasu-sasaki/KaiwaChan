from llama_cpp import Llama
from pathlib import Path

class DialogueEngine:
    def __init__(self, config):
        model_path = config.get_model_path("llama")
        if not model_path.exists():
            raise FileNotFoundError(
                f"LLaMAモデルが見つかりません: {model_path}\n"
                "1. https://huggingface.co/からllama-2-7b-chat.ggufをダウンロード\n"
                f"2. {model_path}に配置してください"
            )
            
        self.model = Llama(
            model_path=str(model_path),
            n_ctx=2048
        )
        self.context = []
        
    def generate_response(self, user_input):
        prompt = self._build_prompt(user_input)
        response = self.model(
            prompt,
            max_tokens=512,
            temperature=0.7
        )
        return response["choices"][0]["text"]
        
    def _build_prompt(self, user_input):
        # コンテキストを含めたプロンプトの構築
        return f"User: {user_input}\nAssistant:" 