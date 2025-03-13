import requests
import json
import soundfile as sf
import numpy as np
from pathlib import Path
import time
import logging
import tempfile
import os
import uuid
import torch
from typing import Optional, Dict, Union, Any
import sounddevice as sd
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from rubyinserter import add_ruby
import warnings


class TTSEngine:
    """
    テキスト音声合成（TTS）を管理するクラス
    VOICEVOXとJapanese Parler-TTSをサポートし、設定に応じて切り替えることができます。
    """

    def __init__(self, config=None):
        """コンストラクタ"""
        self.logger = logging.getLogger(__name__)
        
        # 警告とログの抑制
        self._suppress_warnings()

        # デフォルト設定
        self.base_url = "http://localhost:50021"
        self.speaker_id = 1
        self.max_retries = 3
        self.retry_delay = 1.0
        self.timeout = 30
        self.sample_rate = 24000
        self.cache_size = 100
        self.cache_enabled = True
        self.current_engine = "voicevox"
        self.description = "A female speaker with a slightly high-pitched voice delivers her words at a moderate speed with a quite monotone tone in a confined environment, resulting in a quite clear audio recording"

        # 設定をロード
        self.load_config(config)

        # キャッシュの初期化
        self.cache_dir = Path(tempfile.gettempdir()) / "kaiwachan" / "tts_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = {}

        try:
            # Parler-TTSの初期化（Parler-TTSを使用する場合のみ）
            if self.current_engine == "parler":
                self.initialize_parler_tts()
            self.logger.info("TTSエンジンの初期化が完了しました")
        except Exception as e:
            self.logger.error(f"TTSエンジンの初期化に失敗しました: {e}")
            raise

    def _suppress_warnings(self):
        """警告とログの抑制"""
        # Parler-TTSの警告を抑制
        logging.getLogger("parler_tts.modeling_parler_tts").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")
        
        # transformersの警告を抑制
        logging.getLogger("transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        # PyTorchの警告を抑制
        logging.getLogger("torch").setLevel(logging.ERROR)
        
        # Metal関連の警告を抑制（bf16関連）
        os.environ["METAL_DEBUG_ERROR_MODE"] = "0"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    def load_config(self, config):
        """設定をロードする"""
        if config is None:
            return
        
        try:
            models_config = config.get("models")
            self.current_engine = models_config.get("current_engine", self.current_engine)

            # VOICEVOX設定
            voicevox_config = models_config.get("voicevox")
            if voicevox_config:
                self.base_url = voicevox_config.get("url", self.base_url)
                self.speaker_id = voicevox_config.get("speaker_id", self.speaker_id)
                self.max_retries = voicevox_config.get("max_retries", self.max_retries)
                self.retry_delay = voicevox_config.get("retry_delay", self.retry_delay)
                self.timeout = voicevox_config.get("timeout", self.timeout)
                self.cache_size = voicevox_config.get("cache_size", self.cache_size)
                self.cache_enabled = voicevox_config.get("cache_enabled", self.cache_enabled)

            # Parler-TTS設定
            parler_config = models_config.get("parler")
            if parler_config:
                self.description = parler_config.get("description", self.description)
                self.model_path = parler_config.get("model_path", "2121-8/japanese-parler-tts-mini-bate")
                self.use_fast_tokenizer = parler_config.get("use_fast_tokenizer", True)
                self.add_prefix_space = parler_config.get("add_prefix_space", False)

        except Exception as e:
            self.logger.error(f"設定の読み込み中にエラーが発生しました: {e}")

    def initialize_parler_tts(self):
        """Parler-TTSモデルとトークナイザーの初期化"""
        try:
            self.logger.info("Parler-TTSモデルを初期化中...")
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # モデルの最適化設定
            model_kwargs = {
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,  # 半精度で実行
            }

            model = ParlerTTSForConditionalGeneration.from_pretrained(
                self.model_path,
                **model_kwargs
            ).to(self.device)
            
            # モデルを評価モードに設定
            model.eval()
            
            # JITコンパイルでモデルを最適化（可能な場合）
            if hasattr(torch, 'compile'):
                try:
                    self.model = torch.compile(model)
                    self.logger.info("モデルをJITコンパイルで最適化しました")
                except Exception as e:
                    self.logger.warning(f"モデルの最適化に失敗しました: {e}")
                    self.model = model
            else:
                self.model = model
            
            # トークナイザーの初期化（高速化オプション付き）
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=self.use_fast_tokenizer,
                add_prefix_space=self.add_prefix_space,
                model_max_length=512  # 最大長を制限してメモリ使用を最適化
            )
            
            self.logger.info("✅ Parler-TTSモデルの初期化完了")
        except Exception as e:
            self.logger.error(f"❌ Parler-TTSモデルの初期化に失敗: {e}")
            raise

    def synthesize(self, text: str, speaker_id: Optional[int] = None) -> Optional[str]:
        """
        テキストから音声を合成する

        Args:
            text: 合成するテキスト
            speaker_id: 話者ID（VOICEVOXの場合のみ使用）
            
        Returns:
            str: 生成された音声ファイルのパス
        """
        if not text or text.isspace():
            self.logger.warning("空のテキストが渡されました。音声合成をスキップします")
            return None

        try:
            
            return (
                self.synthesize_with_parler(text)
                if self.current_engine == "parler"
                else self.synthesize_with_voicevox(text, speaker_id)
            )
        except Exception as e:
            self.logger.error(f"音声合成に失敗しました: {e}")
            return None

    def synthesize_with_parler(self, text: str) -> Optional[str]:
        """Parler-TTSを使用して音声を合成"""
        try:
            # キャッシュをチェック
            if self.cache_enabled:
                cache_key = f"parler_{text}_{self.description}"
                if cache_key in self.cache:
                    self.logger.info("✅ キャッシュから音声を取得")
                    return self.cache[cache_key]

            self.logger.info(f"🔊 Parler-TTSで音声合成を開始: {text[:30]}...")

            # ルビ処理
            text = add_ruby(text)
            
            with torch.inference_mode():  # 推論モードで実行（メモリ使用量を削減）
                # 説明文のトークナイズ
                description_tokens = self.tokenizer(
                    self.description,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).input_ids.to(self.device)
                
                # テキストのトークナイズ
                text_tokens = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).input_ids.to(self.device)
                
                # 音声生成
                generation = self.model.generate(
                    input_ids=description_tokens,
                    prompt_input_ids=text_tokens
                )
                
                # CPU上でnumpy配列に変換
                audio_arr = generation.cpu().numpy().squeeze()

            # 音声ファイルの保存
            audio_path = self.cache_dir / f"tts_parler_{uuid.uuid4().hex}.wav"
            sf.write(str(audio_path), audio_arr, self.model.config.sampling_rate)

            # キャッシュに保存
            if self.cache_enabled:
                self.cache[cache_key] = str(audio_path)
                # キャッシュサイズの制限
                if len(self.cache) > self.cache_size:
                    oldest_key = next(iter(self.cache))
                    oldest_file = Path(self.cache[oldest_key])
                    if oldest_file.exists():
                        oldest_file.unlink()
                    del self.cache[oldest_key]

            self.logger.info(f"✅ Parler-TTS音声合成完了: {audio_path}")
            return str(audio_path)

        except Exception as e:
            self.logger.error(f"❌ Parler-TTS音声合成エラー: {e}")
            return None

    def synthesize_with_voicevox(self, text: str, speaker_id: Optional[int] = None) -> Optional[str]:
        """VOICEVOXを使用して音声を合成"""
        current_speaker = speaker_id or self.speaker_id

        self.logger.info(f"🔊 VOICEVOX音声合成を開始: {text[:30]}...")

        try:
            params = {"text": text, "speaker": current_speaker}
            query = requests.post(f"{self.base_url}/audio_query", params=params, timeout=self.timeout)
            query.raise_for_status()

            query_data = query.json()
            synthesis = requests.post(
                f"{self.base_url}/synthesis",
                headers={"Content-Type": "application/json"},
                params={"speaker": current_speaker},
                data=json.dumps(query_data),
                timeout=self.timeout,
            )
            synthesis.raise_for_status()

            audio_path = self.cache_dir / f"tts_voicevox_{uuid.uuid4().hex}.wav"
            with open(audio_path, "wb") as f:
                f.write(synthesis.content)

            self.logger.info(f"✅ VOICEVOX音声合成完了: {audio_path}")
            return str(audio_path)

        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ VOICEVOX音声合成エラー: {e}")
            return None

    def get_system_info(self) -> Dict[str, Any]:
        """
        システム情報を取得する
        
        Returns:
            システム情報の辞書
        """
        info = {
            'current_engine': self.current_engine,
            'cache_enabled': self.cache_enabled,
            'cache_size': self.cache_size,
        }
        
        if self.current_engine == "voicevox":
            info.update({
                'voicevox_url': self.base_url,
                'speaker_id': self.speaker_id,
            })
        elif self.current_engine == "parler":
            info.update({
                'device': getattr(self, 'device', 'not_initialized'),
                'model_loaded': hasattr(self, 'model'),
            })
            
        return info
