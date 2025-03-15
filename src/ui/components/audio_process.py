from PyQt6.QtCore import QThread, pyqtSignal
import traceback


class AudioProcessThread(QThread):
    """
    音声処理を行うスレッド
    
    会話の流れ：
    1. 音声入力の取得
    2. 音声認識（STT）
    3. 対話生成
    4. 音声合成（TTS）
    5. キャラクターアニメーション
    """
    finished = pyqtSignal(str, str)  # (response_text, audio_path)
    error = pyqtSignal(str)
    
    def __init__(self, speech_recognizer, dialogue_engine, tts_engine, character_animator=None):
        super().__init__()
        self.speech_recognizer = speech_recognizer
        self.dialogue_engine = dialogue_engine
        self.tts_engine = tts_engine
        self.character_animator = character_animator
        
    def run(self):
        """スレッドの実行：音声入力から応答生成までの一連の処理"""
        try:
            # 音声入力と認識
            audio = self.speech_recognizer.record_audio()
            text = self.speech_recognizer.transcribe(audio)
            
            # アニメーションを表示している場合は口パク開始
            if self.character_animator:
                self.character_animator.start_talking()
                
            # 応答生成
            response = self.dialogue_engine.generate_response(text)
            
            # 音声合成
            audio_path = self.tts_engine.synthesize(response)
            
            if not audio_path:
                raise Exception("音声合成に失敗しました")
            
            # 応答テキストと音声ファイルのパスをUI側に渡す
            self.finished.emit(response, audio_path)
            
        except Exception as e:
            error_msg = f"エラー発生: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg) 