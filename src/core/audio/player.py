import pyaudio
import numpy as np
import wave
import logging
import sounddevice as sd
import soundfile as sf
import threading
import queue
import time


class AudioPlayer:
    """
    音声再生を管理するクラス
    """
    
    def __init__(self, config):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # 再生設定
        self.sample_rate = config.get('audio', 'sample_rate', 16000)
        self.channels = config.get('audio', 'channels', 1)
        self.chunk_size = config.get('audio', 'chunk_size', 1024)
        self.format = pyaudio.paInt16
        
        # PyAudioインスタンス
        self.audio = None
        self.stream = None
        
        # 再生状態
        self.is_playing = False
        self.play_thread = None
        self.stop_event = threading.Event()
    
    def _initialize_audio(self):
        """PyAudioインスタンスを初期化する"""
        if self.audio is None:
            self.audio = pyaudio.PyAudio()
    
    def _close_audio(self):
        """PyAudioインスタンスを閉じる"""
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
        if self.audio is not None:
            self.audio.terminate()
            self.audio = None
    
    def play_file(self, file_path, blocking=True):
        """
        音声ファイルを再生する
        
        Args:
            file_path: 再生する音声ファイルのパス
            blocking: ブロッキングモードで再生するかどうか
            
        Returns:
            bool: 成功したかどうか
        """
        try:
            if blocking:
                return self._play_file_blocking(file_path)
            else:
                return self._play_file_non_blocking(file_path)
        except Exception as e:
            self.logger.error(f"音声ファイルの再生に失敗しました: {e}")
            return False
    
    def _play_file_blocking(self, file_path):
        """ブロッキングモードで音声ファイルを再生する"""
        try:
            # soundfileを使った再生
            data, sample_rate = sf.read(file_path)
            sd.play(data, sample_rate)
            sd.wait()
            return True
        except Exception as e:
            self.logger.error(f"ブロッキングモードでの再生に失敗しました: {e}")
            # PyAudioを使ったフォールバック再生
            return self._play_file_pyaudio(file_path)
    
    def _play_file_non_blocking(self, file_path):
        """非ブロッキングモードで音声ファイルを再生する"""
        # 再生中なら停止する
        self.stop()
        
        # 再生スレッドを起動
        self.stop_event.clear()
        self.play_thread = threading.Thread(
            target=self._play_thread,
            args=(file_path,)
        )
        self.play_thread.daemon = True
        self.play_thread.start()
        
        return True
    
    def _play_thread(self, file_path):
        """再生用スレッド"""
        try:
            # soundfileを使った再生
            data, sample_rate = sf.read(file_path)
            
            self.is_playing = True
            
            # 再生イベントを作成
            finished_event = threading.Event()
            
            def callback(outdata, frames, time, status):
                if status:
                    self.logger.warning(f"Sounddevice status: {status}")
                    
                # 停止要求があれば再生終了
                if self.stop_event.is_set():
                    raise sd.CallbackStop()
            
            # 再生開始
            with sd.OutputStream(
                samplerate=sample_rate,
                channels=data.shape[1] if len(data.shape) > 1 else 1,
                callback=callback
            ):
                sd.play(data, sample_rate, blocking=False)
                
                # 再生終了まで待機
                while sd.get_stream().active and not self.stop_event.is_set():
                    time.sleep(0.1)
                    
        except Exception as e:
            self.logger.error(f"非ブロッキングモードでの再生に失敗しました: {e}")
            # PyAudioを使ったフォールバック再生
            self._play_file_pyaudio(file_path)
        finally:
            self.is_playing = False
    
    def _play_file_pyaudio(self, file_path):
        """PyAudioを使って音声ファイルを再生する"""
        self._initialize_audio()
        
        try:
            with wave.open(file_path, 'rb') as wf:
                # ストリームを開く
                stream = self.audio.open(
                    format=self.audio.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                
                self.is_playing = True
                
                # チャンクごとに再生
                chunk = wf.readframes(self.chunk_size)
                while chunk and not self.stop_event.is_set():
                    stream.write(chunk)
                    chunk = wf.readframes(self.chunk_size)
                
                # ストリームを閉じる
                stream.stop_stream()
                stream.close()
                
                return True
        except Exception as e:
            self.logger.error(f"PyAudioでの再生に失敗しました: {e}")
            return False
        finally:
            self.is_playing = False
    
    def play_array(self, audio_data, sample_rate=None, blocking=True):
        """
        音声データ（numpy配列）を再生する
        
        Args:
            audio_data: 再生する音声データ（numpy.ndarray）
            sample_rate: サンプリングレート（Noneの場合はデフォルト値を使用）
            blocking: ブロッキングモードで再生するかどうか
            
        Returns:
            bool: 成功したかどうか
        """
        sample_rate = sample_rate or self.sample_rate
        
        try:
            if blocking:
                sd.play(audio_data, sample_rate)
                sd.wait()
            else:
                # 再生中なら停止する
                self.stop()
                
                # 非ブロッキングモードで再生
                self.is_playing = True
                sd.play(audio_data, sample_rate, blocking=False)
                
                # 再生スレッドを起動して終了を待機
                self.stop_event.clear()
                self.play_thread = threading.Thread(
                    target=self._wait_playback_thread
                )
                self.play_thread.daemon = True
                self.play_thread.start()
                
            return True
        except Exception as e:
            self.logger.error(f"音声データの再生に失敗しました: {e}")
            return False
    
    def _wait_playback_thread(self):
        """再生終了待機スレッド"""
        try:
            # 再生終了まで待機
            while sd.get_stream().active and not self.stop_event.is_set():
                time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"再生待機中にエラーが発生しました: {e}")
        finally:
            self.is_playing = False
    
    def stop(self):
        """再生を停止する"""
        if self.is_playing:
            self.stop_event.set()
            
            # sounddeviceの再生を停止
            try:
                sd.stop()
            except Exception:
                pass
            
            # 再生スレッドの終了を待機
            if self.play_thread is not None and self.play_thread.is_alive():
                self.play_thread.join(timeout=1.0)
                
            self.is_playing = False
            self.logger.info("再生を停止しました")
    
    def close(self):
        """リソースを解放する"""
        self.stop()
        self._close_audio() 