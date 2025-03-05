import pyaudio
import wave
import numpy as np
import threading
import time
import logging
import os
from pathlib import Path
import tempfile


class AudioRecorder:
    """
    音声録音を管理するクラス
    マイクからの音声入力を録音し、ファイルに保存する機能を提供
    """
    
    def __init__(self, config=None):
        """
        コンストラクタ
        
        Args:
            config: 設定オブジェクト
        """
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # デフォルト設定
        self.sample_rate = 16000
        self.channels = 1
        self.chunk_size = 1024
        self.format = pyaudio.paInt16
        self.device_index = None
        
        # 設定から値を取得
        if config:
            self.sample_rate = config.get("audio", "sample_rate", self.sample_rate)
            self.channels = config.get("audio", "channels", self.channels)
            self.chunk_size = config.get("audio", "chunk_size", self.chunk_size)
            self.device_index = config.get("audio", "device", self.device_index)
            
        # PyAudioインスタンス
        self.audio = None
        
        # 録音状態
        self.is_recording = False
        self.recording_thread = None
        self.frames = []
        self.start_time = 0
        self.stop_time = 0
        
        # コールバック
        self.on_recording_started = None
        self.on_recording_progress = None
        self.on_recording_stopped = None
        
        # デバイス情報のキャッシュ
        self._device_info_cache = None
    
    def initialize(self):
        """
        録音デバイスを初期化する
        
        Returns:
            bool: 初期化に成功したかどうか
        """
        try:
            if self.audio is None:
                self.audio = pyaudio.PyAudio()
                self.logger.debug("PyAudioを初期化しました")
            return True
        except Exception as e:
            self.logger.error(f"PyAudioの初期化に失敗: {e}")
            return False
    
    def shutdown(self):
        """
        録音デバイスを終了する
        """
        try:
            if self.is_recording:
                self.stop_recording()
                
            if self.audio:
                self.audio.terminate()
                self.audio = None
                self.logger.debug("PyAudioを終了しました")
        except Exception as e:
            self.logger.error(f"PyAudioの終了に失敗: {e}")
    
    def get_available_devices(self):
        """
        利用可能な入力デバイスの一覧を取得する
        
        Returns:
            list: 入力デバイス情報のリスト
        """
        if self._device_info_cache:
            return self._device_info_cache
            
        devices = []
        
        try:
            if not self.initialize():
                return []
                
            info = self.audio.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            for i in range(num_devices):
                device_info = self.audio.get_device_info_by_index(i)
                
                # 入力デバイスのみを抽出
                if device_info.get('maxInputChannels') > 0:
                    devices.append({
                        'index': device_info.get('index'),
                        'name': device_info.get('name'),
                        'channels': device_info.get('maxInputChannels'),
                        'sample_rate': int(device_info.get('defaultSampleRate'))
                    })
            
            self._device_info_cache = devices
            self.logger.debug(f"{len(devices)}個の入力デバイスを検出しました")
            return devices
            
        except Exception as e:
            self.logger.error(f"利用可能なデバイスの取得に失敗: {e}")
            return []
    
    def select_device(self, device_index):
        """
        使用する入力デバイスを選択する
        
        Args:
            device_index: デバイスのインデックス
            
        Returns:
            bool: デバイスの選択に成功したかどうか
        """
        try:
            devices = self.get_available_devices()
            
            for device in devices:
                if device['index'] == device_index:
                    self.device_index = device_index
                    self.logger.info(f"入力デバイスを選択: {device['name']} (index={device_index})")
                    return True
            
            self.logger.warning(f"デバイスインデックス {device_index} は見つかりません")
            return False
            
        except Exception as e:
            self.logger.error(f"デバイスの選択に失敗: {e}")
            return False
    
    def start_recording(self, duration=None, file_path=None):
        """
        録音を開始する
        
        Args:
            duration: 録音時間（秒）。Noneの場合は手動で停止するまで録音
            file_path: 録音データを保存するファイルパス。Noneの場合は一時ファイルを作成
            
        Returns:
            bool: 録音の開始に成功したかどうか
        """
        if self.is_recording:
            self.logger.warning("既に録音中です")
            return False
            
        try:
            if not self.initialize():
                return False
                
            # 録音状態の初期化
            self.frames = []
            self.is_recording = True
            self.start_time = time.time()
            self.stop_time = 0
            
            # 録音スレッドの開始
            self.recording_thread = threading.Thread(
                target=self._recording_thread,
                args=(duration, file_path),
                daemon=True
            )
            self.recording_thread.start()
            
            self.logger.info(f"録音を開始しました（デバイス: {self.device_index or 'デフォルト'}, " +
                            f"時間: {duration or '無制限'}秒）")
            
            # 開始コールバックを呼び出し
            if self.on_recording_started:
                self.on_recording_started()
                
            return True
            
        except Exception as e:
            self.logger.error(f"録音の開始に失敗: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self):
        """
        録音を停止する
        
        Returns:
            tuple: (録音データ, サンプリングレート) または None（失敗時）
        """
        if not self.is_recording:
            self.logger.warning("録音中ではありません")
            return None
            
        try:
            self.is_recording = False
            self.stop_time = time.time()
            
            # 録音スレッドの終了を待機
            if self.recording_thread and self.recording_thread.is_alive():
                self.recording_thread.join(timeout=1.0)
                
            duration = self.stop_time - self.start_time
            self.logger.info(f"録音を停止しました（時間: {duration:.2f}秒）")
            
            # 停止コールバックを呼び出し
            if self.on_recording_stopped:
                self.on_recording_stopped(self.frames, self.sample_rate)
                
            return (self._frames_to_array(), self.sample_rate)
            
        except Exception as e:
            self.logger.error(f"録音の停止に失敗: {e}")
            return None
    
    def get_recording_duration(self):
        """
        現在の録音時間を取得する
        
        Returns:
            float: 録音時間（秒）
        """
        if not self.is_recording:
            if self.stop_time > 0 and self.start_time > 0:
                return self.stop_time - self.start_time
            return 0
            
        return time.time() - self.start_time
    
    def save_recording(self, file_path):
        """
        録音データをファイルに保存する
        
        Args:
            file_path: 保存先のファイルパス
            
        Returns:
            bool: 保存に成功したかどうか
        """
        if len(self.frames) == 0:
            self.logger.warning("保存する録音データがありません")
            return False
            
        try:
            # ディレクトリが存在しない場合は作成
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # WAVファイルに保存
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.audio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(b''.join(self.frames))
                
            self.logger.info(f"録音データを保存しました: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"録音データの保存に失敗: {e}")
            return False
    
    def _recording_thread(self, duration=None, file_path=None):
        """
        録音処理を行うスレッド
        
        Args:
            duration: 録音時間（秒）。Noneの場合は手動で停止するまで録音
            file_path: 録音データを保存するファイルパス。Noneの場合は保存しない
        """
        try:
            # 録音ストリームのオープン
            stream = self.audio.open(
                format=self.format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                input_device_index=self.device_index,
                frames_per_buffer=self.chunk_size
            )
            
            end_time = None
            if duration is not None:
                end_time = time.time() + duration
                
            # 録音ループ
            while self.is_recording:
                if end_time and time.time() >= end_time:
                    self.is_recording = False
                    break
                    
                data = stream.read(self.chunk_size, exception_on_overflow=False)
                self.frames.append(data)
                
                # 進捗コールバックを呼び出し
                if self.on_recording_progress:
                    current_duration = self.get_recording_duration()
                    max_duration = duration or 0
                    self.on_recording_progress(current_duration, max_duration)
            
            # ストリームを閉じる
            stream.stop_stream()
            stream.close()
            
            # ファイルに保存（指定されている場合）
            if file_path and len(self.frames) > 0:
                self.save_recording(file_path)
                
        except Exception as e:
            self.logger.error(f"録音スレッドでエラーが発生: {e}")
            self.is_recording = False
    
    def _frames_to_array(self):
        """
        録音フレームをNumPy配列に変換する
        
        Returns:
            numpy.ndarray: 音声データの配列
        """
        if not self.frames:
            return np.array([])
            
        try:
            # バイトデータを結合
            data = b''.join(self.frames)
            
            # 16ビット整数に変換
            if self.format == pyaudio.paInt16:
                data_np = np.frombuffer(data, dtype=np.int16)
                # 正規化（-1.0〜1.0）
                data_np = data_np.astype(np.float32) / 32768.0
                return data_np
            else:
                self.logger.warning(f"未対応のフォーマット: {self.format}")
                return np.array([])
                
        except Exception as e:
            self.logger.error(f"録音データの変換に失敗: {e}")
            return np.array([])
    
    def get_temp_file_path(self):
        """
        一時ファイルのパスを生成する
        
        Returns:
            str: 一時ファイルのパス
        """
        temp_dir = tempfile.gettempdir()
        timestamp = int(time.time())
        return os.path.join(temp_dir, f"recording_{timestamp}.wav") 