from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QPushButton, 
                            QListWidget, QFileDialog, QInputDialog, QMessageBox, QProgressDialog, QGroupBox, QHBoxLayout, QLineEdit)
from PyQt6.QtCore import Qt, QSize
from pathlib import Path
import logging
import soundfile as sf

class VoiceProfileDialog(QDialog):
    def __init__(self, voice_clone, parent=None):
        super().__init__(parent)
        self.voice_clone = voice_clone
        self.setWindowTitle("音声プロファイル管理")
        self.setModal(True)
        self._setup_ui()
        self.logger = logging.getLogger(__name__)

    def _setup_ui(self):
        """UIの追加機能"""
        layout = QVBoxLayout()
        
        # プロファイルリスト
        self.profile_list = QListWidget()
        self.profile_list.addItems(self.voice_clone.voice_profiles.keys())
        layout.addWidget(self.profile_list)
        
        # テスト再生グループ
        test_group = QGroupBox("テスト再生")
        test_layout = QVBoxLayout()
        
        self.test_text = QLineEdit()
        self.test_text.setPlaceholderText("テスト用のテキストを入力")
        test_layout.addWidget(self.test_text)
        
        test_button = QPushButton("再生")
        test_button.clicked.connect(self._test_voice)
        test_layout.addWidget(test_button)
        
        test_group.setLayout(test_layout)
        layout.addWidget(test_group)
        
        # 操作ボタン
        button_layout = QHBoxLayout()
        
        create_button = QPushButton("新規作成")
        create_button.clicked.connect(self._create_profile)
        button_layout.addWidget(create_button)
        
        delete_button = QPushButton("削除")
        delete_button.clicked.connect(self._delete_profile)
        button_layout.addWidget(delete_button)
        
        layout.addLayout(button_layout)
        
        self.setLayout(layout)

    def _on_profile_selected(self, current, previous):
        """プロファイル選択時の処理"""
        if current:
            profile_name = current.text()
            self.test_text.setEnabled(True)
        else:
            self.test_text.setEnabled(False)

    def _create_profile(self):
        """新規プロファイル作成"""
        # プロファイル作成前の所要時間の見積もり表示
        info_text = (
            "プロファイル作成の所要時間の目安:\n\n"
            "・30秒の音声ファイル: 約1分\n"
            "・1分の音声ファイル: 約2分\n"
            "・2分の音声ファイル: 約4分\n\n"
            "※ 処理時間はファイルサイズと数、PCの性能により変動します。\n"
            "※ 処理中はアプリケーションの操作が制限されます。\n\n"
            "続行しますか？"
        )
        
        reply = QMessageBox.question(
            self,
            "処理時間の目安",
            info_text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.No:
            return

        # プロファイル作成処理
        name, ok = QInputDialog.getText(
            self, 
            "新規プロファイル", 
            "プロファイル名を入力してください："
        )
        if ok and name:
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "音声ファイルを選択",
                "",
                "Audio Files (*.wav *.mp3)"
            )
            if files:
                # 合計処理時間の見積もり
                total_duration = 0
                for file in files:
                    try:
                        wav, sr = sf.read(file)
                        total_duration += len(wav) / sr
                    except Exception:
                        continue
                
                estimated_time = int(total_duration * 2)  # 音声長の約2倍が処理時間の目安
                
                # プログレスダイアログの設定
                progress = QProgressDialog(
                    f"プロファイルを作成中...\n推定所要時間: 約{estimated_time}分",
                    "キャンセル",
                    0,
                    len(files) * 3,  # 3ステップ: 読み込み、特徴抽出、統合
                    self
                )
                progress.setWindowModality(Qt.WindowModality.WindowModal)
                progress.setWindowTitle("処理中")
                
                def update_progress(current, total, message):
                    if progress.wasCanceled():
                        return False
                    progress.setLabelText(f"{message}\n残り推定時間: 約{int(estimated_time * (1 - current/total))}分")
                    progress.setValue(current)
                    return True
                
                success = self.voice_clone.create_profile(
                    files, 
                    name,
                    progress_callback=update_progress
                )
                
                progress.close()
                
                if success:
                    # プロファイルの再読み込み
                    self.voice_clone.profile_manager._load_existing_profiles()
                    # リストの更新
                    self.profile_list.clear()
                    profile_names = self.voice_clone.get_profile_names()
                    self.profile_list.addItems(profile_names.values())
                    # 新しいプロファイルを選択
                    items = self.profile_list.findItems(name, Qt.MatchFlag.MatchExactly)
                    if items:
                        self.profile_list.setCurrentItem(items[0])
                    self.accept()
                else:
                    QMessageBox.warning(
                        self,
                        "エラー",
                        "プロファイルの作成に失敗しました"
                    )

    def _import_voice(self):
        """既存プロファイルに音声を追加"""
        current = self.profile_list.currentItem()
        if not current:
            QMessageBox.warning(
                self,
                "警告",
                "プロファイルを選択してください"
            )
            return
            
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "音声ファイルを選択",
            "",
            "Audio Files (*.wav *.mp3)"
        )
        if files:
            profile_name = current.text()
            success = self.voice_clone.add_voice_samples(profile_name, files)
            if not success:
                QMessageBox.warning(
                    self,
                    "エラー",
                    "音声ファイルの追加に失敗しました"
                ) 

    def _test_voice(self):
        """選択中のプロファイルでテスト音声を生成"""
        current = self.profile_list.currentItem()
        if not current:
            QMessageBox.warning(self, "警告", "プロファイルを選択してください")
            return
        
        text = self.test_text.text().strip()
        if not text:
            QMessageBox.warning(self, "警告", "テキストを入力してください")
            return
        
        try:
            # 音声生成
            profile_name = current.text()
            audio = self.voice_clone.convert_voice(text, profile_name)
            
            if audio is not None:
                # 音声再生
                self.voice_clone.play_audio(audio)
                self.logger.info(f"テスト音声再生: {text}")
            else:
                QMessageBox.warning(
                    self,
                    "エラー",
                    "音声生成に失敗しました"
                )
                
        except Exception as e:
            self.logger.error(f"テスト再生エラー: {str(e)}")
            QMessageBox.critical(
                self,
                "エラー",
                f"テスト再生中にエラーが発生しました: {str(e)}"
            )

    def _delete_profile(self):
        """プロファイルの削除"""
        current = self.profile_list.currentItem()
        if not current:
            QMessageBox.warning(
                self,
                "警告",
                "削除するプロファイルを選択してください"
            )
            return
            
        profile_name = current.text()
        reply = QMessageBox.question(
            self,
            "確認",
            f"プロファイル '{profile_name}' を削除してもよろしいですか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # プロファイルファイルの削除
                profile_path = Path(self.voice_clone.config.get('voice_clone', {}).get(
                    'profiles_dir', 'profiles')) / f"{profile_name}.npz"
                if profile_path.exists():
                    profile_path.unlink()
                
                # メモリ上のプロファイルを削除
                if profile_name in self.voice_clone.voice_profiles:
                    del self.voice_clone.voice_profiles[profile_name]
                
                # リストから削除
                self.profile_list.takeItem(self.profile_list.row(current))
                
                self.logger.info(f"プロファイル削除完了: {profile_name}")
                
            except Exception as e:
                self.logger.error(f"プロファイル削除エラー: {str(e)}")
                QMessageBox.critical(
                    self,
                    "エラー",
                    f"プロファイルの削除に失敗しました: {str(e)}"
                ) 