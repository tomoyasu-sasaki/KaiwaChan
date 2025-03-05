# KaiwaChan (会話ちゃん)
会話ちゃんシステム - ローカル環境で動作する音声対話可能なAIキャラクターチャットシステム

## 概要
KaiwaChan（会話ちゃん）は、ローカル環境で動作する音声対話可能なAIキャラクターチャットシステムです。ユーザーの音声を認識し、AIによる応答を生成した後、設定されたキャラクターの声で回答する機能を持っています。また、アニメーションキャラクターとの対話も可能です。

## 特徴
- 音声認識による対話入力（無音検出による自動録音終了機能搭載）
- ローカル環境で動作するLLMによる応答生成
- 音声合成（VOICEVOXエンジン連携）
- ボイスクローン機能（自分の声や特定の声でキャラクターに話させることが可能）
- 2Dキャラクターアニメーション表示
- プロファイル管理機能
- 詳細なログ出力とデバッグ機能

## 使用技術
- **Python 3.9+**: 基本言語
- **PyQt6**: GUIフレームワーク
- **Llama.cpp**: ローカルLLM推論エンジン
- **Granite-3.1-8b-instruct**: 会話生成AI言語モデル
- **Whisper**: 音声認識（Speech-to-Text）
- **VOICEVOX**: 音声合成（Text-to-Speech）
- **TTS (Coqui TTS)**: ボイスクローン機能
- **PyGame**: キャラクターアニメーション
- **NumPy/SciPy**: 信号処理
- **PyAudio/SoundDevice**: オーディオ入出力
- **YAML**: 設定ファイル管理

## システム要件
- Python 3.9以上
- CUDA対応GPUを推奨（CPU動作も可能）
- 最低8GB以上のRAM（16GB以上推奨）
- VOICEVOXエンジン（音声合成に使用）
- 約15GB以上の空き容量（モデルファイル含む）

## インストール方法
1. リポジトリをクローン
```bash
git clone https://github.com/[username]/KaiwaChan.git
cd KaiwaChan
```

2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

3. 必要なモデルファイルのダウンロード
   - [Granite-3.1-8b-instruct](https://huggingface.co/lmstudio-community/granite-3.1-8b-instruct-GGUF) をダウンロードして `models/` ディレクトリに配置
   - VOICEVOX Engineのインストール（[公式サイト](https://voicevox.hiroshiba.jp/)から入手可能）

4. 設定ファイルの確認と調整
   - `config.yml` ファイルを環境に合わせて調整

## 使用方法
1. VOICEVOXエンジンを起動する

2. KaiwaChanを起動する
```bash
python -m src.main
```

3. メインウィンドウが表示されたら「話す」ボタンをクリックして対話を開始
   - マイクに向かって話しかけると、音声が認識され応答が生成されます
   - 応答はテキストとして表示され、設定された音声で再生されます
   - 会話終了後、2秒間の無音が検出されると自動的に録音が終了します

4. 必要に応じてボイスプロファイルを作成・選択することで、好みの声での応答を設定可能

## ディレクトリ構造
```
KaiwaChan/
├── src/                    # ソースコード
│   ├── core/               # コア機能
│   │   ├── animation/      # アニメーション関連
│   │   │   ├── character_animator.py  # キャラクターアニメーション管理
│   │   │   └── sprite_manager.py      # スプライト管理
│   │   ├── audio/          # オーディオ処理
│   │   │   ├── player.py   # 音声再生
│   │   │   ├── processor.py # 音声処理
│   │   │   └── recorder.py # 音声録音
│   │   ├── dialogue/       # 対話エンジン
│   │   │   └── dialogue_engine.py # LLMを使用した対話生成
│   │   ├── stt/            # 音声認識 (Speech-to-Text)
│   │   │   └── speech_recognizer.py # Whisperによる音声認識
│   │   ├── tts/            # 音声合成 (Text-to-Speech)
│   │   │   └── tts_engine.py # VOICEVOXによる音声合成
│   │   └── voice/          # 音声に関連する追加機能
│   │       ├── feature_extractor.py # 音声特徴抽出
│   │       ├── profile_manager.py   # 音声プロファイル管理
│   │       └── voice_clone_manager.py # ボイスクローン機能
│   ├── config/             # 設定関連
│   │   ├── app_config.py   # アプリケーション設定
│   │   ├── settings_manager.py # 設定管理
│   │   └── user_settings.py # ユーザー設定
│   ├── ui/                 # ユーザーインターフェース
│   │   ├── components/     # UIコンポーネント
│   │   │   └── message_display.py # メッセージ表示コンポーネント
│   │   ├── main_window.py  # メインウィンドウ
│   │   └── voice_profile_dialog.py # ボイスプロファイル設定ダイアログ
│   ├── utils/              # ユーティリティ
│   │   ├── error_handler.py # エラー処理
│   │   ├── file_manager.py # ファイル管理
│   │   ├── logger.py       # ロギング機能
│   │   └── model_downloader.py # モデルダウンロード
│   └── main.py             # エントリーポイント
├── models/                 # モデルファイル格納ディレクトリ
├── profiles/               # ボイスプロファイル保存ディレクトリ
├── assets/                 # 画像・アニメーション素材
├── characters/             # キャラクター設定
├── logs/                   # ログファイル
├── cache/                  # キャッシュデータ
├── config.yml              # 設定ファイル
└── requirements.txt        # 依存パッケージリスト
```

## 主要機能の説明

### 音声認識 (STT)
- Whisperモデルを使用した高精度な音声認識
- 無音検出による自動録音終了（改良版）
  - 設定ファイルで調整可能なsilence_threshold値
  - 有声検出後の無音のみを終了判定に使用（雑音対策）
  - 2秒間（デフォルト）の無音検出で自動終了
- 日本語に最適化された設定
- タイムスタンプ付き認識にも対応

### 対話エンジン
- Granite-3.1-8b-instructモデルを使用
- ローカル環境での高速な応答生成
- キャラクターに合わせたパーソナリティの調整が可能
- 会話履歴の保持と管理
- 会話のエクスポート機能

### 音声合成 (TTS)
- VOICEVOXエンジンを利用した自然な日本語音声合成
- 複数のキャラクターボイス選択が可能
- キャッシュ機能による高速な応答
- カスタマイズ可能な音声パラメータ（速度、音程など）

### ボイスクローン
- 自分の声や特定の声でキャラクターに話させることが可能
- 複数のプロファイル管理
- サンプル音声から特徴を抽出し、声質を変換
- プロファイルのインポート・エクスポート機能

### キャラクターアニメーション
- PyGameを使用した2Dキャラクター表示
- 表情変化や口パクアニメーション
- 会話状態に応じたアニメーション切り替え
- カスタマイズ可能なアニメーション設定

### ロギングシステム
- 詳細なログレベル設定（DEBUG、INFO、WARNING、ERROR、CRITICAL）
- 設定ファイルでログレベルを調整可能
- ファイルとコンソールへの出力
- ローテーションによるログファイル管理

## 設定ファイル（config.yml）の説明

### audio
- **duration**: デフォルトの最大録音時間（秒）
- **sample_rate**: サンプルレート（Hz）
- **channels**: チャンネル数
- **device**: 使用するオーディオデバイス（nullの場合はデフォルト）
- **silence_threshold**: 無音判定の閾値（0.03推奨）

### models
- **llm**: LLMモデルの設定
  - **file**: 使用するモデルファイル
  - **n_threads**: 使用するスレッド数
  - **n_batch**: バッチサイズ
  - **max_tokens**: 最大生成トークン数
- **voicevox**: VOICEVOXエンジンの設定
  - **engine_path**: エンジンの実行パス
  - **cache_size**: キャッシュサイズ
  - **timeout**: タイムアウト設定
- **whisper**: 使用するWhisperモデルサイズ

### logging
- **level**: ログレベル（DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50）
- **format**: ログフォーマット
- **date_format**: 日付フォーマット
- **max_files**: 保存する最大ログファイル数
- **max_size_mb**: 各ログファイルの最大サイズ（MB）

### character
- **default_id**: デフォルトキャラクターID
- **expressions**: 表情リスト
- **window**: ウィンドウサイズ設定
- **fps**: アニメーションのFPS

### voice_clone
- **model_path**: ボイスクローンモデルのパス
- **profiles_dir**: プロファイル保存ディレクトリ
- **sample_rate**: サンプルレート
- **chunk_size**: チャンクサイズ
- **feature_dim**: 特徴次元数

## カスタマイズと拡張
- `config.yml`ファイルで各種設定を調整可能
- 無音検出の閾値（`silence_threshold`）を環境に合わせて調整することで、音声入力の精度を向上
- `characters/`ディレクトリにキャラクター設定を追加することで拡張可能
- 音声プロファイルをカスタマイズしてオリジナルの声を作成可能
- ログレベルを変更することで、デバッグや問題解決に必要な情報を表示可能

## トラブルシューティング
- VOICEVOXエンジンが動作していない場合、音声合成機能は使用できません
- GPUメモリ不足の場合は、`config.yml`のモデル設定を調整してみてください
- 音声認識の精度が低い場合は、静かな環境で使用するか、マイク設定を確認してください
- 無音検出が正しく機能しない場合は、`config.yml`の`silence_threshold`値を環境に合わせて調整してください
  - 値を上げる: 背景ノイズが多い環境での無音検出精度向上
  - 値を下げる: 静かな環境での検出感度向上

## ライセンス
このプロジェクトは[ライセンス名]の下で公開されています。詳細はLICENSEファイルを参照してください。

## 謝辞
- [Whisper](https://github.com/openai/whisper): 音声認識モデル
- [VOICEVOX](https://voicevox.hiroshiba.jp/): 音声合成エンジン
- [Llama.cpp](https://github.com/ggerganov/llama.cpp): 高速なLLM推論エンジン
- [Granite](https://huggingface.co/lmstudio-community/granite-3.1-8b-instruct-GGUF): 高品質な言語モデル
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/): GUIフレームワーク
- [SoundDevice](https://python-sounddevice.readthedocs.io/): オーディオ入出力ライブラリ
