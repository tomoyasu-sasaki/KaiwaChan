# KaiwaChan
会話ちゃんシステム - ローカル環境で動作する音声対話可能なAIキャラクターチャットシステム

## 概要
KaiwaChanは、プライバシーを重視したローカル完結型の音声対話システムです。音声認識、対話生成、音声合成をすべてローカルで実行します。

## 現在の実装状況（Phase 1）

### 実装済み機能
- 音声認識 (Whisper)
- 対話生成 (LLaMA)
- 音声合成 (VOICEVOX)
- 基本的なGUIインターフェース
- 設定管理システム
- ログ機能

### システム構成
```
KaiwaChan/
├── src/
│ ├── core/ # コア機能
│ │ ├── stt.py # 音声認識 (Whisper)
│ │ ├── dialogue.py # 対話生成 (LLaMA)
│ │ └── tts.py # 音声合成 (VOICEVOX)
│ ├── ui/ # ユーザーインターフェース
│ │ └── main_window.py
│ └── utils/ # ユーティリティ
│ ├── config.py # 設定管理
│ └── logger.py # ログ機能
├── models/ # モデルファイル
├── logs/ # ログファイル
└── config.yml # 設定ファイル
```

### 使用技術
- **音声認識**: OpenAI Whisper
- **対話生成**: LLaMA 2
- **音声合成**: VOICEVOX
- **GUI**: PyQt6
- **その他**: numpy, scipy, sounddevice

### 動作要件
- Python 3.8以上
- RAM: 16GB以上
- GPU: VRAM 6GB以上推奨
- ストレージ: 20GB以上の空き容量

## セットアップ

1. 仮想環境の作成
```bash
python -m venv .kaiwachan
source .kaiwachan/bin/activate  # Linux/Mac
# または
.kaiwachan\Scripts\activate     # Windows
```

2. 依存関係のインストール
```bash
pip install -r requirements.txt
```

3. 必要なモデルの準備
- Whisperモデル（自動ダウンロード）
- LLaMA 2モデル（要別途ダウンロード）
- VOICEVOXエンジン（要別途インストール）

4. 設定ファイルの確認
- `config.yml`の内容を環境に合わせて調整

5. 実行
```bash
python -m src.main
```

## 今後の開発予定
- キャラクター表示システム（Live2D）
- 音声クローニング機能
- マルチキャラクター対応
- プラグインシステム

## ライセンス
[ライセンス情報を追加予定]

## 機能の処理フロー

### 1. 音声対話の基本フロー
1. **音声入力** (Whisper)
   - マイクから音声を録音 (sounddevice)
   - 音声データをテキストに変換 (OpenAI Whisper)
   - 使用技術: sounddevice (録音), numpy (音声データ処理)

2. **対話生成** (LLaMA 2)
   - テキストから文脈を理解
   - 適切な応答を生成
   - 使用技術: LLaMA 2 (大規模言語モデル)

3. **音声合成** (VOICEVOX)
   - 生成されたテキストを音声に変換
   - キャラクターの声で出力
   - 使用技術: VOICEVOX (音声合成エンジン)

### 2. システムの主要コンポーネント

#### GUI (PyQt6)
- メインウィンドウの表示
- ボタン操作の処理
- テキスト表示
- 使用技術: PyQt6 (グラフィカルインターフェース)

#### 設定管理
- 設定ファイル (YAML) の読み込み
- モデルパスの管理
- 音声設定の管理
- 使用技術: PyYAML (設定ファイル処理)

#### ログ機能
- システムの動作記録
- エラー情報の保存
- デバッグ情報の出力
- 使用技術: Python logging (ログ管理)

### 3. 技術詳細

#### 音声認識 (Whisper)
- **用途**: 音声をテキストに変換
- **特徴**: 
  - 高精度な音声認識
  - 多言語対応
  - ローカルで動作
- **処理フロー**:
マイク入力 → 音声データ取得 → ノイズ処理 → Whisperモデルで認識 → テキスト出力

#### 対話生成 (LLaMA 2)
- **用途**: 自然な対話の生成
- **特徴**:
  - 高度な文脈理解
  - カスタマイズ可能な応答
  - ローカルで高速処理
- **処理フロー**:
テキスト入力 → 文脈理解 → 応答生成 → テキスト出力

#### 音声合成 (VOICEVOX)
- **用途**: テキストを音声に変換
- **特徴**:
  - 高品質な音声合成
  - 複数のキャラクターボイス
  - リアルタイム処理
- **処理フロー**:
テキスト入力 → 音声パラメータ設定 → 音声合成 → 音声出力

### 4. データの流れ
```
音声入力 → テキスト変換 (Whisper) → 対話生成 (LLaMA 2) → 音声合成 (VOICEVOX) → 音声出力
```

各ステップでのデータ形式:
1. 音声入力: WAVデータ (numpy配列)
2. テキスト変換: 文字列
3. 対話生成: 文字列
4. 音声合成: WAVデータ

### 5. 使用ライブラリと役割
- **numpy**: 音声データの数値処理
- **scipy**: 信号処理とフィルタリング
- **sounddevice**: マイク入力の制御
- **PyQt6**: グラフィカルインターフェース
- **requests**: VOICEVOXとの通信
- **yaml**: 設定ファイルの処理
- **logging**: システムログの管理

## 開発状況

### Phase 1 ✅
- 基本的な音声認識/合成
- 対話生成
- UIの基本実装

### Phase 2 🚧
- キャラクター表示システム
- アニメーション制御
- 音声同期

## ライセンス
MIT License

## 貢献
プルリクエストやイシューの報告を歓迎します。

## 謝辞
- [Whisper](https://github.com/openai/whisper)
- [VOICEVOX](https://voicevox.hiroshiba.jp/)
- [LLaMA](https://github.com/facebookresearch/llama)
- [PyGame](https://www.pygame.org/)
