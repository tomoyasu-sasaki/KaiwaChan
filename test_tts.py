from src.core.tts.tts_engine import TTSEngine
import logging

# ログ設定
logging.basicConfig(level=logging.INFO)

def test_voicevox():
    print('VOICEVOXのテスト開始...')
    config = {
        'models': {
            'current_engine': 'voicevox',
            'voicevox': {
                'speaker_id': 1
            }
        }
    }
    engine = TTSEngine(config)
    result = engine.synthesize('こんにちは、テストです。')
    print(f'VOICEVOX合成結果: {result}')

def test_parler():
    print('\nParler-TTSのテスト開始...')
    config = {
        'models': {
            'current_engine': 'parler',
            'parler': {
                'description': 'A female speaker with a clear voice'
            }
        }
    }
    engine = TTSEngine(config)
    result = engine.synthesize('こんにちは、テストです。')
    print(f'Parler-TTS合成結果: {result}')

if __name__ == '__main__':
    test_voicevox()
    test_parler()