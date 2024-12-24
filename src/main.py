import sys
from pathlib import Path
import signal
sys.path.append(str(Path(__file__).parent.parent))

from PyQt6.QtWidgets import QApplication
from src.ui.main_window import MainWindow

def main():
    # Ctrl+C のシグナルハンドリングを有効化
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    
    try:
        sys.exit(app.exec())
    except KeyboardInterrupt:
        window.cleanup()
        sys.exit(0)

if __name__ == "__main__":
    main() 