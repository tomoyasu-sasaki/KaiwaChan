import logging
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, config):
        log_dir = Path(config.config["paths"]["logs"])
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"kaiwa_chan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("KaiwaChan")

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def warning(self, message):
        self.logger.warning(message) 