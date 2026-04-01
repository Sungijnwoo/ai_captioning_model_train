import os

from loguru import logger


class Logger:
    _initialized = False

    @classmethod
    def _init_logger(cls):
        if cls._initialized:
            return

        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # 파일 (날짜별)
        logger.add(
            f"{log_dir}/app_{{time:YYYY-MM-DD}}.log",
            rotation="00:00",
            retention="30 days",
            compression="zip",
            level="DEBUG",
            encoding="utf-8",
            enqueue=True,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {extra[class_name]} | {message}",
        )

        cls._initialized = True

    @classmethod
    def get_logger(cls, name: str = "GLOBAL"):
        cls._init_logger()
        return logger.bind(class_name=name)
