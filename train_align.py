import os

from domain.dto.AlignConfig import AlignConfig
from trainer.AlignTrainer import AlignTrainer
from util.Logger import Logger
from util.Util import Util


def train(config_path: str):
    logger = Logger.get_logger(__name__)
    logger.info("학습 시작")
    config = AlignConfig.load_config(config_path)
    logger.info(f"\n{config}")
    if not os.path.exists(config.clip_model_path):
        logger.info("CLIP 모델 다운로드 필요")
        Util.download_clip_model(config.clip_model_id, config.clip_model_path)
        logger.info("CLIP 모델 다운로드 완료")

    align_trainer = AlignTrainer(config)
    align_trainer.train()


if __name__ == "__main__":
    train("config/align_config.yaml")
