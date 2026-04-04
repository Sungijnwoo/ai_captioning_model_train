import os
import argparse

from domain.dto.CaptionConfig import CaptionConfig
from trainer.CaptionTrainer import CaptionTrainer
from util.Logger import Logger
from util.Util import Util


def train(config_path: str):
    logger = Logger.get_logger(__name__)
    logger.info("학습 시작")
    config = CaptionConfig.load_config(config_path)

    if not os.path.exists(config.clip_model_path):
        logger.info("CLIP 모델 다운로드 필요")
        Util.download_clip_model(config.clip_model_id, config.clip_model_path)
        logger.info("CLIP 모델 다운로드 완료")

    caption_trainer = CaptionTrainer(config)
    caption_trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="config/caption_config.yaml")

    args = parser.parse_args()
    train(args.config)
