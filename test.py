from pathlib import Path

from domain.dto.CaptionConfig import CaptionConfig
from domain.dto.TestConfig import TestConfig
from trainer.CaptionTester import CaptionTester
from util.Logger import Logger
from util.Util import Util


def test(config_path: str):
    logger = Logger.get_logger(__name__)
    logger.info("테스트 시작")
    config = TestConfig.load_config(config_path)

    caption_config_path = Path(config.model_dir) / "caption_config.yaml"
    caption_config = CaptionConfig.load_config(str(caption_config_path))
    if not Path(caption_config.clip_model_path).exists():
        logger.info("CLIP 모델 다운로드 필요")
        Util.download_clip_model(caption_config.clip_model_id, caption_config.clip_model_path)
        logger.info("CLIP 모델 다운로드 완료")

    caption_tester = CaptionTester(
        config.model_dir,
        llm_name=config.llm_name,
        projector_name=config.projector_name,
    )
    caption_tester.test(config)


if __name__ == "__main__":
    test("config/test_config.yaml")
