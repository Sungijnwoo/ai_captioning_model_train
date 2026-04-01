import torch

from domain.dto.Config import Config
from model.ClipToLlmProjector import ClipToLlmProjector
from util.Logger import Logger
from util.Util import Util


class AlignTrainer:
    def __init__(self, config: Config):
        self.logger = Logger.get_logger(__name__)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.clip_model, self.clip_preprocess_train, self.clip_preprocess_val = Util.load_clip_model(self.config)
        self.llm_model, self.llm_tokenizer = Util.load_llm_model(self.config)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            clip_dim = int(self.clip_model.encode_image(dummy).shape[-1])
            self.logger.info(f"CLIP DIM: {clip_dim}")

        llm_dim = int(self.llm_model.config.hidden_size)
        self.logger.info(f"LLM DIM: {llm_dim}")
        self.projector = ClipToLlmProjector(clip_dim, llm_dim, self.config.project_multi_rate).to(self.device)

        self.logger.info("모델 세팅 완료")
