from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from PIL import Image
from torch.utils.data import Dataset


class Flickr8kCaptionDataset(Dataset):
    def __init__(self, image_path: str, token_path: str, seed: int, split_path: Optional[str] = None):
        self.image_path = image_path
        self.split_path = split_path
        allowed_images = self.load_image_names(split_path)
        self.items: List[Tuple[str, str]] = []
        for image_name, caption in self.load_items(token_path, allowed_images):
            self.items.append((image_name, caption))
        if not self.items:
            raise ValueError(f"no dataset rows found for token_path={token_path}, split_path={split_path}")

    def __getitem__(self, idx: int) -> dict:
        path, caption = self.items[idx]
        image = Image.open(f"{self.image_path}/{path}").convert("RGB")
        return {"image": image, "caption": caption}

    def __len__(self):
        return len(self.items)

    @staticmethod
    def collate_fn(batch: List[Dict]):
        return {
            "images": [b["image"] for b in batch],
            "captions": [b["caption"] for b in batch],
        }

    @staticmethod
    def load_image_names(split_path: Optional[str]) -> Optional[Set[str]]:
        if split_path is None:
            return None

        with open(split_path, encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    @classmethod
    def load_items(cls, token_path: str, allowed_images: Optional[Set[str]] = None) -> List[Tuple[str, str]]:
        with open(token_path, encoding="utf-8") as f:
            line_list = f.readlines()

        items: List[Tuple[str, str]] = []
        for line in line_list:
            parsed = cls.parse_line(line)
            if parsed is None:
                continue

            image_name, caption = parsed
            if allowed_images is not None and image_name not in allowed_images:
                continue
            items.append((image_name, caption))

        return items

    @classmethod
    def load_grouped_captions(
        cls, token_path: str, allowed_images: Optional[Set[str]] = None
    ) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = defaultdict(list)
        for image_name, caption in cls.load_items(token_path, allowed_images):
            grouped[image_name].append(caption)
        return dict(grouped)

    @staticmethod
    def parse_line(line: str) -> Optional[Tuple[str, str]]:
        stripped = line.rstrip("\n")
        if not stripped:
            return None

        if stripped == "image,caption":
            return None

        if "\t" in stripped:
            image_name, caption = stripped.split("\t", 1)
            return Flickr8kCaptionDataset.normalize_image_name(image_name), caption.strip()

        split_list = stripped.split(",")
        if len(split_list) < 2:
            return None

        image_name = split_list[0].strip()
        caption = ",".join(split_list[1:]).strip()
        return Flickr8kCaptionDataset.normalize_image_name(image_name), caption

    @staticmethod
    def normalize_image_name(image_name: str) -> str:
        return image_name.split("#", 1)[0].strip()
