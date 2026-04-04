from typing import Dict, List, Tuple

from PIL import Image
from torch.utils.data import Dataset


class Flickr8kCaptionDataset(Dataset):
    def __init__(self, image_path: str, token_path: str, seed: int):
        with open(token_path, encoding="utf-8") as f:
            line_list = f.readlines()

        self.image_path = image_path
        self.items: List[Tuple[str, str]] = []
        for line in line_list[1:]:
            split_list = line.split(",")
            image_name = split_list[0]
            caption = ",".join(split_list[1:])
            self.items.append((image_name, caption))

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
