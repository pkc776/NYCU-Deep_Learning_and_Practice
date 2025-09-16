import os
import json
from typing import Callable, Optional, List, Dict

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class IclevrDataset(Dataset):
    """
    Dataset for the iCLEVR folder structure:
      - images/ (contains PNG files)
      - train.json (or test.json/new_test.json) mapping filename -> list of object names
      - objects.json mapping object name -> index

    Returns dict with keys: 'image' (Tensor CxHxW), 'cond' (multi-hot Tensor num_objects), 'fname' (str)

    Parameters:
      root: path to the iCLEVR folder (contains images, train.json, objects.json)
      split_file: name of the JSON file with annotations (default 'train.json')
      transform: torchvision transform applied to PIL image; if None a sensible default is used
      image_size: optional int to resize smaller/ larger images (keeps aspect by center crop after resize to square)
    """

    def __init__(self,
                 root: str,
                 split_file: str = "train.json",
                 objects_file: str = "objects.json",
                 transform: Optional[Callable] = None,
                 image_size: Optional[int] = None):
        self.root = root
        self.images_dir = os.path.join(root, "images")
        ann_path = os.path.join(root, split_file)
        obj_path = os.path.join(root, objects_file)

        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")
        if not os.path.exists(obj_path):
            raise FileNotFoundError(f"Objects file not found: {obj_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            self.anns: Dict[str, List[str]] = json.load(f)

        with open(obj_path, "r", encoding="utf-8") as f:
            self.objects: Dict[str, int] = json.load(f)

        # reverse map index->name if needed
        self.num_objects = len(self.objects)

        # build list of (filename, object_list)
        self.items = []
        for fname, obj_list in self.anns.items():
            # skip missing image files gracefully
            img_path = os.path.join(self.images_dir, fname)
            if not os.path.exists(img_path):
                # try if fname is full path
                if os.path.exists(fname):
                    img_path = fname
                else:
                    # skip and warn
                    # print(f"Warning: image not found: {img_path}")
                    continue
            self.items.append((fname, obj_list))

        # default transform: ToTensor + normalize to [-1,1]
        if transform is None:
            tlist = []
            if image_size is not None:
                # Resize to the exact square target instead of Resize + CenterCrop.
                # CenterCrop can cut objects that lie near image edges; using
                # Resize((H,W)) preserves the whole image content while scaling
                # to the desired resolution.
                tlist.append(transforms.Resize((image_size, image_size)))
            tlist.extend([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
            self.transform = transforms.Compose(tlist)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.items)

    def _make_multihot(self, obj_list: List[str]) -> torch.Tensor:
        mh = torch.zeros(self.num_objects, dtype=torch.float32)
        for obj in obj_list:
            if obj in self.objects:
                mh[self.objects[obj]] = 1.0
            else:
                # unknown object name: ignore or raise depending on preference
                # here we ignore but you can print a warning
                # print(f"Warning: unknown object '{obj}' in annotation")
                pass
        return mh

    def __getitem__(self, idx: int):
        fname, obj_list = self.items[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        cond = self._make_multihot(obj_list)
        return {"image": img, "cond": cond, "fname": fname}


def make_dataloader(root: str, split_file: str = "train.json", objects_file: str = "objects.json",
                    batch_size: int = 8, shuffle: bool = True, num_workers: int = 4,
                    image_size: Optional[int] = None) -> DataLoader:
    """Utility to create a DataLoader for the iCLEVR dataset."""
    ds = IclevrDataset(root, split_file=split_file, objects_file=objects_file, image_size=image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)


if __name__ == "__main__":
    # quick smoke test (adjust path as needed)
    ds = IclevrDataset("./iclevr", split_file="train.json", image_size=128)
    dl = DataLoader(ds, batch_size=4)
    batch = next(iter(dl))
    print(batch["image"].shape, batch["cond"].shape)
