import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import torch


class CSLDaily(torch.utils.data.Dataset):
    """
    Dataset class for CSL-Daily sign language translation.
    """
    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = "dev",
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = "",
        spatiotemporal_postfix: Union[str, List[str]] = ""
    ):
        super().__init__()

        self.anno_root = Path(anno_root)
        self.vid_root = Path(vid_root)
        self.feat_root = Path(feat_root)
        self.mae_feat_root = Path(mae_feat_root)
        self.mode = mode
        self.spatial = spatial
        self.spatiotemporal = spatiotemporal
        self.spatial_postfix = spatial_postfix
        self.spatiotemporal_postfix = spatiotemporal_postfix

        if not (spatial or spatiotemporal):
            raise ValueError("At least one of 'spatial' or 'spatiotemporal' must be True")

        anno_path = self.anno_root / f"{mode}_info_ml.npy"
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        loaded = np.load(anno_path, allow_pickle=True).item()
        if isinstance(loaded, dict):
            int_keys = sorted(k for k in loaded.keys() if isinstance(k, int))
            self.data = [loaded[k] for k in int_keys]
            self.prefix = loaded.get("prefix")
        else:
            self.data = loaded
            self.prefix = None

        # CSL-Daily features may be stored either by split subfolders or flat.
        split_spatial_dir = self.feat_root / self.mode
        self.spatial_dir = split_spatial_dir if split_spatial_dir.exists() else self.feat_root

        split_spatiotemporal_dir = self.mae_feat_root / self.mode
        self.spatiotemporal_dir = (
            split_spatiotemporal_dir if split_spatiotemporal_dir.exists() else self.mae_feat_root
        )
        self._validate_directories()

    def _validate_directories(self) -> None:
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature directory not found: {self.spatial_dir}")

        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(f"Spatiotemporal feature directory not found: {self.spatiotemporal_dir}")

    def _load_spatial_features(self, file_id: str) -> torch.Tensor:
        feat_path = self.spatial_dir / f"{file_id}{self.spatial_postfix}.npy"
        if not feat_path.exists():
            raise FileNotFoundError(f"Spatial feature file not found: {feat_path}")
        return torch.tensor(np.load(feat_path))

    def _load_spatiotemporal_features(self, file_id: str) -> Union[torch.Tensor, List[torch.Tensor]]:
        if isinstance(self.spatiotemporal_postfix, str):
            path = self.spatiotemporal_dir / f"{file_id}{self.spatiotemporal_postfix}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
            return torch.tensor(np.load(path))

        features = []
        for postfix in self.spatiotemporal_postfix:
            path = self.spatiotemporal_dir / f"{file_id}{postfix}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Spatiotemporal feature file not found: {path}")
            features.append(torch.tensor(np.load(path)))
        return features

    def __getitem__(self, index: int) -> Dict[str, Any]:
        data = self.data[index]
        file_id = data["fileid"]

        pixel_value = None
        glor_value = None

        if self.spatial:
            try:
                pixel_value = self._load_spatial_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                pixel_value = torch.tensor([])

        if self.spatiotemporal:
            try:
                glor_value = self._load_spatiotemporal_features(file_id)
            except FileNotFoundError as e:
                print(f"Warning: {e}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, str):
                    glor_value = torch.tensor([])
                else:
                    glor_value = [torch.tensor([])]

        result = {
            "pixel_value": pixel_value,
            "glor_value": glor_value,
            "bool_mask_pos": None,
            "text": self._normalize_text(data.get("text", "")),
            "gloss": data.get("gloss", ""),
            "id": file_id,
            "num_frames": len(pixel_value) if pixel_value is not None else 0,
            "vid_path": str(self.vid_root / data.get("folder", "")),
            "lang": "Chinese",
            "en_text": data.get("en_text", ""),
            "es_text": data.get("es_text", ""),
            "fr_text": data.get("fr_text", ""),
            "original_info": data,
        }

        return result

    def _normalize_text(self, text: str) -> str:
        return text.strip()

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
