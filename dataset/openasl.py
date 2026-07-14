import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
import torch


class OpenASL(torch.utils.data.Dataset):
    """
    Dataset class for OpenASL sign language translation.
    """
    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = "val",
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

        # Load annotations from TSV file
        anno_path = self.anno_root / f"openasl-v1.0-{mode}.tsv"
        if not anno_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        # Read TSV file
        df = pd.read_csv(anno_path, sep='\t')
        if 'tokenized-text' not in df.columns:
            raise KeyError(
                f"OpenASL TSV at {anno_path} has no 'tokenized-text' column. "
                f"Re-run scripts/preprocess/generate_openasl_labels.py with "
                f"--pseudo_gloss_json to bake pseudo-gloss into this column. "
                f"See README Step 0/Step 1."
            )
        self.data = df.to_dict('records')

        print(f"Loaded {len(self.data)} samples for {mode} split")

        # OpenASL features are stored flat (not in split subfolders)
        self.spatial_dir = self.feat_root
        self.spatiotemporal_dir = self.mae_feat_root

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
        # Use 'vid' field as file_id (e.g., "Mci9oyb5V2E-00:00:06.000-00:00:06.589")
        file_id = data['vid']

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

        # 'tokenized-text' holds pseudo-gloss — we enforce this at __init__.
        # No silent fallback to raw-text: the OpenASL pipeline is locked to
        # pseudo-gloss, and feeding raw English here would quietly train the
        # model on function words it cannot align to any frame.
        text = data.get('tokenized-text', '') or ''
        original_text = data.get('raw-text', text) or text

        result = {
            "pixel_value": pixel_value,
            "glor_value": glor_value,
            "bool_mask_pos": None,
            "text": self._normalize_text(text),
            "gloss": data.get('gloss', ''),
            "id": file_id,
            "num_frames": len(pixel_value) if pixel_value is not None else 0,
            "vid_path": str(self.vid_root / data.get('yid', '')),  # yid is the YouTube ID
            "lang": "English",  # OpenASL is in English
            "en_text": self._normalize_text(text),  # For ICL compatibility
            "original_text": original_text,  # Keep original for reference
            "original_info": data,
        }

        return result

    def _normalize_text(self, text: str) -> str:
        """Normalize text by stripping whitespace."""
        return text.strip() if isinstance(text, str) else ""

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def collate_fn(batch: List[Dict]) -> List[Dict]:
        return batch
