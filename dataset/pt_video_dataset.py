import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


class PTVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pt_root: str,
        anno_root: str,
        mode: str = "train",
        resize: int = 224,
        frame_stride: int = 1,
        max_frames: Optional[int] = None,
        random_sample: bool = False,
        pt_ext: str = ".pt",
    ) -> None:
        super().__init__()
        self.pt_root = Path(pt_root)
        self.anno_root = Path(anno_root)
        self.mode = mode
        self.resize = resize
        self.frame_stride = max(1, int(frame_stride))
        self.max_frames = max_frames
        self.random_sample = random_sample
        self.pt_ext = pt_ext

        self.data = self._load_annotations()
        self.base_roots = self._build_base_roots()

    def _build_base_roots(self) -> List[Path]:
        roots = []
        mode_root = self.pt_root / self.mode
        if mode_root.exists():
            roots.append(mode_root)
        roots.append(self.pt_root)
        return roots

    def _load_annotations(self) -> List[Dict[str, Any]]:
        candidates = [
            self.anno_root / f"{self.mode}_info.npy",
            self.anno_root / f"{self.mode}_info_ml.npy",
        ]
        anno_path = None
        for cand in candidates:
            if cand.exists():
                anno_path = cand
                break
        if anno_path is None:
            raise FileNotFoundError(
                f"No annotation file found in {self.anno_root} for mode '{self.mode}'"
            )

        raw = np.load(anno_path, allow_pickle=True)
        if isinstance(raw, np.ndarray) and raw.shape == () and raw.dtype == object:
            raw = raw.item()

        if isinstance(raw, dict):
            keys = []
            for key in raw.keys():
                if key == "prefix":
                    continue
                if isinstance(key, (int, np.integer)):
                    keys.append(int(key))
                else:
                    try:
                        keys.append(int(key))
                    except Exception:
                        continue
            keys = sorted(keys)
            return [raw[k] for k in keys]
        if isinstance(raw, np.ndarray):
            return raw.tolist()
        raise ValueError(f"Unsupported annotation format at {anno_path}")

    def _candidate_paths(self, item: Dict[str, Any]) -> List[Path]:
        candidates = []
        fileid = item.get("fileid") or item.get("id")
        folder = item.get("folder")

        folder_rel = None
        folder_rel_no_mode = None
        if folder:
            folder_rel = str(folder).replace("\\", "/")
            if "*" in folder_rel:
                folder_rel = folder_rel.split("*")[0]
            folder_rel = folder_rel.rstrip("/").rstrip(os.sep)
            if folder_rel.startswith(f"{self.mode}/"):
                folder_rel_no_mode = folder_rel[len(self.mode) + 1 :]
            else:
                folder_rel_no_mode = folder_rel

        for base in self.base_roots:
            if fileid:
                candidates.append(base / f"{fileid}{self.pt_ext}")
            if folder_rel:
                rel = folder_rel_no_mode if base.name == self.mode else folder_rel
                if rel:
                    candidates.append(base / f"{rel}{self.pt_ext}")
        return candidates

    def _resolve_pt_path(self, item: Dict[str, Any]) -> Path:
        for path in self._candidate_paths(item):
            if path.exists():
                return path
        fileid = item.get("fileid") or item.get("id")
        folder = item.get("folder")
        candidates = [str(p) for p in self._candidate_paths(item)]
        raise FileNotFoundError(
            f"Missing .pt file for fileid={fileid}, folder={folder}. Tried: {candidates}"
        )

    def _reshape_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() == 3:
            frames = frames.unsqueeze(0)
        if frames.dim() != 4:
            raise ValueError(f"Expected frames with 4 dims, got {frames.shape}")
        if frames.shape[1] != 3 and frames.shape[-1] == 3:
            frames = frames.permute(0, 3, 1, 2)
        return frames

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if self.frame_stride > 1:
            frames = frames[:: self.frame_stride]
        if self.max_frames is None or frames.shape[0] <= self.max_frames:
            return frames
        if self.random_sample:
            indices = torch.randperm(frames.shape[0])[: self.max_frames]
            return frames[indices]
        return frames[: self.max_frames]

    def __getitem__(self, index: int) -> torch.Tensor:
        item = self.data[index]
        pt_path = self._resolve_pt_path(item)
        frames = torch.load(pt_path)
        frames = self._reshape_frames(frames)
        frames = frames.float()
        if frames.max() > 1.0:
            frames = frames / 255.0
        frames = self._sample_frames(frames)
        frames = F.interpolate(
            frames,
            size=(self.resize, self.resize),
            mode="bilinear",
            align_corners=False,
        )
        return frames

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def collate_fn(batch: List[torch.Tensor]) -> torch.Tensor:
        if len(batch) == 0:
            return torch.empty(0)
        return torch.cat(batch, dim=0)
