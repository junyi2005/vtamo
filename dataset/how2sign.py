import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class How2Sign(torch.utils.data.Dataset):
    """
    Dataset class for How2Sign features and annotations.

    Loads spatial CLIP features (and optional spatiotemporal features) paired with text.
    """

    def __init__(
        self,
        anno_root: str,
        vid_root: str,
        feat_root: str,
        mae_feat_root: str,
        mode: str = "train",
        spatial: bool = False,
        spatiotemporal: bool = False,
        spatial_postfix: str = "",
        spatiotemporal_postfix: Union[str, List[str]] = "",
        lang: str = "English",
    ) -> None:
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
        self.lang = lang

        if not (spatial or spatiotemporal):
            raise ValueError("At least one of 'spatial' or 'spatiotemporal' must be True")

        self.data = self._load_annotations()

        self.feature_mode = self._resolve_feature_mode()
        self.spatial_dir = self.feat_root / self.feature_mode
        self.spatiotemporal_dir = self.mae_feat_root / self.feature_mode

        self._validate_directories()

    def _validate_directories(self) -> None:
        if self.spatial and not self.spatial_dir.exists():
            raise FileNotFoundError(f"Spatial feature directory not found: {self.spatial_dir}")
        if self.spatiotemporal and not self.spatiotemporal_dir.exists():
            raise FileNotFoundError(
                f"Spatiotemporal feature directory not found: {self.spatiotemporal_dir}"
            )

    def _resolve_feature_mode(self) -> str:
        if self.mode == "val":
            dev_spatial = self.feat_root / "dev"
            dev_spatiotemporal = self.mae_feat_root / "dev"
            if dev_spatial.exists() or dev_spatiotemporal.exists():
                return "dev"
        return self.mode

    def _load_annotations(self) -> List[Dict[str, Any]]:
        candidates = [
            self.anno_root / f"{self.mode}_info_ml.npy",
            self.anno_root / f"{self.mode}_info.npy",
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

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").strip()
        if text and not text.endswith("."):
            text = f"{text}."
        return text

    @staticmethod
    def _get_timecodes(item: Dict[str, Any]) -> Dict[str, Optional[float]]:
        start = item.get("start") or item.get("START") or item.get("START_REALIGNED")
        end = item.get("end") or item.get("END") or item.get("END_REALIGNED")
        return {"start": start, "end": end}

    def _format_postfix(self, postfix: str, item: Dict[str, Any]) -> str:
        if "{" not in postfix:
            return postfix
        times = self._get_timecodes(item)
        try:
            return postfix.format(**times)
        except Exception:
            return postfix

    def _spatial_candidates(self, file_id: str, item: Dict[str, Any]) -> List[str]:
        postfix = self._format_postfix(self.spatial_postfix, item)
        times = self._get_timecodes(item)
        candidates = []
        if times["start"] is not None:
            candidates.append(f"{file_id}_{times['start']}{postfix}.npy")
        candidates.append(f"{file_id}{postfix}.npy")
        return candidates

    def _spatiotemporal_candidates(self, file_id: str, item: Dict[str, Any]) -> List[str]:
        postfixes = self.spatiotemporal_postfix
        if isinstance(postfixes, str):
            postfixes = [postfixes]
        candidates = []
        for postfix in postfixes:
            formatted = self._format_postfix(postfix, item)
            candidates.append(f"{file_id}{formatted}.npy")
        return candidates

    @staticmethod
    def _load_first_existing(base_dir: Path, candidates: List[str], max_retries: int = 3) -> torch.Tensor:
        for name in candidates:
            path = base_dir / name
            if path.exists():
                for attempt in range(max_retries):
                    try:
                        return torch.tensor(np.load(path))
                    except OSError as e:
                        if attempt < max_retries - 1:
                            time.sleep(0.5 * (attempt + 1))
                            continue
                        raise OSError(f"Failed to load {path} after {max_retries} retries: {e}") from e
        raise FileNotFoundError(f"Missing feature file; tried: {candidates}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.data[index]
        file_id = item.get("fileid") or item.get("id")

        pixel_value = torch.tensor([])
        glor_value = None

        if self.spatial:
            try:
                pixel_value = self._load_first_existing(
                    self.spatial_dir, self._spatial_candidates(file_id, item)
                )
            except (FileNotFoundError, OSError) as exc:
                print(f"Warning: {exc}. Returning empty tensor.")
                pixel_value = torch.tensor([])

        if self.spatiotemporal:
            try:
                if isinstance(self.spatiotemporal_postfix, list):
                    glor_value = [
                        self._load_first_existing(self.spatiotemporal_dir, [name])
                        for name in self._spatiotemporal_candidates(file_id, item)
                    ]
                else:
                    glor_value = self._load_first_existing(
                        self.spatiotemporal_dir, self._spatiotemporal_candidates(file_id, item)
                    )
            except (FileNotFoundError, OSError) as exc:
                print(f"Warning: {exc}. Returning empty tensor.")
                if isinstance(self.spatiotemporal_postfix, list):
                    glor_value = [torch.tensor([]) for _ in self.spatiotemporal_postfix]
                else:
                    glor_value = torch.tensor([])

        # Both train and inference read the 'text' field as-is. When the
        # annotation file was built with convert_how2sign_annotations.py +
        # --pseudo_gloss_json (README Step 0 / Step 1), this field already holds
        # pseudo-gloss. 'original_text' (the raw English sentence) is kept
        # alongside it purely for logging / ICL / debugging.
        text = self._normalize_text(item.get("text", ""))
        original_text = item.get("original_text", text)

        return {
            "pixel_value": pixel_value,
            "glor_value": glor_value,
            "bool_mask_pos": None,
            "text": text,
            "gloss": item.get("gloss", ""),
            "id": file_id,
            "num_frames": int(pixel_value.shape[0]) if pixel_value is not None else 0,
            "vid_path": str(self.vid_root / self.mode / f"{file_id}.mp4"),
            "lang": self.lang,
            "en_text": text,  # Also use pseudo gloss for ICL
            "es_text": text,
            "fr_text": text,
            "original_text": original_text,  # Keep original for reference
            "original_info": item,
        }

    def __len__(self) -> int:
        return len(self.data)

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return batch
