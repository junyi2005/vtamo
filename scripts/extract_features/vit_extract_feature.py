import argparse
import os
import os.path as osp
import glob
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, CLIPVisionModel

import sys
sys.path.append('./')

from utils.s2wrapper import forward as multiscale_forward
from utils.helpers import read_video, get_img_list


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)


def build_save_root(args, canonical_ds, mode):
    model_id = os.path.split(args.model_name)[-1]
    fname = f'{model_id}_feat_{canonical_ds}'
    return os.path.join(args.save_dir, fname, mode)


def build_postfix(args, st):
    postfix = ""
    if args.s2_mode != "":
        postfix = f"_{args.s2_mode}"
    if len(args.scales) == 3:
        postfix = f'{postfix}_large'
    if st is not None:
        postfix = f'_{st}{postfix}'
    return postfix



class ViTFeatureReader(object):
    def __init__(
        self,
        model_name='openai/clip-vit-large-patch14',
        cache_dir=None,
        device='cuda:0',
        s2_mode='s2wrapping',
        scales=[1, 2],
        nth_layer=-1,
        projector_path=None,
        image_processor_name=None
    ):
        self.s2_mode = s2_mode
        self.device = device
        self.scales = scales
        self.nth_layer = nth_layer
        self.projector = None

        self.model = CLIPVisionModel.from_pretrained(
            model_name, output_hidden_states=True, cache_dir=cache_dir
        ).to(device).eval()

        if projector_path:
            state = torch.load(projector_path, map_location="cpu")
            in_dim = self.model.config.hidden_size
            out_dim = state["weight"].shape[0]
            self.projector = torch.nn.Linear(in_dim, out_dim)
            self.projector.load_state_dict(state)
            self.projector = self.projector.to(device).eval()

        processor_source = image_processor_name or model_name
        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                processor_source, cache_dir=cache_dir
            )
        except OSError:
            if image_processor_name is None:
                self.image_processor = AutoImageProcessor.from_pretrained(
                    "openai/clip-vit-base-patch16", cache_dir=cache_dir
                )
            else:
                raise

    @torch.no_grad()
    def forward_features(self, inputs):
        outputs = self.model(inputs).hidden_states
        outputs = outputs[self.nth_layer]
        if self.projector is not None:
            outputs = self.projector(outputs)
        return outputs

    @torch.no_grad()
    def get_feats(self, video):
        inputs = self.image_processor(list(video), return_tensors="pt").to(self.device).pixel_values
        if self.s2_mode == "s2wrapping":
            outputs = multiscale_forward(self.forward_features, inputs, scales=self.scales, num_prefix_token=1)
        else:
            outputs = self.forward_features(inputs)
        return outputs[:, 0]


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_root', help='location of tsv files', required=True)
    parser.add_argument('--video_root', help='location of tsv files', required=True)
    parser.add_argument('--device', help='device to use', default='cuda:0')
    parser.add_argument('--s2_mode', default='')
    parser.add_argument('--scales', nargs='+', type=int, help='List of scales', default=[])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--nth_layer', type=int, default=-1)
    parser.add_argument('--cache_dir', help='cache dir for model', default=None)
    parser.add_argument('--projector_path', help='optional projector path', default=None)
    parser.add_argument('--image_processor_name', help='override image processor source', default=None)
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    parser.add_argument('--skip_existing', action='store_true', help='skip if output exists')

    parser.add_argument('--save_dir', help='where to save the output', required=True)
    parser.add_argument('--model_name', help='ViT model name', default='openai/clip-vit-large-patch14')

    return parser

def get_iterator(args, mode):
    batch_size = args.batch_size

    anno_candidates = [
        os.path.join(args.anno_root, f'{mode}_info.npy'),
        os.path.join(args.anno_root, f'{mode}_info_ml.npy')
    ]
    anno_path = None
    for cand in anno_candidates:
        if os.path.exists(cand):
            anno_path = cand
            break
    if anno_path is None:
        raise FileNotFoundError(f"No annotation file found in {args.anno_root} for mode '{mode}' (tried _info.npy and _info_ml.npy)")

    raw = np.load(anno_path, allow_pickle=True)
    if isinstance(raw, np.ndarray) and raw.shape == () and raw.dtype == object:
        raw = raw.item()

    if isinstance(raw, dict):
        # Phoenix-style dict with a 'prefix' entry; keep only int-like keys
        keys = []
        for k in raw.keys():
            if k == 'prefix':
                continue
            if isinstance(k, (int, np.integer)):
                keys.append(int(k))
            else:
                try:
                    keys.append(int(k))
                except Exception:
                    continue
        keys = sorted(keys)
        data = [raw[k] for k in keys]
    elif isinstance(raw, np.ndarray):
        data = raw.tolist()
    else:
        raise ValueError(f"Unsupported annotation format at {anno_path}")

    num = len(data)
    ds_name = osp.split(args.anno_root)[-1]
    ds_name_norm = ds_name.lower()
    if ds_name_norm in ('phoenix14t', 'phoenix-2014t'):
        canonical_ds = 'Phoenix14T'
    elif ds_name_norm in ('csl-daily', 'csldaily', 'csl_daily'):
        canonical_ds = 'CSL-Daily'
    elif ds_name_norm == 'how2sign':
        canonical_ds = 'How2Sign'
    else:
        raise ValueError(f"Unsupported dataset name: {ds_name}")
    reader = ViTFeatureReader(
        args.model_name,
        device=args.device,
        s2_mode=args.s2_mode,
        scales=args.scales,
        nth_layer=args.nth_layer,
        cache_dir=args.cache_dir,
        projector_path=args.projector_path,
        image_processor_name=args.image_processor_name
    )

    save_root = build_save_root(args, canonical_ds, mode)

    def resolve_video_path(base, split):
        if osp.isabs(base):
            return base, osp.exists(base)
        base_path = base
        _, ext = osp.splitext(base_path)
        if ext == '':
            base_path = f"{base_path}.mp4"
        base_stripped = base_path.lstrip('-')
        candidates = []
        if args.video_root:
            candidates.append(osp.join(args.video_root, split, base_path))
            candidates.append(osp.join(args.video_root, base_path))
            if base_stripped != base_path:
                candidates.append(osp.join(args.video_root, split, base_stripped))
                candidates.append(osp.join(args.video_root, base_stripped))
        candidates.append(base_path)
        if base_stripped != base_path:
            candidates.append(base_stripped)

        for cand in candidates:
            if osp.exists(cand):
                return cand, True
        return candidates[0], False

    def iterate():
        for i in range(num):
            fname = data[i]['folder']
            file_id = data[i]['fileid']
            st_value = None

            if canonical_ds in ('Phoenix14T', 'CSL-Daily'):
                postfix = build_postfix(args, st_value)
                if args.skip_existing:
                    save_path = osp.join(save_root, f'{file_id}{postfix}.npy')
                    if osp.exists(save_path):
                        yield None, file_id, st_value
                        continue

                image_list = get_img_list(canonical_ds, args.video_root, fname)
                videos = [Image.open(image).convert('RGB') for image in image_list]
                if len(videos) == 0:
                    pt_path = osp.join(
                        args.video_root,
                        "features",
                        "fullFrame-256x256px",
                        mode,
                        f"{data[i]['fileid']}.pt",
                    )
                    if osp.exists(pt_path):
                        frames = torch.load(pt_path)
                        if frames.dim() == 4 and frames.shape[1] == 3:
                            frames = frames.permute(0, 2, 3, 1)
                        if frames.dtype != torch.uint8:
                            if frames.max() <= 1.0:
                                frames = (frames * 255.0).clamp(0, 255)
                            frames = frames.to(torch.uint8)
                        videos = [frame.numpy() for frame in frames]
                    else:
                        # mp4 fallback for augmented data
                        mp4_path = osp.join(args.video_root, f"{file_id}.mp4")
                        if not osp.exists(mp4_path):
                            mp4_path = osp.join(args.video_root, fname)
                        if osp.exists(mp4_path) and mp4_path.endswith('.mp4'):
                            videos = read_video(mp4_path)
                        else:
                            print(f"Warning: no frames found for {file_id}")

                video_feats = []
                for j in range(0, len(videos), batch_size):
                    video_batch = videos[j:min(j + batch_size, len(videos))]
                    feats = reader.get_feats(video_batch).cpu().numpy()
                    video_feats.append(feats)

                if len(video_feats) == 0:
                    yield [], file_id, st_value
                else:
                    yield np.concatenate(video_feats, axis=0), file_id, st_value

            else:
                if canonical_ds == 'How2Sign':
                    start_time = data[i].get('start') or data[i].get('START') or data[i].get('START_REALIGNED')
                    end_time = data[i].get('end') or data[i].get('END') or data[i].get('END_REALIGNED')
                    orig_start, orig_end = start_time, end_time
                    video_path = fname
                    # If we already have sentence-level clips (folder == fileid), ignore timecodes.
                    if data[i].get('folder') == data[i].get('fileid'):
                        start_time, end_time = None, None

                    video_path, found = resolve_video_path(video_path, mode)
                    if not found:
                        alt_base = data[i].get('video_name')
                        if alt_base:
                            alt_path, alt_found = resolve_video_path(alt_base, mode)
                            if alt_found:
                                video_path = alt_path
                                start_time, end_time = orig_start, orig_end

                    st_value = str(start_time) if start_time is not None else None
                    postfix = build_postfix(args, st_value)
                    if args.skip_existing:
                        save_path = osp.join(save_root, f'{file_id}{postfix}.npy')
                        if osp.exists(save_path):
                            yield None, file_id, st_value
                            continue
                    videos = read_video(video_path, start_time=start_time, end_time=end_time)
                    # If timecodes are out of bounds and no frames returned, retry full clip
                    if len(videos) == 0 and (start_time is not None or end_time is not None):
                        videos = read_video(video_path, start_time=None, end_time=None)
                else:
                    raise ValueError(f"Unsupported dataset name: {ds_name}")

                if len(videos) > 0:
                    video_feats = []
                    for j in range(0, len(videos), batch_size):
                        video_batch = videos[j:min(j + batch_size, len(videos))]
                        feats = reader.get_feats(video_batch).cpu().numpy()
                        video_feats.append(feats)
                    yield np.concatenate(video_feats, axis=0), file_id, st_value
                else:
                    yield [], file_id, st_value

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()
    mode = args.splits
    for m in mode:

        ds_name = osp.split(args.anno_root)[-1]
        ds_name_norm = ds_name.lower()
        if ds_name_norm == 'how2sign':
            canonical_ds = 'How2Sign'
        elif ds_name_norm in ('phoenix14t', 'phoenix-2014t'):
            canonical_ds = 'Phoenix14T'
        elif ds_name_norm in ('csl-daily', 'csldaily', 'csl_daily'):
            canonical_ds = 'CSL-Daily'
        else:
            canonical_ds = ds_name
        _model_name = os.path.split(args.model_name)[-1]
        fname = f'{_model_name}_feat_{canonical_ds}'

        os.makedirs(osp.join(args.save_dir, fname, m), exist_ok=True)

        if canonical_ds == 'How2Sign':
            if m == 'dev': _m = 'val'
            else: _m = m
        elif ds_name == 'NIASL2021':
            if m == 'dev': _m = 'validation'
        else:
            _m = m

        generator, num = get_iterator(args, _m)
        iterator = generator()

        for vit_feat in tqdm.tqdm(iterator, total=num):
            feats, id, st = vit_feat
            save_path = build_save_root(args, canonical_ds, m)

            if feats is None:
                continue
            postfix = build_postfix(args, st)

            np.save(osp.join(save_path, f'{id}{postfix}.npy'), feats)


if __name__ == "__main__":
    main()
