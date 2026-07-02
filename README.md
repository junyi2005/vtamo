# VTaMo: Video–Text Alignment Model for Sign Language Translation

> Official repository for **VTaMo**, a gloss-free sign language translation framework built on **explicit multi-granularity vision–text alignment**.

<p align="center">
  <a href="#-todo--release-plan"><img src="https://img.shields.io/badge/status-code%20coming%20soon-orange"></a>
  <img src="https://img.shields.io/badge/task-Sign%20Language%20Translation-blue">
  <img src="https://img.shields.io/badge/setting-gloss--free-green">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey"></a>
</p>

**Junyi Hu**<sup>1</sup>, **Zhewen He**<sup>1</sup>, **Haomian Huang**<sup>1</sup>, **Aoxiang Yang**<sup>1</sup>, **Yi Fang**<sup>1,2,†</sup>

<sup>1</sup> New York University Abu Dhabi &nbsp;&nbsp; <sup>2</sup> ChatSign Technology &nbsp;&nbsp; <sup>†</sup> Corresponding author

---

## Overview

Sign language translation (SLT) converts continuous sign videos into spoken-language text. Gloss-free approaches leverage pre-trained visual encoders and language models, but rely on **implicit** cross-modal alignment learned inside the decoder from translation supervision alone. This is fragile because sign languages often do not follow the word order of the corresponding spoken language, so visual features along the video timeline are frequently misaligned with the text tokens an autoregressive decoder is trained to predict.

**VTaMo** makes alignment **explicit**. Instead of asking the decoder to simultaneously learn translation *and* silently discover a latent cross-modal permutation, VTaMo learns the correspondence between visual segments and text tokens directly, and presents the decoder with a semantically ordered visual sequence. Because many spoken words (articles, prepositions, auxiliaries) have no sign-level counterpart, VTaMo aligns against and decodes a content-word **pseudo-gloss** of the target rather than the raw sentence, and restores fluent sentences afterwards with a lightweight, text-only recovery model.

<p align="center"><i>Explicit alignment sharpens frame-to-token correspondences, simplifies decoder learning, and improves robustness — especially on large-vocabulary benchmarks.</i></p>

## Method

<p align="center">
  <img src="assets/pipeline.png" width="100%" alt="VTaMo pipeline">
</p>
<p align="center"><i>The VTaMo pipeline. A sign video is encoded by a frozen CLIP-ViT backbone with a lightweight temporal encoder and fusion projection (A, B). Given text embeddings (D), VTaMo performs <b>local alignment</b> with an entropy-regularized OT (Sinkhorn) solver and <b>global alignment</b> via an orthogonal transform with a memory queue (C1, C2). The correspondence drives <b>window reordering</b> (C3) and <b>position-aligned contrastive learning</b> (C4), and a LoRA-adapted Flan-T5 decoder generates the translation (E).</i></p>

VTaMo introduces alignment at **three complementary granularities**, trained jointly with the standard translation objective:

- **① Local alignment — entropy-regularized Optimal Transport.**
  A Sinkhorn solver estimates a soft, fine-grained correspondence between temporal visual segments and pseudo-gloss tokens, using cosine similarity as the transport cost. A single **learnable null token** absorbs transitional gestures and co-articulation frames that correspond to no explicit word, so the model is never forced to map every frame onto a content token. A multi-phase ε-annealing schedule, a temporal-variation term, and a null-cost regularizer keep the learned plan sharp, temporally coherent, and stable.

- **② Global alignment — learnable orthogonal transformation.**
  Because the visual and textual encoders are pre-trained on different modalities, paired sentence embeddings can suffer an orientation mismatch even when semantically equivalent. VTaMo applies a **learnable orthogonal transform** (a rotation that preserves norms and angles) to the pooled visual sentence embedding and calibrates the two spaces through an Earth Mover's Distance objective, computed over a FIFO **memory queue** that diversifies the sentence pairs.

- **③ Position-aligned contrastive learning.**
  Using the transport plan, visual features are **reordered** into target-token order during training and bound to their text-token embeddings via an InfoNCE objective. This gives each visual token discriminative, token-level grounding without altering the frozen language-model embedding space.

**Reordering & inference.** During training, a window-based reordering step places the visual features in target-token order so the decoder's cross-entropy gradient reinforces (rather than fights) the alignment objectives. At inference the target order is unknown, so **no reordering is applied**: the decoder reads visual tokens in signing order and emits a pseudo-gloss; a **text-only recovery model** — trained purely by shuffling content words of plain sentences and learning to reconstruct them — restores spoken word order and re-inserts the dropped function words.

**Architecture.** A frozen **CLIP-ViT-Large** backbone (with multi-scale S²-Wrapper features) encodes frames; an attention-based temporal encoder downsamples them; a fusion projector maps to the language-model space; and a **LoRA-adapted Flan-T5-XL** decoder generates the pseudo-gloss. Only lightweight adapters and alignment modules are trained — the visual and language backbones stay frozen.

## 📌 TODO / Release Plan

We are actively cleaning up the codebase and model artifacts for release. Progress will be tracked here:

- [ ] **Release training code** — end-to-end training pipeline for the alignment objectives and decoder.
- [ ] **Release inference & experiment code** — evaluation scripts, pseudo-gloss recovery, and reproduction of the reported benchmarks.
- [ ] **Release checkpoints** — pre-trained VTaMo weights for the four benchmarks.

⭐ **Star / watch this repo** to be notified when each component lands.

## Citation

If you find VTaMo useful in your research, please consider citing:

```bibtex
@inproceedings{hu2026vtamo,
  title     = {VTaMo: Video-Text Alignment Model for Sign Language Translation},
  author    = {Hu, Junyi and He, Zhewen and Huang, Haomian and Yang, Aoxiang and Fang, Yi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## License

This project is released under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](LICENSE) license. You are free to share and adapt the material for **non-commercial** purposes with appropriate attribution. The license may be updated in the future.

## Acknowledgements

This work was partially supported by ChatSign Technology, Ltd.; and the NYUAD Center for AI and Robotics (CAIR), funded by Tamkeen under the NYUAD Research Institute Award CG010. Computational support was provided by the HPC resources at NYU Abu Dhabi and NYU New York.
