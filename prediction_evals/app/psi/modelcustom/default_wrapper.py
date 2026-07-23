"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

PSI-0.5 wrapper for IntPhys2 prediction-based evaluation.

modelcustom API requirements (same as all other wrappers in this repo):

  init_module(frames_per_clip, nb_context_frames, checkpoint,
              model_kwargs, wrapper_kwargs) -> nn.Module

  The returned module's forward(x) must satisfy:
    :param x:       Video clip [B, C, T, H, W]  (ImageNet-normalised, float32)
    :returns:       (preds, targets) each [B, n_pred * N_patches, patch_dim]
    where F.l1_loss(preds, targets) is the per-window surprise score.

------------------------------------------------------------------------------

Architecture & adaptation strategy
====================================
PSI-0.5 is an autoregressive transformer that predicts future frames
token-by-token in a discrete visual codebook.  The IntPhys2 harness expects
a (preds, targets) pair in a continuous patch embedding space.

Multi-frame prediction strategy (matching V-JEPA / VideoMAEv2):

  1. Extract all context frames (frames 0 … n_context-1) → fed to PSI.
  2. PSI.generate("rgb0,...,rgb{ctx}->rgb{tgt_start},...,rgb{tgt_end}", ...)
       → returns a tuple of n_pred PIL Images in a single call.
  3. Patchify each predicted frame AND each ground-truth future frame with
     per-patch normalisation (VideoMAEv2 convention).
  4. Concatenate across all predicted frames:
         preds   = cat([patchify(pred_j) for j], dim=1)  → [B, n_pred*N, D]
         targets = cat([patchify(gt_j)   for j], dim=1)  → [B, n_pred*N, D]
  5. Return (preds, targets); harness computes F.l1_loss(...).mean((1,2)).

Frames-to-predict behaviour (controlled by num_frames_to_pred in the config):
  - num_frames_to_pred == -1  : predict ALL remaining frames after the context.
    eval.py keeps model.frames_per_clip == frames_per_clip (full clip).
  - num_frames_to_pred == K   : predict exactly K frames after the context.
    eval.py sets model.frames_per_clip = nb_context_frames + K before
    calling unfold(), so the clips fed to forward() are already the right
    length.  n_pred = T - n_context = K automatically.
  - Safety clamp: if num_frames_to_pred exceeds what is actually available
    in the clip (T - n_context), n_pred is clamped down to the maximum
    available (≥ 1).  eval.py applies an analogous clip-length clamp before
    the unfold so that 0-window situations are avoided.

Visualization:
  Two animated GIFs are saved per window (if viz_dir is set):
    context_frame{raw_ctx:05d}_ctx{n:02d}.gif
        Loops through the n_context context frames.
    pred_frame{raw_tgt:05d}_ctx{n:02d}_L1_{score:.4f}.gif
        One animation frame per predicted step; each frame is a 3-panel
        side-by-side: PSI Prediction | Ground Truth | Diff×5.
        The L1 in the filename is the mean over all predicted frames.

Key choices:
  - Patch size    : 16×16 pixels (VideoMAEv2 convention).
                    For 224×224 inputs → 14×14 = 196 patches/frame.
  - Per-patch norm: zero-mean, unit-std (VideoMAEv2 convention).
  - GIF delay     : gif_duration_ms (default 200 ms per frame).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from einops import rearrange

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# PSI uses 16×16 pixel patches internally.
# For 224×224 input: 14×14 = 196 patches per frame, each 16×16×3 = 768 dims.
PSI_PATCH_SIZE = 16


# ---------------------------------------------------------------------------
# Public entry point called by eval.py's init_module()
# ---------------------------------------------------------------------------

def init_module(
    frames_per_clip: int,
    nb_context_frames: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    **kwargs,
) -> "AnticipativePSIWrapper":
    """
    Load PSI-0.5 from HuggingFace (or a local path) and return a wrapped
    nn.Module compatible with the IntPhys2 prediction eval harness.

    Args:
        frames_per_clip:    Total frames per sliding window (e.g. 16).
        nb_context_frames:  Initial context length (mutated at runtime).
        checkpoint:         HuggingFace repo ID or local directory.
        model_kwargs:       Contents of the YAML ``pretrain_kwargs`` block.
                            Recognised key: ``resolution`` (int, default 224).
        wrapper_kwargs:     Contents of the YAML ``wrapper_kwargs`` block.
                            Recognised keys:
                              gen_temp        (float, default 1.0)
                              gen_top_k       (int,   default 1000)
                              gen_top_p       (float, default 1.0)
                              gen_seed        (int,   default 42)
                              viz_dir         (str|None, default None)
                              viz_stride      (int,   default 2)
                              viz_frame_step  (int,   default 10)
                              gif_duration_ms (int,   default 200)
    """
    from transformers import AutoModel  # type: ignore

    psi_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading PSI-0.5 from '{checkpoint}' onto {psi_device} ...")

    psi_predictor = AutoModel.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        device=psi_device,
    )

    resolution = (model_kwargs or {}).get("resolution", 224)

    model = AnticipativePSIWrapper(
        psi_predictor=psi_predictor,
        frames_per_clip=frames_per_clip,
        nb_context_frames=nb_context_frames,
        resolution=resolution,
        **(wrapper_kwargs or {}),
    )

    return model


# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------

class AnticipativePSIWrapper(nn.Module):
    """
    Wraps PSI-0.5 (``PSI2Predictor``) as a multi-frame prediction-surprise
    module compatible with V-JEPA / VideoMAEv2's evaluation protocol.

    The eval loop updates three *mutable* attributes before each forward pass:

        self.nb_context_frames  — how many leading frames are the "context"
        self.frames_per_clip    — total frames in the sliding window
        self.grid_depth         — frames_per_clip // 2 (dummy; not used here)

    forward(x: [B, C, T, H, W]) -> (preds, targets)
        preds, targets are each [B, n_pred * N_patches, patch_dim]
        where n_pred = clamp(T - nb_context_frames, min=1)
    """

    def __init__(
        self,
        psi_predictor,
        frames_per_clip: int = 16,
        nb_context_frames: int = 8,
        resolution: int = 224,
        gen_temp: float = 1.0,
        gen_top_k: int = 1000,
        gen_top_p: float = 1.0,
        gen_seed: int = 42,
        viz_dir: str | None = None,
        viz_stride: int = 2,
        viz_frame_step: int = 10,
        gif_duration_ms: int = 200,
    ):
        super().__init__()

        # PSI2Predictor is NOT an nn.Module — stored as a plain attribute so
        # that .to(), .eval(), and parameter iteration do not affect it.
        self.psi_predictor = psi_predictor

        # Mutable attributes read/written by the eval loop
        self.frames_per_clip = frames_per_clip
        self.nb_context_frames = nb_context_frames
        self.grid_depth = frames_per_clip // 2

        self.resolution = resolution

        # PSI generation hyper-parameters
        self.gen_temp = gen_temp
        self.gen_top_k = gen_top_k
        self.gen_top_p = gen_top_p
        self.gen_seed = gen_seed

        # Visualization state
        self.viz_dir = viz_dir
        self.gif_duration_ms = gif_duration_ms
        # Set by eval.py to group GIFs into per-video subfolders.
        self.current_video_name: str = "unknown"
        # Set by eval.py to enable absolute frame-index computation.
        self._viz_stride: int = viz_stride
        self._viz_frame_step: int = viz_frame_step
        # True while eval.py is in the max_context_mode inner loop.
        self._viz_is_max_context: bool = True
        # Global window index of the first item in the current chunk.
        self._viz_chunk_offset: int = 0

        # ImageNet normalisation constants applied by the IntPhys2 transform.
        # Registered as buffers so they follow .to(device) automatically.
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, T, H, W] — ImageNet-normalised video clip (float32).
               T is exactly model.frames_per_clip (set by eval.py before
               each unfold), which already encodes num_frames_to_pred.

        Returns:
            preds:   [B, n_pred * N_patches, patch_dim]
            targets: [B, n_pred * N_patches, patch_dim]
            where n_pred = clamp(T - nb_context_frames, min=1).
        """
        B, C, T, H, W = x.shape
        x_f32 = x.float()

        # ------------------------------------------------------------------ #
        # Compute frame indices
        # Clamp n_context so there is always at least 1 frame to predict.
        # This also handles the edge case where num_frames_to_pred is set
        # to a value exceeding the available clip length.
        # ------------------------------------------------------------------ #
        n_context = min(self.nb_context_frames, T - 1)   # ≥ 1 context frame, ≤ T-1
        tgt_start = n_context                             # index of first future frame
        tgt_end   = T - 1                                 # index of last future frame
        n_pred    = tgt_end - tgt_start + 1               # ≥ 1 by construction

        logger.info(
            f"PSI forward: T={T}, n_context={n_context}, "
            f"n_pred={n_pred}, notation-range=rgb{tgt_start}..rgb{tgt_end}"
        )

        # ------------------------------------------------------------------ #
        # Denormalise all frames to [0, 1]
        # ------------------------------------------------------------------ #
        # context_pixels[i] : [B, 3, H, W] for frame i
        context_pixels: list[torch.Tensor] = [
            (x_f32[:, :, i] * self.img_std + self.img_mean).clamp(0.0, 1.0)
            for i in range(n_context)
        ]
        # target_pixels[j] : [B, 3, H, W] for future frame tgt_start + j
        target_pixels: list[torch.Tensor] = [
            (x_f32[:, :, tgt_start + j] * self.img_std + self.img_mean).clamp(0.0, 1.0)
            for j in range(n_pred)
        ]

        # ------------------------------------------------------------------ #
        # Build PSI multi-output notation
        # e.g. n_context=4, predict frames 4..15:
        #   "rgb0,rgb1,rgb2,rgb3->rgb4,rgb5,...,rgb15"
        # ------------------------------------------------------------------ #
        in_side  = ",".join(f"rgb{i}" for i in range(n_context))
        out_side = ",".join(f"rgb{i}" for i in range(tgt_start, tgt_end + 1))
        notation = f"{in_side}->{out_side}"

        # ------------------------------------------------------------------ #
        # Run PSI sequentially over batch samples (PSI is not natively batched)
        # preds_per_sample[b] = list of n_pred PIL Images
        # ------------------------------------------------------------------ #
        preds_per_sample: list[list[Image.Image]] = []

        for b in range(B):
            # Convert every context frame to a uint8 PIL Image.
            rgb_kwargs: dict[str, Image.Image] = {}
            for i in range(n_context):
                frame_np = (
                    context_pixels[i][b]    # [3, H, W]
                    .permute(1, 2, 0)       # [H, W, 3]
                    .cpu()
                    .numpy()
                )
                frame_np = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
                rgb_kwargs[f"rgb{i}"] = Image.fromarray(frame_np)

            with torch.no_grad():
                raw_output = self.psi_predictor.generate(
                    notation,
                    **rgb_kwargs,
                    temp=self.gen_temp,
                    top_k=self.gen_top_k,
                    top_p=self.gen_top_p,
                    seed=self.gen_seed,
                )

            # ---- DEBUG: log PSI return structure --------------------------------
            logger.info(f"[PSI DEBUG] type(raw_output)={type(raw_output)}")
            if isinstance(raw_output, dict):
                logger.info(f"[PSI DEBUG] outer dict keys={list(raw_output.keys())}")
                first_val = next(iter(raw_output.values()), None)
                logger.info(f"[PSI DEBUG] first value type={type(first_val)}")
                if isinstance(first_val, dict):
                    logger.info(f"[PSI DEBUG] inner dict keys={list(first_val.keys())}")
                    first_inner = next(iter(first_val.values()), None)
                    logger.info(f"[PSI DEBUG] inner dict first value type={type(first_inner)}")
                elif not isinstance(first_val, Image.Image):
                    logger.info(f"[PSI DEBUG] first value repr={repr(first_val)[:200]}")
            elif isinstance(raw_output, (tuple, list)):
                logger.info(f"[PSI DEBUG] sequence len={len(raw_output)}")
                if raw_output:
                    first_val = raw_output[0]
                    logger.info(f"[PSI DEBUG] first element type={type(first_val)}")
                    if isinstance(first_val, dict):
                        logger.info(f"[PSI DEBUG] first element keys={list(first_val.keys())}")
                    elif not isinstance(first_val, Image.Image):
                        logger.info(f"[PSI DEBUG] first element repr={repr(first_val)[:200]}")
            else:
                logger.info(f"[PSI DEBUG] repr={repr(raw_output)[:200]}")
            # ---- end DEBUG ------------------------------------------------------

            if isinstance(raw_output, dict):
                raw_output = [raw_output[f"rgb{i}"] for i in range(tgt_start, tgt_end + 1)]
            elif not isinstance(raw_output, (tuple, list)):
                raw_output = (raw_output,)

            pil_preds: list[Image.Image] = []
            for pil in raw_output:
                if not isinstance(pil, Image.Image):
                    raise TypeError(
                        f"PSI generate() returned unexpected type {type(pil)}; "
                        f"expected PIL.Image.Image. notation='{notation}'"
                    )
                pil = pil.convert("RGB")
                if pil.size != (W, H):
                    pil = pil.resize((W, H), Image.BILINEAR)
                pil_preds.append(pil)

            if len(pil_preds) != n_pred:
                raise ValueError(
                    f"PSI returned {len(pil_preds)} frames but {n_pred} were "
                    f"expected (notation='{notation}')"
                )

            preds_per_sample.append(pil_preds)


        # ------------------------------------------------------------------ #
        # Visualization: save context.gif and prediction.gif per batch sample
        # ------------------------------------------------------------------ #
        if self.viz_dir is not None:
            for b in range(B):
                # Compute absolute sampled-frame index of the first predicted frame.
                if self._viz_is_max_context:
                    abs_tgt_start = tgt_start
                else:
                    window_idx = self._viz_chunk_offset + b
                    abs_tgt_start = window_idx * self._viz_stride + tgt_start

                context_pils = [
                    Image.fromarray(
                        (context_pixels[i][b].permute(1, 2, 0).cpu().numpy() * 255)
                        .clip(0, 255).astype(np.uint8)
                    )
                    for i in range(n_context)
                ]
                gt_pils = [
                    Image.fromarray(
                        (target_pixels[j][b].permute(1, 2, 0).cpu().numpy() * 255)
                        .clip(0, 255).astype(np.uint8)
                    )
                    for j in range(n_pred)
                ]
                self._save_gifs(
                    n_context=n_context,
                    abs_tgt_start=abs_tgt_start,
                    context_pils=context_pils,
                    psi_preds=preds_per_sample[b],
                    gt_pils=gt_pils,
                )

        # ------------------------------------------------------------------ #
        # Patchify and concatenate across all predicted frames
        # ------------------------------------------------------------------ #
        # Convert each list of per-sample PIL images to a batched [B, 3, H, W]
        # tensor, then patchify.
        pred_patches_list: list[torch.Tensor] = []
        tgt_patches_list: list[torch.Tensor] = []

        for j in range(n_pred):
            # Predicted frame j: gather from all batch samples
            pred_batch = torch.stack([
                torch.from_numpy(
                    np.array(preds_per_sample[b][j]).astype(np.float32) / 255.0
                ).permute(2, 0, 1)
                for b in range(B)
            ]).to(x.device)   # [B, 3, H, W]

            pred_patches_list.append(self._patchify(pred_batch))          # [B, N, D]
            tgt_patches_list.append(self._patchify(target_pixels[j]))     # [B, N, D]

        preds   = torch.cat(pred_patches_list, dim=1)   # [B, n_pred*N, D]
        targets = torch.cat(tgt_patches_list,  dim=1)   # [B, n_pred*N, D]

        return preds, targets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide an image tensor into non-overlapping 16×16 patches and
        normalise each patch independently (zero-mean, unit-std).

        Args:
            x: [B, 3, H, W]  float32, pixel values in [0, 1]

        Returns:
            [B, N_patches, patch_dim]
            where N_patches = (H // 16) * (W // 16)
            and   patch_dim = 16 * 16 * 3 = 768
        """
        p = PSI_PATCH_SIZE
        B, C, H, W = x.shape

        # Crop to nearest multiple of patch size if needed
        H_c = (H // p) * p
        W_c = (W // p) * p
        if H_c != H or W_c != W:
            x = x[:, :, :H_c, :W_c]

        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

        # Per-patch zero-mean, unit-std (VideoMAEv2 convention)
        patch_mean = patches.mean(dim=-1, keepdim=True)
        patch_std  = patches.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
        return (patches - patch_mean) / patch_std

    def _save_gifs(
        self,
        n_context: int,
        abs_tgt_start: int,
        context_pils: list[Image.Image],
        psi_preds: list[Image.Image],
        gt_pils: list[Image.Image],
    ) -> None:
        """
        Save two animated GIFs for a single window / batch sample.

        context_frame{raw_ctx:05d}_ctx{n_context:02d}.gif
            Loops through the n_context context frames with a dark header
            showing the frame index and raw video frame number.

        pred_frame{raw_tgt:05d}_ctx{n_context:02d}_L1_{score:.4f}.gif
            One animation frame per predicted step; each GIF frame is a
            3-panel side-by-side: PSI Prediction | Ground Truth | Diff×5.
            The mean L1 over all predicted frames is encoded in the filename.

        Args:
            n_context:    Number of context frames used.
            abs_tgt_start: Absolute sampled-frame index of the first predicted
                           frame (used to compute raw video frame numbers).
            context_pils: List of n_context PIL Images (context frames).
            psi_preds:    List of n_pred PIL Images (PSI predictions).
            gt_pils:      List of n_pred PIL Images (ground-truth frames).
        """
        n_pred   = len(psi_preds)
        H, W     = context_pils[0].height, context_pils[0].width
        HEADER_H = 24
        GAP      = 3
        dur      = self.gif_duration_ms

        # Absolute raw-video frame index of the last context frame
        abs_ctx_frame     = abs_tgt_start - 1
        raw_ctx_frame     = abs_ctx_frame * self._viz_frame_step
        raw_tgt_start     = abs_tgt_start * self._viz_frame_step

        video_folder = os.path.join(self.viz_dir, self.current_video_name)
        os.makedirs(video_folder, exist_ok=True)

        # ── 1. context.gif ────────────────────────────────────────────────
        ctx_gif_frames: list[Image.Image] = []
        for i, pil in enumerate(context_pils):
            canvas = Image.new("RGB", (W, H + HEADER_H), color=(20, 20, 20))
            canvas.paste(pil, (0, HEADER_H))
            draw = ImageDraw.Draw(canvas)
            raw_frame_i = (abs_tgt_start - n_context + i) * self._viz_frame_step
            draw.text(
                (4, 4),
                f"Context {i+1}/{n_context}  raw_frame={raw_frame_i}  ctx={n_context}",
                fill=(230, 230, 230),
            )
            ctx_gif_frames.append(canvas)

        ctx_gif_path = os.path.join(
            video_folder,
            f"context_frame{raw_ctx_frame:05d}_ctx{n_context:02d}.gif",
        )
        ctx_gif_frames[0].save(
            ctx_gif_path,
            save_all=True,
            append_images=ctx_gif_frames[1:],
            loop=0,
            duration=dur,
        )

        # ── 2. prediction.gif (PSI pred | GT | Diff×5) ───────────────────
        canvas_w = 3 * W + 2 * GAP
        pred_gif_frames: list[Image.Image] = []
        total_l1 = 0.0

        for j, (pil_pred, pil_gt) in enumerate(zip(psi_preds, gt_pils)):
            pred_f = np.array(pil_pred).astype(np.float32) / 255.0
            gt_f   = np.array(pil_gt).astype(np.float32) / 255.0
            diff   = np.abs(pred_f - gt_f)
            l1_j   = float(diff.mean())
            total_l1 += l1_j

            # Amplify diff ×5, map to red channel
            diff_amp = (diff.mean(axis=-1, keepdims=True) * 5.0).clip(0.0, 1.0)
            diff_rgb = np.concatenate(
                [diff_amp, diff_amp * 0.3, np.zeros_like(diff_amp)], axis=-1
            )
            pil_diff = Image.fromarray((diff_rgb * 255).astype(np.uint8))

            canvas = Image.new("RGB", (canvas_w, H + HEADER_H), color=(20, 20, 20))
            draw   = ImageDraw.Draw(canvas)

            raw_frame_j = (abs_tgt_start + j) * self._viz_frame_step
            draw.text(
                (4, 4),
                f"Step {j+1}/{n_pred}  ctx={n_context}  raw_frame={raw_frame_j}",
                fill=(230, 230, 230),
            )

            panels = [
                (pil_pred, "PSI Prediction"),
                (pil_gt,   "Ground Truth"),
                (pil_diff, f"Diff\u00d75  L1={l1_j:.4f}"),
            ]
            for k, (panel, label) in enumerate(panels):
                x_off = k * (W + GAP)
                canvas.paste(panel.resize((W, H), Image.BILINEAR), (x_off, HEADER_H))
                draw.text((x_off + 4, 4), label, fill=(200, 200, 200))

            pred_gif_frames.append(canvas)

        mean_l1 = total_l1 / max(n_pred, 1)
        pred_gif_path = os.path.join(
            video_folder,
            f"pred_frame{raw_tgt_start:05d}_ctx{n_context:02d}_L1_{mean_l1:.4f}.gif",
        )
        pred_gif_frames[0].save(
            pred_gif_path,
            save_all=True,
            append_images=pred_gif_frames[1:],
            loop=0,
            duration=dur,
        )

    def __repr__(self) -> str:
        p = PSI_PATCH_SIZE
        H = W = self.resolution
        n_patches = (H // p) * (W // p)
        patch_dim = p * p * 3
        return (
            f"AnticipativePSIWrapper(\n"
            f"  frames_per_clip={self.frames_per_clip},\n"
            f"  nb_context_frames={self.nb_context_frames},\n"
            f"  resolution={self.resolution},\n"
            f"  patch_size={p}, n_patches={n_patches}, patch_dim={patch_dim},\n"
            f"  gen=(temp={self.gen_temp}, top_k={self.gen_top_k}, "
            f"top_p={self.gen_top_p}, seed={self.gen_seed})\n"
            f")"
        )
