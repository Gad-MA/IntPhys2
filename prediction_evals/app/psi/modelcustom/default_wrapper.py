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
    :param x:       Video clip [B, C, T, H, W]  (ImageNet-normalised, float32 or bfloat16)
    :returns:       (preds, targets) each [B, N_patches, patch_dim]
    where L1(preds, targets) is the per-window surprise score.

------------------------------------------------------------------------------

Architecture & adaptation strategy
====================================
PSI-0.5 is an autoregressive transformer that predicts future frames
token-by-token in a discrete visual codebook.  The IntPhys2 harness expects
a (preds, targets) pair in a continuous patch embedding space.

We bridge the gap using the **pixel-space L1** strategy (same as VideoMAEv2):

  1. Extract the LAST context frame  → fed to PSI as rgb0
  2. PSI.generate("rgb0->rgb1", ...)  → predicted next frame (PIL Image)
  3. Patchify both predicted & actual future frames with per-patch normalisation
  4. Return (preds_patches, target_patches), both [B, N_patches, patch_dim]

The evaluation harness then computes F.l1_loss(preds, targets) as the
surprise score: a higher score means PSI was more shocked by what happened,
which — for an impossible event — should be systematically larger than for
the paired possible video.

Key choices (matching the plan's recommendations):
  - Context feeding  : Option A — only the LAST context frame is given to PSI.
                       Rationale: simplicity; PSI has no native multi-frame
                       conditioning notation without chaining.
  - Resolution       : PSI is run at whatever resolution the video is cropped
                       to (224×224 by default in IntPhys2), because PSI is
                       patch-based and resolution-flexible.  The generated
                       PIL image is resized back to (H, W) before patchifying.
  - Patch size       : 16×16 pixels (matching PSI's internal patch grid).
                       For 224×224 inputs this yields 14×14 = 196 patches/frame.

Performance note
================
PSI generates frames autoregressively, so each forward() call is slower than
feature-space models.  Expect ~10-60 s/video depending on GPU and resolution.
Start validation on the Debug split (60 videos) before running the Main set.
"""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from einops import rearrange

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# PSI uses 16×16 pixel patches internally.
# For 224×224 input: 14 × 14 = 196 patches per frame, each 16×16×3 = 768 dims.
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
    nn.Module that is compatible with the IntPhys2 prediction eval harness.

    Args:
        frames_per_clip:    Total frames per sliding window (e.g. 16).
        nb_context_frames:  Initial context length (will be mutated by the eval
                            loop at runtime, so the value here is just a placeholder).
        checkpoint:         HuggingFace repo ID or local directory, e.g.
                            "StanfordNeuroAILab/psi0_5".
        model_kwargs:       Contents of the YAML ``pretrain_kwargs`` block.
                            Recognised keys: ``resolution`` (int, default 224).
        wrapper_kwargs:     Contents of the YAML ``wrapper_kwargs`` block.
                            Recognised keys:
                              gen_temp  (float, default 1.0)
                              gen_top_k (int,   default 1000)
                              gen_top_p (float, default 1.0)
                              gen_seed  (int,   default 42)
    """
    # ------------------------------------------------------------------
    # Load PSI via the Transformers AutoModel shim.
    # PSI manages its own device internally, so we detect the best one here.
    # The eval harness limits each process to one visible GPU via
    # CUDA_VISIBLE_DEVICES, so cuda:0 is always the correct target.
    # ------------------------------------------------------------------
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
    Wraps PSI-0.5 (``PSI2Predictor``) as a prediction-surprise module.

    The module exposes three *mutable* attributes that the eval loop updates
    before every forward pass (matching the V-JEPA / VideoMAEv2 convention):

        self.nb_context_frames  — how many leading frames are the "context"
        self.frames_per_clip    — total frames in the sliding window
        self.grid_depth         — frames_per_clip // 2 (dummy; not used here)

    forward(x: [B, C, T, H, W]) -> (preds, targets)
        both tensors of shape [B, N_patches, patch_dim]
        where N_patches = (H // PSI_PATCH_SIZE) * (W // PSI_PATCH_SIZE)
        and   patch_dim = PSI_PATCH_SIZE * PSI_PATCH_SIZE * 3
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
    ):
        super().__init__()

        # PSI2Predictor is NOT an nn.Module — stored as a plain attribute so
        # that .to(), .eval(), and parameter iteration do not affect it.
        self.psi_predictor = psi_predictor

        # Mutable attributes read/written by the eval loop
        self.frames_per_clip = frames_per_clip
        self.nb_context_frames = nb_context_frames
        self.grid_depth = frames_per_clip // 2  # updated externally

        self.resolution = resolution

        # PSI generation hyper-parameters
        self.gen_temp = gen_temp
        self.gen_top_k = gen_top_k
        self.gen_top_p = gen_top_p
        self.gen_seed = gen_seed

        # Visualization: if viz_dir is set, a 4-panel comparison image
        # (context | PSI prediction | ground truth | diff) is written to
        # disk immediately after each PSI generation call.
        self.viz_dir = viz_dir
        self._sample_count = 0      # monotonically increasing across all forward() calls
        # Set this attribute before each video to group frames into named subfolders.
        # eval.py sets it via:  model.current_video_name = video_stem
        self.current_video_name: str = "unknown"
        self._current_video_frame_count = 0   # resets when current_video_name changes
        self._prev_video_name: str = ""
        # ---- Attributes written by eval.py to enable absolute frame indexing ----
        # stride between windows (matches stride_sliding_window in config)
        self._viz_stride: int = viz_stride
        # frame_step used when subsampling the video (matches frame_steps in config).
        # raw_frame = sampled_frame × viz_frame_step
        self._viz_frame_step: int = viz_frame_step
        # True while eval.py is in the max_context_mode inner loop;
        # False during the main sliding-window loop.
        self._viz_is_max_context: bool = True
        # Global window index of the first item in the current chunk.
        # eval.py sets this to chunk_id * CHUNK_SIZE before each chunk call.
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
            x: [B, C, T, H, W] — ImageNet-normalised video clip.
               May be float32 or bfloat16 (eval uses autocast).

        Returns:
            preds:   [B, N_patches, patch_dim]   PSI-generated future frame patches
            targets: [B, N_patches, patch_dim]   Actual future frame patches
        """
        B, C, T, H, W = x.shape

        # Work in float32 throughout (PIL/numpy do not support bfloat16).
        x_f32 = x.float()

        # Determine context boundary.  Clamp so we never go out of bounds.
        ctx_idx = min(self.nb_context_frames - 1, T - 2)
        tgt_idx = ctx_idx + 1          # frame to predict
        n_context = ctx_idx + 1        # total context frames: 0 … ctx_idx

        # ------------------------------------------------------------------
        # Denormalise all context frames and the target frame to [0, 1].
        # ------------------------------------------------------------------
        # context_pixels_list[i]: [B, 3, H, W] for frame i in the window
        context_pixels_list = []
        for i in range(n_context):
            frame = x_f32[:, :, i, :, :]
            pixels = (frame * self.img_std + self.img_mean).clamp(0.0, 1.0)
            context_pixels_list.append(pixels)

        target_pixels = (
            (x_f32[:, :, tgt_idx, :, :] * self.img_std + self.img_mean)
            .clamp(0.0, 1.0)
        )

        # ------------------------------------------------------------------
        # Build PSI notation dynamically from the context length.
        #
        # Example — ctx_idx=3 (4 context frames):
        #   notation  = "rgb0,rgb1,rgb2,rgb3->rgb4"
        #   kwargs    = {rgb0: PIL0, rgb1: PIL1, rgb2: PIL2, rgb3: PIL3}
        #   output    = (PIL4,)   → unwrap → PIL4
        #
        # This uses PSI's native multi-RGB conditioning, giving the model full
        # temporal context rather than just the last frame.
        # ------------------------------------------------------------------
        input_side  = ",".join(f"rgb{i}" for i in range(n_context))
        notation    = f"{input_side}->rgb{tgt_idx}"

        # ------------------------------------------------------------------
        # Run PSI on each sample in the batch sequentially.
        # (PSI is autoregressive and not natively batched.)
        # ------------------------------------------------------------------
        predicted_pixels_list: list[torch.Tensor] = []

        for b in range(B):
            # Convert every context frame to a uint8 PIL Image.
            rgb_kwargs: dict[str, Image.Image] = {}
            for i in range(n_context):
                frame_np = (
                    context_pixels_list[i][b]   # [3, H, W]
                    .permute(1, 2, 0)           # [H, W, 3]
                    .cpu()
                    .numpy()
                )
                frame_np = (frame_np * 255.0).clip(0, 255).astype(np.uint8)
                rgb_kwargs[f"rgb{i}"] = Image.fromarray(frame_np)

            # The last context frame is used as the reference in the viz panel.
            pil_context = rgb_kwargs[f"rgb{ctx_idx}"]

            # Generate the next frame.
            # PSI.generate() returns a tuple — one element per output variable
            # in the notation.  We request exactly one output (rgb{tgt_idx}).
            with torch.no_grad():
                raw_output = self.psi_predictor.generate(
                    notation,
                    **rgb_kwargs,
                    temp=self.gen_temp,
                    top_k=self.gen_top_k,
                    top_p=self.gen_top_p,
                    seed=self.gen_seed,
                )

            # Unwrap the 1-tuple PSI always returns
            if isinstance(raw_output, (tuple, list)):
                pil_pred = raw_output[0]
            else:
                pil_pred = raw_output

            # Guard against unexpected return types
            if not isinstance(pil_pred, Image.Image):
                raise TypeError(
                    f"PSI generate() returned unexpected type: {type(pil_pred)}. "
                    f"Expected PIL.Image.Image. notation='{notation}', "
                    f"full output: {raw_output!r}"
                )
            pil_pred = pil_pred.convert("RGB")

            # Ensure output matches the input spatial size
            if pil_pred.size != (W, H):
                pil_pred = pil_pred.resize((W, H), Image.BILINEAR)

            # ── real-time visualization ────────────────────────────────────
            if self.viz_dir is not None:
                # Compute the absolute sampled-frame index that was predicted.
                if self._viz_is_max_context:
                    predicted_frame_idx = tgt_idx
                else:
                    window_idx = self._viz_chunk_offset + b
                    predicted_frame_idx = window_idx * self._viz_stride + tgt_idx

                self._save_visualization(
                    predicted_frame_idx=predicted_frame_idx,
                    pil_context=pil_context,
                    pil_pred=pil_pred,
                    target_pixels_b=target_pixels[b],
                )
            self._sample_count += 1
            # ──────────────────────────────────────────────────────────────

            # Back to float32 tensor in [0, 1]: [3, H, W]
            pred_np = np.array(pil_pred).astype(np.float32) / 255.0
            pred_tensor = torch.from_numpy(pred_np).permute(2, 0, 1)
            predicted_pixels_list.append(pred_tensor)


        # Stack into [B, 3, H, W] and move to the same device as x
        predicted_pixels = torch.stack(predicted_pixels_list).to(x.device)

        # ------------------------------------------------------------------
        # Patchify both tensors with per-patch normalisation.
        # This mirrors the VideoMAEv2 convention already used in the repo.
        # ------------------------------------------------------------------
        preds   = self._patchify(predicted_pixels)   # [B, N_patches, patch_dim]
        targets = self._patchify(target_pixels)      # [B, N_patches, patch_dim]

        return preds, targets

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Divide an image tensor into non-overlapping spatial patches and
        normalise each patch independently (zero-mean, unit-std).

        Args:
            x: [B, 3, H, W]  float32, pixel values in [0, 1]

        Returns:
            [B, N_patches, patch_dim]
            where N_patches = (H // PSI_PATCH_SIZE) * (W // PSI_PATCH_SIZE)
            and   patch_dim = PSI_PATCH_SIZE * PSI_PATCH_SIZE * 3
        """
        p = PSI_PATCH_SIZE
        B, C, H, W = x.shape

        if H % p != 0 or W % p != 0:
            # Crop to the nearest multiple of patch size
            H_c = (H // p) * p
            W_c = (W // p) * p
            x = x[:, :, :H_c, :W_c]

        # [B, N_h*N_w, p*p*C]
        patches = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=p, p2=p)

        # Per-patch zero-mean, unit-std  (VideoMAEv2 convention)
        patch_mean = patches.mean(dim=-1, keepdim=True)
        patch_std  = patches.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6
        patches = (patches - patch_mean) / patch_std

        return patches

    def _save_visualization(
        self,
        predicted_frame_idx: int,
        pil_context: "Image.Image",
        pil_pred: "Image.Image",
        target_pixels_b: torch.Tensor,
    ) -> None:
        """
        Save a 4-panel side-by-side image to viz_dir/<video_name>/ immediately
        after each PSI generation.

        Layout:
            [ Context (rgb0) | PSI Prediction | Ground Truth | Diff ×5 ]

        Filename:
            frame{predicted_frame_idx:04d}_ctx{nb_context_frames:02d}_L1_{score:.4f}.png

        ``predicted_frame_idx`` is the index of the predicted frame in the
        subsampled video sequence (i.e. after applying frame_step).  For the
        max_context_mode phase this equals tgt_idx (clip starts at frame 0);
        for the main sliding-window loop it equals
        (window_idx × stride) + tgt_idx.

        Args:
            predicted_frame_idx: Absolute sampled-frame index being predicted.
            pil_context:         Frame fed to PSI as rgb0 (PIL Image, RGB).
            pil_pred:            PSI-generated next frame (PIL Image, RGB).
            target_pixels_b:     Actual next frame tensor [3, H, W] float32 [0,1].
        """
        import os
        from PIL import ImageDraw

        # ── Convert ground truth tensor → PIL ────────────────────────────
        tgt_np = (
            target_pixels_b.permute(1, 2, 0).cpu().float().numpy()
        )
        tgt_np = (tgt_np * 255.0).clip(0, 255).astype(np.uint8)
        pil_target = Image.fromarray(tgt_np)

        # ── Compute per-pixel L1 for filename and diff panel ─────────────
        pred_f = np.array(pil_pred).astype(np.float32) / 255.0
        tgt_f  = tgt_np.astype(np.float32) / 255.0
        diff   = np.abs(pred_f - tgt_f)                   # [H, W, 3]
        l1_score = float(diff.mean())

        # Amplify diff ×5 and convert to a visible heatmap (grayscale mapped
        # to red channel so cold areas stay dark and hot areas go red).
        diff_amp = (diff.mean(axis=-1, keepdims=True) * 5.0).clip(0.0, 1.0)  # [H,W,1]
        diff_rgb = np.concatenate([
            diff_amp,                       # R — amount of error
            diff_amp * 0.3,                 # G — subtle tint
            np.zeros_like(diff_amp),        # B
        ], axis=-1)
        pil_diff = Image.fromarray((diff_rgb * 255).astype(np.uint8))

        # ── Build 4-panel canvas ──────────────────────────────────────────
        W, H      = pil_context.size       # PIL: (width, height)
        HEADER_H  = 22                     # px reserved for label text
        GAP       = 3                      # px gap between panels
        N_PANELS  = 4
        canvas_w  = N_PANELS * W + (N_PANELS - 1) * GAP
        canvas_h  = H + HEADER_H
        canvas    = Image.new("RGB", (canvas_w, canvas_h), color=(20, 20, 20))

        panels = [
            (pil_context, f"Context  (frame {self.nb_context_frames - 1})"),
            (pil_pred,    "PSI Prediction"),
            (pil_target,  "Ground Truth"),
            (pil_diff,    f"Diff \u00d75   L1={l1_score:.4f}"),
        ]

        draw = ImageDraw.Draw(canvas)
        for i, (panel, label) in enumerate(panels):
            x = i * (W + GAP)
            canvas.paste(panel.resize((W, H), Image.BILINEAR), (x, HEADER_H))
            # White label text in the dark header strip
            draw.text((x + 4, 4), label, fill=(230, 230, 230))

        # ── Resolve output folder: viz_dir / <video_name> / ──────────────
        # Reset the per-video frame counter whenever the video changes.
        if self.current_video_name != self._prev_video_name:
            self._current_video_frame_count = 0
            self._prev_video_name = self.current_video_name

        video_folder = os.path.join(self.viz_dir, self.current_video_name)
        os.makedirs(video_folder, exist_ok=True)

        # frame{N}: N is the raw video frame index (0-based in the original
        #   video file) that PSI was asked to predict.
        #   raw_frame = predicted_sampled_frame × frame_step
        # ctx{K}: the context length used.
        # L1: the pixel-space surprise score for this prediction.
        raw_frame_idx = predicted_frame_idx * self._viz_frame_step
        filename = (
            f"frame{raw_frame_idx:05d}"
            f"_ctx{self.nb_context_frames:02d}"
            f"_L1_{l1_score:.4f}"
            f".png"
        )
        canvas.save(os.path.join(video_folder, filename))
        self._current_video_frame_count += 1

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
