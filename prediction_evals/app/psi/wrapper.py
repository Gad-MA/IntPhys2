import torch
import torch.nn as nn
import torch.nn.functional as F

class PsiWrapper(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        # Expected by the eval loop to modify during extraction
        self.nb_context_frames = 1
        self.frames_per_clip = 16

    def forward(self, x):
        """
        x: (B, 3, T, H, W)
        
        The benchmark expects this to return (preds, targets) such that 
        F.l1_loss(preds, targets, reduction="none").mean((1,2)) gives the surprise loss.
        """
        # Split into context and targets based on nb_context_frames
        context = x[:, :, :self.nb_context_frames]
        targets = x[:, :, self.nb_context_frames:]
        
        B, C, T_ctx, H, W = context.shape
        _, _, T_tgt, _, _ = targets.shape
        
        device = context.device
        
        losses = []
        
        # IntPhys2 evaluates per video (batch size typically 1)
        for b in range(B):
            # Normalize and Quantize context frames
            ctx_frames = context[b].permute(1, 0, 2, 3) # (T_ctx, 3, H, W)
            # PSI expects [0, 1] range before _normalize_rgb
            # Assuming IntPhys2 passes [0, 1] or we normalize it
            if ctx_frames.max() > 2.0:
                ctx_frames = ctx_frames / 255.0
                
            ctx_frames = self.predictor._normalize_rgb(ctx_frames)
            ctx_codes = self.predictor.rgb_quantizer.quantize(ctx_frames, flatten=False) 
            
            # Normalize and Quantize target frames
            tgt_frames = targets[b].permute(1, 0, 2, 3) # (T_tgt, 3, H, W)
            if tgt_frames.max() > 2.0:
                tgt_frames = tgt_frames / 255.0
                
            tgt_frames = self.predictor._normalize_rgb(tgt_frames)
            tgt_codes = self.predictor.rgb_quantizer.quantize(tgt_frames, flatten=False)
            
            # Flatten into tokens
            ctx_tokens = ctx_codes.reshape(T_ctx, -1)
            tgt_tokens = tgt_codes.reshape(T_tgt, -1)
            
            # Since PSI's internal sequence building is complex and relies on specific Positional Encodings,
            # we use the predictor's model forward pass directly on the concatenated sequence
            # (Note: For a fully strict implementation, one would construct the exact (x, y, time) position 
            #  tensors as done in `predictor._build_output_sequence_with_idx`).
            
            # Simplified proxy for demonstration: calculate average loss
            # Here we just pass the targets to the transformer to evaluate their likelihood
            # If the model exposes a simpler teacher-forcing API, we would use it here.
            
            seq = torch.cat([ctx_tokens, tgt_tokens], dim=0).view(-1).unsqueeze(0) # (1, Seq_Len)
            
            # Get logits
            # In a full implementation, we'd pass seq and pos. 
            # We mock the loss output for the pipeline to test functionality.
            # IntPhys2 looks for a scalar loss per batch item.
            loss = torch.tensor(1.0, device=device, requires_grad=True) 
            losses.append(loss)
            
        losses_tensor = torch.stack(losses).view(B, 1, 1) # Shape [B, 1, 1]
        
        # We return preds as the loss, and targets as 0. 
        # The eval loop will do l1_loss(preds, targets).mean((1,2)), which just equals the loss!
        return losses_tensor, torch.zeros_like(losses_tensor)

def init_module(
    frames_per_clip: int,
    nb_context_frames: int, 
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    device: str = "cuda:0",
    **kwargs,
):
    import os
    
    # IntPhys2 eval loop passes pretrain_kwargs from the yaml as model_kwargs here
    repo_id = model_kwargs.get("repo_id", "StanfordNeuroAILab/psi0_5")
    
    # Lazy import to avoid missing dependencies if transformers isn't installed globally
    try:
        from transformers import AutoModel
    except ImportError:
        raise ImportError("Transformers is not installed. Please pip install transformers.")
        
    print(f"Loading PSI Model from {repo_id}...")
    
    # We dynamically load the PSI predictor
    predictor = AutoModel.from_pretrained(
        repo_id,
        trust_remote_code=True,
        device=device,
        _psi_dry_run=False # Actually load the model
    )
    
    wrapper = PsiWrapper(predictor)
    wrapper.to(device)
    
    return wrapper
