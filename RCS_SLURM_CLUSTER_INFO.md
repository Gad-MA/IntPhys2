# RCS SLURM Cluster — Node & GPU Reference

> Last updated: 2026-08-12  
> Gathered via `sinfo -N -o "%.12N %.10P %.6G %f"` and `scontrol show node <node>`

---

## Partitions

| Partition | Notes |
|-----------|-------|
| `gpu` | Default GPU partition (marked `*`) |
| `debug` | Debug partition — all nodes are also members |

---

## Node Summary

| Node | GPU Model | GPU VRAM | CPU Cores (Effective) | RAM | GRES Tag | Status for using with PSI0_5 model|
|------|-----------|----------|-----------------------|-----|----------|--------|
| gpu-02 | NVIDIA GeForce RTX 5060 Ti | 16,311 MiB (~16 GB) | 12 | 29 GB | `gpu:rtx5060ti:1` | ⚠️ Excluded (OOM on PSI-0.5) |
| gpu-03 | NVIDIA GeForce GTX 1080 Ti + RTX 3060 | 11,264 + 12,288 MiB | 24 | 29 GB | `gpu:gtx1080ti:1, gpu:rtx3060:1` | ❌ Excluded (2 GPUs, low VRAM) |
| gpu-04 | NVIDIA GeForce RTX 4060 Ti | 16,380 MiB (~16 GB) | 16 | 29 GB | `gpu:rtx4060ti:1` | ⚠️ Excluded (OOM on PSI-0.5) |
| gpu-05 | NVIDIA GeForce RTX 3090 | 24,576 MiB (~24 GB) | 30 | 60 GB | `gpu:rtx3090:1` | ✅ Eligible |
| gpu-06 | NVIDIA GeForce RTX 5090 | 32,607 MiB (~32 GB) | 56 | 76 GB | `gpu:rtx5090:1` | ✅ Eligible |
| gpu-07 | NVIDIA GeForce RTX 3090 | 24,576 MiB (~24 GB) | 32 | 122 GB | `gpu:rtx3090:1` | ✅ Eligible |

---

## Detailed Node Info

### gpu-02
- **GPU:** NVIDIA GeForce RTX 5060 Ti — `gpu:rtx5060ti:1`
- **VRAM:** 16,311 MiB
- **CPU:** 12 effective cores (6 cores/socket × 1 socket × 2 threads/core)
- **RAM:** 29,000 MB (~29 GB)
- **OS:** Linux 6.8.0-136-generic (Ubuntu)
- **Partitions:** `debug`, `gpu`
- **SLURM_MEM safe max:** ~24 GB (leave ~5 GB for OS)
- **Notes:** Excluded from PSI-0.5 jobs — 16 GB VRAM causes CUDA OOM.

### gpu-03
- **GPU:** NVIDIA GeForce GTX 1080 Ti (`gpu:gtx1080ti:1`) + RTX 3060 (`gpu:rtx3060:1`)
- **VRAM:** 11,264 MiB + 12,288 MiB
- **CPU:** 24 effective cores (12 cores/socket × 1 socket × 2 threads/core)
- **RAM:** 29,000 MB (~29 GB)
- **OS:** Linux 6.8.0-136-generic (Ubuntu)
- **Partitions:** `debug`, `gpu`
- **Notes:** Excluded — has 2 GPUs (non-standard), both low VRAM. GTX 1080 Ti is legacy (Pascal arch, no bfloat16 hardware support).

### gpu-04
- **GPU:** NVIDIA GeForce RTX 4060 Ti — `gpu:rtx4060ti:1`
- **VRAM:** 16,380 MiB (~16 GB)
- **CPU:** 16 effective cores (8 cores/socket × 1 socket × 2 threads/core)
- **RAM:** 29,000 MB (~29 GB)
- **OS:** Linux 6.8.0-136-generic (Ubuntu)
- **Partitions:** `debug`, `gpu`
- **SLURM_MEM safe max:** ~24 GB (leave ~5 GB for OS)
- **Notes:** Excluded from PSI-0.5 jobs — 16 GB VRAM causes CUDA OOM.

### gpu-05
- **GPU:** NVIDIA GeForce RTX 3090 — `gpu:rtx3090:1`
- **VRAM:** 24,576 MiB (~24 GB)
- **CPU:** 30 effective cores (16 cores/socket × 1 socket × 2 threads/core, 1 reserved for OS)
- **RAM:** 60,000 MB (~60 GB)
- **OS:** Linux 6.17.0-40-generic (Ubuntu, newer kernel)
- **Partitions:** `debug`, `gpu`
- **SLURM_MEM safe max:** ~55 GB (leave ~5 GB for OS) — **cluster bottleneck**
- **Notes:** Smallest RAM among eligible nodes. Sets the ceiling for `SLURM_MEM` cluster-wide.

### gpu-06
- **GPU:** NVIDIA GeForce RTX 5090 — `gpu:rtx5090:1`
- **VRAM:** 32,607 MiB (~32 GB)
- **CPU:** 56 effective cores (56 cores/socket × 1 socket × 1 thread/core)
- **RAM:** 76,000 MB (~76 GB)
- **OS:** Linux 6.8.0-136-generic (Ubuntu)
- **Partitions:** `debug`, `gpu`
- **SLURM_MEM safe max:** ~71 GB
- **Notes:** Highest VRAM on the cluster. Best node for large models.

### gpu-07
- **GPU:** NVIDIA GeForce RTX 3090 — `gpu:rtx3090:1`
- **VRAM:** 24,576 MiB (~24 GB)
- **CPU:** 32 effective cores (16 cores/socket × 1 socket × 2 threads/core)
- **RAM:** 122,000 MB (~122 GB)
- **OS:** Linux 6.8.0-136-generic (Ubuntu)
- **Partitions:** `debug`, `gpu`
- **SLURM_MEM safe max:** ~117 GB
- **Notes:** Largest RAM on the cluster. Same GPU as gpu-05 but 2× the system memory.

---

## PSI-0.5 Job Configuration (Current)

| Setting | Value | Rationale |
|---------|-------|-----------|
| `--exclude` | `gpu-02,gpu-03,gpu-04` | ≤16 GB VRAM → CUDA OOM with PSI-0.5 |
| `nodes` | `3` | Only 3 eligible nodes remain (gpu-05/06/07) |
| `tasks_per_node` | `1` | 1 GPU per node; PSI is memory-intensive |
| `SLURM_MEM` | `55G` | Max safe for all eligible nodes (bottleneck: gpu-05, 60 GB RAM) |
| `SLURM_CLUSTER_NAME` | `rcs` | Not set automatically by the cluster; must be exported manually |

---

## VRAM Requirements Guide

| VRAM Needed | Available Nodes |
|-------------|-----------------|
| ≤ 16 GB | gpu-02, gpu-04, gpu-05, gpu-06, gpu-07 |
| ≤ 24 GB | gpu-05, gpu-06, gpu-07 |
| ≤ 32 GB | gpu-06 only |
| > 32 GB | ❌ Not available on this cluster (single-GPU per node) |

> **Multi-GPU note:** gpu-03 has 2 GPUs (11 GB + 12 GB = 23 GB combined), but SLURM allocates them as separate GRES resources. Using both requires `gpus_per_node=2` and `device_map="auto"` in the model — and even then it is excluded due to the GTX 1080 Ti lacking bfloat16 hardware support.

---

## Useful SLURM Commands

```bash
# List all nodes with partition and GRES info
sinfo -N -o "%.12N %.10P %.6G %f"

# Show detailed info for a specific node
scontrol show node gpu-05

# Check currently running jobs
squeue -u $USER

# Show job details
scontrol show job <JOB_ID>
```
