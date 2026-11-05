# K2-Horizon-375B-A23B — inference-performance dataset

Layout, components, record schema, launch metadata, and reproduction are
shared across the K2-Horizon family — see the [top-level
README](../../README.md). This file documents what is specific to this
checkpoint.

## Cells in this dataset

**147 points / 441 leaves / 116.0 MB.**

| cell (placement/runtime/dtype) | points | leaves |
|---|---:|---:|
| `1x4/sglang/fp8` | 23 | 69 |
| `1x4/vllm/fp8` | 24 | 72 |
| `1x8/sglang/bf16` | 23 | 69 |
| `1x8/sglang/fp8` | 23 | 69 |
| `1x8/vllm/bf16` | 26 | 78 |
| `1x8/vllm/fp8` | 26 | 78 |
| `2x8/sglang/bf16` | 2 | 6 |

`2x8` is the multi-node placement: two hosts of eight GPUs, sixteen
devices in total. A node exposes `gpu:8`, so this checkpoint cannot be
served from one host at that width. Its cell carries `nodes` alongside
`gpus`, and its `server_launch` entry gives one command per host — see
[Placement](../../README.md#placement).