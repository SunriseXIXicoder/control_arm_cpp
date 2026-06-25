#!/bin/bash
# H8 小网格 split draw 验证算例：只跑 +z,-z 拔模方向。
# 默认 32 个 MPI rank；算力充足时可用 sbatch -N 2 --ntasks-per-node=32 submit_h8_draft_split_z.sbatch。
# 输出默认写入 result/h8_draft_small_split_z_<NX>_<NY>_<NZ>_<JOB_ID>/。
# 如需保持长:宽:高约 2:1:1，建议设置 NX=2*(NZ-1)+1 且 NY=NZ。
#SBATCH -J h8_split_z
#SBATCH -N 1
#SBATCH --ntasks-per-node=32
#SBATCH --partition=normal,normal1,normal2,normal3,normal4
#SBATCH --output=result/%j.h8_split_z.out
#SBATCH --error=result/%j.h8_split_z.err

set -euo pipefail

export CASE_ID="${CASE_ID:-3}"
export H8_TEE_CASE_LOG="${H8_TEE_CASE_LOG:-true}"

exec bash submit_h8_draft_small_5cases.sbatch
