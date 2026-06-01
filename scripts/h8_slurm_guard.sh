#!/bin/bash

h8_print_slurm_diagnostics() {
  echo "SLURM_NTASKS=${SLURM_NTASKS:-unset}"
  echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-unset}"
  echo "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-unset}"

  if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_ID:-}" ]; then
    scontrol show job "${SLURM_JOB_ID}" | grep -E 'TRES|MinMemory|NumNodes|NumCPUs|Reason|ExitCode' || true
    local first_node=""
    first_node="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null | sed -n '1p' || true)"
    if [ -n "${first_node}" ]; then
      scontrol show node "${first_node}" | grep -E 'RealMemory|AllocMem|FreeMem|CPUTot|CPUAlloc' || true
    fi
  fi
}

h8_guard_slurm_memory_accounting() {
  if ! command -v scontrol >/dev/null 2>&1 || [ -z "${SLURM_JOB_ID:-}" ]; then
    return 0
  fi

  local first_node=""
  local node_info=""
  local real_mem_mb=""
  local min_real_mem_mb="${MIN_SLURM_REAL_MEMORY_MB:-1024}"

  first_node="$(scontrol show hostnames "${SLURM_JOB_NODELIST:-}" 2>/dev/null | sed -n '1p' || true)"
  if [ -z "${first_node}" ]; then
    return 0
  fi

  node_info="$(scontrol show node "${first_node}" 2>/dev/null || true)"
  real_mem_mb="$(printf '%s\n' "${node_info}" | sed -n 's/.*RealMemory=\([0-9][0-9]*\).*/\1/p')"

  if [ "${SLURM_MEMORY_GUARD:-true}" != "false" ] &&
     [ -n "${real_mem_mb}" ] &&
     [ "${real_mem_mb}" -le "${min_real_mem_mb}" ]; then
    echo "ERROR: Slurm reports RealMemory=${real_mem_mb} MB for node ${first_node}." >&2
    echo "This is below MIN_SLURM_REAL_MEMORY_MB=${min_real_mem_mb} and can turn whole-node requests into tiny memory allocations." >&2
    echo "Ask the cluster admin to fix Slurm node RealMemory/DefMem accounting, or rerun with SLURM_MEMORY_GUARD=false only for diagnostics." >&2
    exit 3
  fi
}

h8_guard_petsc_hypre() {
  case "${H8_PC_TYPE:-}" in
    *hypre*) ;;
    *) return 0 ;;
  esac

  local petsc_variables="${PETSC_DIR:-}/${PETSC_ARCH:-}/lib/petsc/conf/petscvariables"
  local petsc_conf="${PETSC_DIR:-}/${PETSC_ARCH:-}/include/petscconf.h"

  if [ ! -f "${petsc_variables}" ] && [ ! -f "${petsc_conf}" ]; then
    echo "ERROR: PETSc arch files not found for PETSC_DIR=${PETSC_DIR:-unset} PETSC_ARCH=${PETSC_ARCH:-unset}." >&2
    exit 2
  fi

  if ! grep -qi 'HYPRE' "${petsc_variables}" "${petsc_conf}" 2>/dev/null; then
    echo "ERROR: H8_PC_TYPE=${H8_PC_TYPE} requires PETSc built with HYPRE." >&2
    echo "Use PETSC_ARCH=arch-linux-c-opt-hypre or rebuild PETSc with --download-hypre." >&2
    exit 2
  fi
}

h8_guard_required_hypre_pc() {
  if [ "${REQUIRE_H8_HYPRE_PC:-true}" = "false" ]; then
    return 0
  fi

  case "${H8_PC_TYPE:-}" in
    *hypre*) return 0 ;;
  esac

  echo "ERROR: production H8 run requires a hypre-backed preconditioner." >&2
  echo "Current H8_PC_TYPE=${H8_PC_TYPE:-unset}; expected aux_elastic_hypre for 10M/100M runs." >&2
  echo "Set H8_PC_TYPE=aux_elastic_hypre, or set REQUIRE_H8_HYPRE_PC=false only for explicit diagnostics." >&2
  exit 2
}
