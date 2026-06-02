#!/bin/sh
set -eu

if [ -z "${PETSC_DIR:-}" ]; then
  echo "PETSC_DIR is not set. Example: export PETSC_DIR=/data/app/petsc" >&2
  exit 2
fi

mkdir -p bin result
make all
