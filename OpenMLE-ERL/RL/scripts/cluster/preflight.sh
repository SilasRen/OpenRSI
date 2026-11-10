#!/usr/bin/env bash
# Read-only login-node inventory. No model loading, installation or GPU allocation.
set -u
printf '\nHost and OS\n'
hostname
uname -srm
printf '\nOnline storage\n'
if [ -d "$HOME/online1" ]; then
  ls -ld "$HOME/online1"
  df -h "$HOME/online1"
else
  ls -ld /online1 2>/dev/null || true
fi
printf '\nAvailable tools\n'
for tool in python python3 conda module sbatch salloc srun sinfo apptainer singularity docker podman; do
  command -v "$tool" 2>/dev/null || true
done
printf '\nPython version\n'
python3 --version 2>/dev/null || true
printf '\nSlurm partitions and GPU resources\n'
if command -v sinfo >/dev/null 2>&1; then
  sinfo -o '%P %a %l %D %G' 2>/dev/null || true
fi
printf '\nYour existing jobs\n'
if command -v squeue >/dev/null 2>&1; then
  squeue -u "$(id -un)" -o '%.18i %.12P %.20j %.8T %.10M %.6D %R' 2>/dev/null || true
fi
printf '\nLoaded modules\n'
if type module >/dev/null 2>&1; then
  module list 2>&1 || true
fi
printf '\nContainer versions (no workloads launched)\n'
for tool in apptainer singularity docker podman; do
  if command -v "$tool" >/dev/null 2>&1; then "$tool" --version 2>/dev/null || true; fi
done
printf '\nInventory finished. No compute jobs submitted.\n'
