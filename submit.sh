#!/bin/bash
#SBATCH --job-name=vmc_heisenberg              # 作业名，可随意命名
#SBATCH --time=24:00:00                 # 最大运行时间，格式为 HH:MM:SS
#SBATCH --nodes=1                       # 请求的节点数
#SBATCH --ntasks=96                     # MPI 任务数（我们不使用 mpi4py，这里设为1）
#SBATCH --cpus-per-task=1              # 每个任务分配的 CPU 核心数
#SBATCH --partition=v6_384             # 分区名称，根据集群设置可能是 core、gpu、debug 等
#SBATCH --output=slurm_out/%x_%j.log # 标准输出日志 (需先创建 slurm_out 目录)
#SBATCH --error=slurm_out/%x_%j.err  # 错误日志

## --------- 下面是作业脚本主体 ---------

#
# (1) 固定 “进程 × 线程” 乘积不要超节点核数
#
set -euo pipefail
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PATH=$HOME/.julia/bin:$PATH
ndefect=7
L=8
default_sz=$(((L*L-ndefect)%2))
mkdir -p logs
mkdir -p slurm_out
for seed in {1..3}; do
mkdir -p "logs/defect_seed_${seed}"
summary_file="logs/defect_seed_${seed}/sector_min_energy.txt"
printf "# sz\tmin_energy\n" > "$summary_file"

mpiexecjl -n $SLURM_NTASKS  julia demo_defect_SR.jl \
    --Lx $L --Ly $L --J1 1.0 --J2 0.5 --J3 1.25 --Ndefect $ndefect \
    --nMC 38400 --rMC 50 --wMC 100 --dMC 2 --nSR 100 --lr 0.04 \
    --mu 0.0 --chi1 1.0 --chi2 -0.5 --chi3 -0.5 \
    --etad1 0.3 --etad2 0.01 --etad3 -0.6 \
    --etas1 0.01 --etas2 0.01 --etas3 0.1 \
    --bcx 0.999 --bcy 1.001 --mz 0.2 \
    --defectansatz SOFT_FPS --target_sz $default_sz --job SR --defect_seed $seed
# Search Sz
offsets=(-6 -4 -2 2 4 6 0)
for os in "${offsets[@]}"; do
  sz=$((default_sz + os))
  sz_log=$(mktemp)
  mpiexecjl -n $SLURM_NTASKS  julia demo_defect_SR.jl \
    --Lx $L --Ly $L --J1 1.0 --J2 0.5 --J3 1.25 --Ndefect $ndefect \
    --nMC 19200 --rMC 50 --wMC 100 --dMC 2 --nSR 100 --lr 0.04 \
    --mu 0.0 --chi1 1.0 --chi2 -0.5 --chi3 -0.5 \
    --etad1 0.3 --etad2 0.01 --etad3 -0.6 \
    --etas1 0.01 --etas2 0.01 --etas3 0.1 \
    --bcx 0.999 --bcy 1.001 --mz 0.2 \
    --defectansatz SOFT_FPS --target_sz $sz --job SR --defect_seed $seed \
    --init_params_json "logs/defect_seed_${seed}/target_sz_${default_sz}/min_params.json" | tee "$sz_log"

  min_e=$(awk -F': ' '/Min energy:/ {print $2; exit}' "$sz_log")
  rm -f "$sz_log"
  printf "%s\t%s\n" "$sz" "$min_e" >> "$summary_file"
done

best_line=$(awk '($1 !~ /^#/ && NF>=2){if(min=="" || $2<min){min=$2; best=$1}} END{print best, min}' "$summary_file")
best_sz=$(echo "$best_line" | awk '{print $1}')
best_energy=$(echo "$best_line" | awk '{print $2}')
echo "Best sector: sz=${best_sz}, min_energy=${best_energy}"

mpiexecjl -n $SLURM_NTASKS  julia demo_defect_SR.jl \
    --Lx $L --Ly $L --J1 1.0 --J2 0.5 --J3 1.25 --Ndefect $ndefect \
    --nMC 96000 --rMC 50 --wMC 100 --dMC 2 --nSR 200 --lr 0.03 --lr_end 0.01 \
    --mu 0.0 --chi1 1.0 --chi2 -0.5 --chi3 -0.5 \
    --etad1 0.3 --etad2 0.01 --etad3 -0.6 \
    --etas1 0.01 --etas2 0.01 --etas3 0.1 \
    --bcx 0.999 --bcy 1.001 --mz 0.2 \
    --defectansatz SOFT_FPS --target_sz $best_sz --job SR --defect_seed $seed \
    --init_params_json "logs/defect_seed_${seed}/target_sz_${best_sz}/min_params.json"

mpiexecjl -n $SLURM_NTASKS  julia demo_defect_SR.jl \
    --Lx $L --Ly $L --J1 1.0 --J2 0.5 --J3 1.25 --Ndefect $ndefect \
    --nMC 96000 --rMC 50 --wMC 100 --dMC 20 --nSR 200 --lr 0.04 \
    --mu 0.0 --chi1 1.0 --chi2 -0.5 --chi3 -0.5 \
    --etad1 0.3 --etad2 0.01 --etad3 -0.6 \
    --etas1 0.01 --etas2 0.01 --etas3 0.1 \
    --bcx 0.999 --bcy 1.001 --mz 0.2 \
    --defectansatz SOFT_FPS --target_sz $best_sz --job measure --defect_seed $seed \
    --init_params_json "logs/defect_seed_${seed}/target_sz_${best_sz}/min_params.json"

python plot_mz.py "logs/defect_seed_${seed}/target_sz_${best_sz}" --L $L
python calculate_Sq.py "logs/defect_seed_${seed}/target_sz_${best_sz}" --L $L
done
