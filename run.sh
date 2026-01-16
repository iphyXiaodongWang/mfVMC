#!/bin/bash
# hardware requirements
#SBATCH --time=23-47:00:00                    # enter a maximum runtime for the job. (format: DD-HH:MM:SS, or just HH:MM:SS)
#SBATCH --partition=amd_512                   # optional, cpu is default. needed for gpu/classes. See `sinfo` for options
#SBATCH --qos=normal                        # Submit debug job for quick test. See `sacctmgr show qos` for options
#SBATCH --nodes=1                   # 使用的节点数
#SBATCH --ntasks-per-node=128        # 每个节点的 MPI 进程数 (根据你的集群核心数修改，例如 16, 24, 48)
#SBATCH --cpus-per-task=1           # 每个 MPI 进程使用的 CPU 核数
#SBATCH --output=slurm_out/%x_%j.log # 标准输出日志 (需先创建 slurm_out 目录)
#SBATCH --error=slurm_out/%x_%j.err  # 错误日志

# 让每个 MPI 进程只使用一个线程，性能通常最好
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

mkdir -p logs
mkdir -p slurm_out

LX=$1
LY=${LX}
BCX=1
BCY=1
MODEL=$2
if [ "$MODEL" == "square" ]; then
    PHI=0.2
elif [ "$MODEL" == "triangular" ]; then
    PHI=0.0
else
    # 处理未知情况，比如报错退出
    echo "Error: Unknown model $MODEL"
    exit 1
fi
# 蒙特卡洛参数
NMC=$3   # 采样步数
WMC=1000     # 热身步数
RMC=2000
SEED=1234    # 随机种子

QX=1.0
QY=1.0

echo "Starting job on $(hostname) at $(date)"
echo "Parameters: Lx=$LX, Ly=$LY, Phi=$PHI, Q=($QX, $QY)"
echo "VMC Parameters: NMC=$NMC, WMC=$WMC, RMC=$RMC, seed=${SEED}"

# === 3. 运行命令 ===
# 使用 srun 或 mpirun 运行
# 注意: $SLURM_NTASKS 会自动获取上面设置的 ntasks 数
julia --project=. -e 'using Pkg; Pkg.instantiate(); using PartonTriangular; println("Precompilation done.")'

mpiexecjl -n $SLURM_NTASKS julia demo_sxsx_${MODEL}.jl \
    --Lx $LX \
    --Ly $LY \
    --bcx $BCX \
    --bcy $BCY \
    --phi $PHI \
    --qx $QX \
    --qy $QY \
    --nMC $NMC \
    --wMC $WMC \
    --rMC $RMC \
    --seed $SEED

echo "Job finished at $(date)"

