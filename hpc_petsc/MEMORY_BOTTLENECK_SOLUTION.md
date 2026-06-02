# 上亿自由度内存瓶颈解决方案

## 1. 结论

当前 MATLAB 程序的主要瓶颈不是 CPU 数量，而是内存模型：MATLAB 端会构造全局稀疏矩阵、全局索引数组、全局密度/灵敏度数组以及若干中间历史量。上亿自由度时，这些数组即使只出现一次，也会超过单机内存；如果再叠加 `iK/jK/sK` 和 AMG 预条件器，内存会迅速进入百 GB 到 TB 级别。

本目录采用的解决路线是：

1. WSL Ubuntu 只做编译、小规模验算和内存趋势验证。
2. 超算多节点使用 PETSc/MPI 分布式矩阵、向量和 KSP 求解器。
3. 先把全局线性方程迁出 MATLAB，再把密度、滤波、灵敏度和优化更新迁入 PETSc 分布式向量。
4. 最终完整优化循环不再依赖 MATLAB 全局 sparse，也不显式保存完整 H8 全局刚度矩阵。

## 2. 当前内存瓶颈在哪里

原 MATLAB 路线中最危险的对象包括：

- `iK/jK/sK`：全局装配索引和值数组。H8 体单元每个单元有 `24 x 24 = 576` 个刚度项，元素数上来后索引本身就非常大。
- `K=sparse(...)` 或 `fsparse(...)`：单进程全局稀疏矩阵。即使矩阵稀疏，数值、列号、行指针和重分配开销仍集中在一个进程内。
- `all_EMs_K`：每个粗单元保存 576 个刚度项，大模型会形成很大的 dense 中间数组。
- `all_shape_functions`：多尺度/ANN 路线中保存形函数历史，尺寸随粗单元数和细节点数增长。
- `variable_it/rho/rho_phys/dc/dv/filter`：密度、滤波密度、目标函数灵敏度、体积分灵敏度如果全部留在 MATLAB 内存中，优化循环本身也无法上亿自由度。

因此，仅仅把 MATLAB 的 `parfor` 开大，或者把 `sparse` 换成一个更快的 MEX，不能从根本上解决上亿自由度问题。必须把数据结构变成分布式。

## 3. 已实施的第一阶段

已在 `hpc_petsc` 中实现一个 PETSc/MPI 分布式求解内核：

- 源码：`src/control_arm_petsc.c`
- 编译入口：`Makefile`
- WSL smoke test：`run_wsl_smoke.sh`
- WSL 内存阶梯测试：`run_wsl_memory_ladder.sh`
- 超算可调规模脚本：`submit_petsc_scale.sbatch`
- 100M 默认脚本：`submit_petsc_100m.sbatch`

程序当前提供 `-operator low_order`。它是一个 3D、每节点 3 自由度的 SPD 刚度基准算子，用来验证 MPI、PETSc、GAMG、Slurm 和内存增长规律。默认规模为：

```text
nx = 700
ny = 400
nz = 120
DOF = 3 * nx * ny * nz = 100,800,000
```

当前程序已经做了这些内存优化：

- 每个 MPI rank 只拥有自己的矩阵行和向量片段。
- `MatCreateAIJ` 使用逐行精确 `d_nnz/o_nnz` 预分配，避免上亿行矩阵的非零元预分配浪费。
- 默认不写解向量，避免 `100M` 长向量频繁落盘。
- 小规模诊断可用 `-write_vtk true` 写 legacy VTK，包含 `displacement`、`displacement_magnitude` 和 `density`。
- 控制臂 mask 诊断可用 `-write_mask_vtk true` 写只包含实体单元的 `*_solid.vtk`，避免 ParaView 对 point-data density 做 Threshold 时产生误导。
- 每次运行输出 `<output_prefix>_summary.txt`，记录自由度、rank 数、装配时间、求解时间、迭代次数、残差、顺应度和 PETSc 统计到的峰值内存。
- PETSc `-log_view` 输出单独日志，便于看 KSP、PC、Mat、Vec 的真实耗时和内存行为。

## 4. WSL 本机验证流程

你的 WSL Ubuntu 当前已经具备：

```text
gcc / make / cmake / mpicc / mpirun / pkg-config
PETSc 3.19.6
Open MPI 4.1.6
```

但 WSL 当前可用内存约 `15GiB`，所以只建议做小规模验证。推荐先把 Windows 用户目录下的 `.wslconfig` 调整为：

```ini
[wsl2]
memory=24GB
processors=24
swap=64GB
localhostForwarding=true
```

示例文件在 `.wslconfig.example`。复制到 `C:\Users\<your-user>\.wslconfig` 后，在 PowerShell 执行：

```powershell
wsl --shutdown
```

然后重新进入 Ubuntu，检查：

```sh
free -h
```

WSL 中运行：

```sh
cd /mnt/c/Users/administered/Desktop/PIML拔模SIMP/hpc_petsc
make wsl
sh run_wsl_smoke.sh
sh run_wsl_control_arm_mask.sh
sh run_wsl_memory_ladder.sh
```

`run_wsl_smoke.sh` 默认会生成：

```text
result/wsl_smoke_80_40_24_np4.vtk
```

这个 VTK 是小规模诊断文件，可以在 ParaView 中查看 `displacement_magnitude` 或 `density`。内存阶梯和超算脚本默认不写 VTK，如需小规模导出：

控制臂 mask 诊断文件为：

```text
result/control_arm_mask_80_40_24_np4_solid.vtk
```

这个文件已经只包含 `density >= 0.5` 的实体单元，打开后不需要再加 Threshold。

```sh
WRITE_VTK=true sh run_wsl_memory_ladder.sh
```

默认 smoke test 为 `80 x 40 x 24`。内存阶梯默认为：

```text
80 x 40 x 24
120 x 60 x 32
160 x 80 x 40
```

可通过环境变量覆盖：

```sh
NP=8 CASES="80 40 24;160 80 40;220 100 50" sh run_wsl_memory_ladder.sh
```

## 5. 超算生产运行流程

先跑 debug：

```sh
cd hpc_petsc
sbatch submit_petsc_debug.sbatch
```

再跑可调规模脚本：

```sh
sbatch --nodes=4 --ntasks-per-node=28 \
  --export=ALL,NX=700,NY=400,NZ=120 \
  submit_petsc_scale.sbatch
```

也可以直接使用默认 100M 脚本：

```sh
sbatch submit_petsc_100m.sbatch
```

建议生产前按阶梯提交：

```text
1M DOF   : NX=150 NY=80  NZ=30
5M DOF   : NX=250 NY=120 NZ=56
10M DOF  : NX=320 NY=160 NZ=65
100M DOF : NX=700 NY=400 NZ=120
```

每个作业会输出：

- `result/*.out` / `result/*.err`
- `result/*_summary.txt`
- `result/*_petsc.log`

其中 `*_summary.txt` 是最先看的文件，重点关注：

- `ksp_iterations`
- `ksp_residual`
- `compliance`
- `max_peak_memory_bytes`

## 6. 当前接口与边界

已支持：

```text
-nx -ny -nz
-operator low_order
-output_prefix <path>
-write_solution true|false
-write_vtk true|false
-vtk_file <path>
-vtk_max_points <n>
-write_mask_vtk true|false
-mask_vtk_file <path>
-mask_threshold <rho>
-mask_vtk_max_cells <n>
-control_arm_mask true|false
```

已预留但未在第一阶段启用：

```text
-operator h8_matrix_free
-density_file <path>
```

如果现在使用这两个选项，程序会明确报 `PETSC_ERR_SUP`。这是有意为之：真实 H8 matrix-free 和分布式密度文件必须同时处理单元邻域、边界条件、滤波半径和灵敏度回传，不能用当前低阶内核偷偷替代，否则会给出看似能跑但物理含义错误的结果。

## 7. 后续完整移植路线

### 阶段 2：真实 H8 matrix-free

目标是实现 `-operator h8_matrix_free`：

- 使用 PETSc `DMDA` 或自定义 ownership 管理三维网格。
- 不显式装配完整全局 H8 刚度矩阵。
- 在 `MatMult` 中按单元局部计算 `Ke * ue` 并累加到残差。
- 边界条件通过行/列消元或约束投影处理。
- 小规模结果与 MATLAB H8 FEM 比较，位移/顺应度相对误差目标 `<1e-5`。

### 阶段 3：分布式密度与灵敏度

目标是让这些对象变成 PETSc `Vec` 或 HDF5/PETSc binary 分布式文件：

```text
/rho
/rho_phys
/dc
/dv
/fixed_dofs
/load
```

原则：

- 不在 MATLAB 中一次性 `load` 全量密度。
- 按 z-slab 或 PETSc ownership 分块读写。
- 滤波和投影只访问局部 ghost 区域。
- 体积分数用 MPI reduction 汇总。

### 阶段 4：优化循环迁移

目标是让 OC/MMA 更新在 PETSc 驱动程序中运行：

- MATLAB 只负责生成初始几何、载荷、固定域和可视化。
- PETSc 程序负责密度更新、投影、滤波、求解、灵敏度和日志。
- 每若干步输出一个轻量 checkpoint，供 MATLAB/ParaView 查看。

## 8. 如何判断内存瓶颈已经被解决

满足以下条件才算真正解决：

1. 100M DOF 作业中没有 MATLAB 全局 sparse。
2. PETSc 日志显示矩阵、向量、KSP 和 PC 分布在所有 MPI ranks 上。
3. 单 rank 峰值内存随 rank 数增加而下降。
4. 同一网格用不同 MPI rank 数运行，顺应度差异小于 `1e-6`。
5. 分布式密度/滤波后，体积分数误差小于 `1e-4`。
6. `rho/dc/dv` 不再由 MATLAB 一次性持有。

第一阶段已经完成第 1、2、3 项在低阶内核上的验证基础；第 4、5、6 项需要在真实 H8 与分布式优化变量阶段继续完成。
