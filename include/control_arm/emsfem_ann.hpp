#pragma once

#include "control_arm/ann_model.hpp"
#include "control_arm/geometry.hpp"
#include "control_arm/low_order_optimizer.hpp"

#include <petscksp.h>

namespace control_arm {

// EMsFEM ANN 模块的运行选项。
// 这些字段通常由 main.cpp 里的 PETSc options 读取后覆盖默认值，
// 用来控制神经网络目录、单元细分数、单元刚度缓存、控制臂边界条件和载荷工况。
struct EmSfemAnnOptions {
  // ANN 模型文件所在目录，里面应包含当前 sub_n 对应的权重和归一化数据。
  char ann_dir[PETSC_MAX_PATH_LEN] = "../input_20";
  // 每个粗单元在 x/y/z 方向细分的细单元数量，例如 5 表示 5x5x5。
  PetscInt sub_n = 20;
  // 是否缓存每个粗单元的 EMsFEM 等效刚度矩阵；优化模式需要开启。
  PetscBool cache_element_matrices = PETSC_TRUE;
  // 本 rank K-cache 允许使用的内存上限，0 表示不限制。
  PetscReal cache_gib_limit = 0.0;
  // 是否使用控制臂专用固定端和加载区域；false 时使用简单测试边界。
  PetscBool control_arm_bc = PETSC_FALSE;
  // 控制臂载荷工况编号；0 为旧 1-3 加权组合，4 为对称竖直下压单工况。
  PetscInt load_case = 2;
  // 是否同时包含弹簧/衬套区域载荷。
  PetscBool include_spring_load = PETSC_TRUE;
  // 无控制臂边界条件时使用的矩形域 benchmark：cantilever 或 torsion。
  char benchmark_case[32] = "cantilever";
  // fine 保留旧细单元滤波；coarse 用粗单元中心和粗单元半径滤波。
  char filter_mode[32] = "fine";
  // legacy_z 保留旧 ANN z-prefix/suffix 闭包；axis 使用共享 H8 有符号轴向闭包。
  char draft_closure_mode[32] = "legacy_z";
};

// 创建一次 EMsFEM ANN 线性求解所需的 matrix-free 矩阵、位移向量和载荷向量。
// 主要用于 solve/postsolve 路径；拓扑优化主循环会在内部创建更完整的 KSP 和预条件器。
PetscErrorCode create_emsfem_ann_system(const Grid &grid,
                                        const DensityOptions &density_options,
                                        const EmSfemAnnOptions &ems_options,
                                        PetscReal load_scale,
                                        Mat *A,
                                        Vec *u,
                                        Vec *b);

// 将 EMsFEM ANN 的位移/密度结果写出为 VTK，方便用 ParaView 等软件查看。
// Mat A 内部携带 EMsFEM 上下文，因此这里可以从 A 中取回网格和密度信息。
PetscErrorCode write_emsfem_ann_solution_vtk(Mat A,
                                             Vec displacement,
                                             const char *path,
                                             const Grid &grid,
                                             const DensityOptions &options);

// 运行 EMsFEM ANN 拓扑优化主流程。
// 该入口会初始化细密度、构造预条件器、循环求解多工况线性系统、更新设计变量并写 checkpoint/VTK。
PetscErrorCode run_emsfem_ann_optimizer(const Grid &grid,
                                        const DensityOptions &density_options,
                                        const OptimizerOptions &optimizer_options,
                                        const EmSfemAnnOptions &ems_options,
                                        const char *output_prefix,
                                        const char *final_vtk_file);

// 运行 EMsFEM ANN 后处理/补求解流程。
// 它读取已有 density/mask checkpoint，重建线性系统并求解位移，适合补写位移场或 VTK。
PetscErrorCode run_emsfem_ann_postsolve(const Grid &grid,
                                        const DensityOptions &density_options,
                                        const OptimizerOptions &optimizer_options,
                                        const EmSfemAnnOptions &ems_options,
                                        const char *density_file,
                                        const char *mask_file,
                                        const char *output_prefix);

} // namespace control_arm
