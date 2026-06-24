#include "control_arm/emsfem_ann.hpp"

#include "control_arm/petsc_utils.hpp"
#include "control_arm/vtk.hpp"

#include <petscdmda.h>
#include <petscmath.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace control_arm {
namespace {

constexpr PetscInt kElementDofs = 24;

// 记录当前 MPI rank 已经缓存了哪些粗单元的刚度矩阵。
// xs/ys/zs 是缓存盒子的起点，xm/ym/zm 是三个方向上的单元数量。
struct ElementCacheBox {
  PetscInt xs = 0, ys = 0, zs = 0;
  PetscInt xm = 0, ym = 0, zm = 0;
};

// EMsFEM ANN 算法的核心上下文。
// 它把网格、材料参数、神经网络模型、单元刚度缓存和边界条件信息打包在一起，
// 之后 MatShell 的矩阵乘法、灵敏度计算、辅助预条件器都会通过这个对象取数据。
struct EmSfemAnnContext {
  DM da = nullptr;
  Vec local_x = nullptr;
  Grid grid;
  DensityOptions density_options;
  EmSfemAnnOptions ems_options;
  AnnShapeModel ann;
  PetscReal fine_ke[kElementDofs * kElementDofs]{};
  PetscReal solid_ke[kElementDofs * kElementDofs]{};
  ElementCacheBox cache_box;
  std::vector<PetscReal> k_cache;
};

// 自定义 block-Jacobi 预条件器的上下文。
// 每个粗节点保存一个 3x3 对角块逆矩阵，用来近似求解局部位移自由度。
struct EmsBlockJacobiContext {
  EmSfemAnnContext *ems = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0;
  PetscInt xm = 0, ym = 0, zm = 0;
  std::vector<PetscReal> inverse_blocks;
};

// 优化流程的分阶段计时统计。
// 这些字段会累加 ANN 读取、K-cache 重建、KSP 求解、灵敏度和 VTK/结果写出等耗时。
struct TimingTotals {
  PetscReal setup_s = 0.0;
  PetscReal ann_load_s = 0.0;
  PetscReal density_filter_s = 0.0;
  PetscReal draft_closure_s = 0.0;
  PetscReal ann_cache_s = 0.0;
  PetscReal ann_shape_predict_s = 0.0;
  PetscReal ems_matrix_assembly_s = 0.0;
  PetscReal fine_material_sampling_s = 0.0;
  PetscReal initial_ann_cache_s = 0.0;
  PetscReal preconditioner_setup_s = 0.0;
  PetscReal load_assembly_s = 0.0;
  PetscReal ksp_solve_s = 0.0;
  PetscReal sensitivity_s = 0.0;
  PetscReal sensitivity_filter_s = 0.0;
  PetscReal optimizer_update_s = 0.0;
  PetscReal checkpoint_write_s = 0.0;
  PetscReal iteration_total_s = 0.0;
  PetscReal final_eval_s = 0.0;
  PetscReal vtk_write_s = 0.0;
  PetscReal total_s = 0.0;
};

// 单次 EMsFEM 单元刚度缓存重建的细分计时。
// 用来区分材料采样、ANN 形函数预测、单元矩阵积分装配和总耗时。
struct EmsCacheTiming {
  PetscReal total_s = 0.0;
  PetscReal material_sampling_s = 0.0;
  PetscReal ann_shape_predict_s = 0.0;
  PetscReal matrix_assembly_s = 0.0;
};

// 计算从 start 到当前时刻的并行最大耗时。
// 每个 rank 先计算自己的本地耗时，然后用 MPI_MAX 取所有 rank 中最慢的那个。
// 并行程序的实际等待时间由最慢 rank 决定，所以日志里一般记录这个最大值。
PetscErrorCode elapsed_max_since(PetscLogDouble start, PetscReal *elapsed) {
  PetscLogDouble now = 0.0;
  PetscLogDouble local_elapsed = 0.0;
  PetscLogDouble global_elapsed = 0.0;
  PetscCall(PetscTime(&now));
  local_elapsed = now - start;
  PetscCallMPI(MPI_Allreduce(&local_elapsed, &global_elapsed, 1, MPI_DOUBLE,
                             MPI_MAX, PETSC_COMM_WORLD));
  *elapsed = static_cast<PetscReal>(global_elapsed);
  return 0;
}

PetscInt h8_local_node(PetscInt lx, PetscInt ly, PetscInt lz) {
  if (lz == 0) {
    if (ly == 0) return lx == 0 ? 0 : 1;
    return lx == 0 ? 3 : 2;
  }
  if (ly == 0) return lx == 0 ? 4 : 5;
  return lx == 0 ? 7 : 6;
}

// 给定粗单元编号 ex/ey/ez，列出 H8 单元 8 个角点的粗节点坐标。
// nodes[a][0..2] 分别保存第 a 个角点的 i/j/k 索引。
// 后续装配单元矩阵、提取局部位移、施加单元贡献时都依赖这个节点顺序。
void h8_node_coords(PetscInt ex, PetscInt ey, PetscInt ez,
                    PetscInt node, PetscInt *i, PetscInt *j, PetscInt *k) {
  static const PetscInt dx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
  static const PetscInt dy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
  static const PetscInt dz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
  *i = ex + dx[node];
  *j = ey + dy[node];
  *k = ez + dz[node];
}

PetscInt fine_node_id(PetscInt ix, PetscInt iy, PetscInt iz, PetscInt sub_n) {
  const PetscInt nn = sub_n + 1;
  return ix + nn * (iy + nn * iz);
}

PetscInt fine_cell_id(PetscInt fx, PetscInt fy, PetscInt fz, PetscInt sub_n) {
  return fx + sub_n * (fy + sub_n * fz);
}

PetscInt fine_nelx(const Grid &grid, PetscInt sub_n) {
  return (grid.nx - 1) * sub_n;
}

PetscInt fine_nely(const Grid &grid, PetscInt sub_n) {
  return (grid.ny - 1) * sub_n;
}

PetscInt fine_nelz(const Grid &grid, PetscInt sub_n) {
  return (grid.nz - 1) * sub_n;
}

PetscInt fine_cell_count(const Grid &grid, PetscInt sub_n) {
  return fine_nelx(grid, sub_n) * fine_nely(grid, sub_n) *
         fine_nelz(grid, sub_n);
}

// 检查 EMsFEM 细网格尺寸是否有效，并避免 PetscInt 溢出。
// 粗节点数是 grid.nx/ny/nz，粗单元数是节点数减一，细密度单元数再乘 sub_n。
// 如果尺寸非法或超过当前 PETSc 整数范围，就提前报错，避免后面创建 Vec/DMDA 时崩溃。
PetscErrorCode check_ems_fine_grid_size(const Grid &grid, PetscInt sub_n) {
  const long double fnx = static_cast<long double>(grid.nx - 1) *
                          static_cast<long double>(sub_n);
  const long double fny = static_cast<long double>(grid.ny - 1) *
                          static_cast<long double>(sub_n);
  const long double fnz = static_cast<long double>(grid.nz - 1) *
                          static_cast<long double>(sub_n);
  const long double cells = fnx * fny * fnz;
  PetscCheck(fnx <= static_cast<long double>(PETSC_MAX_INT) &&
                 fny <= static_cast<long double>(PETSC_MAX_INT) &&
                 fnz <= static_cast<long double>(PETSC_MAX_INT) &&
                 cells <= static_cast<long double>(PETSC_MAX_INT),
             PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "EMsFEM fine-density grid exceeds PetscInt capacity; rebuild PETSc with --with-64-bit-indices=1 or reduce nx/ny/nz/sub_n");
  return 0;
}

// 返回细密度单元 (i,j,k) 的初始几何密度/设计域掩码密度。
// 它把细网格坐标映射回控制臂几何函数，用于初始化 mask 和 rho。
// 设计域内通常返回实体密度，设计域外返回 void_density。
PetscReal fine_cell_density(PetscInt i, PetscInt j, PetscInt k,
                            const Grid &grid,
                            const DensityOptions &options,
                            PetscInt sub_n) {
  const PetscReal x = (static_cast<PetscReal>(i) + 0.5) /
                      static_cast<PetscReal>(fine_nelx(grid, sub_n));
  const PetscReal y = (static_cast<PetscReal>(j) + 0.5) /
                      static_cast<PetscReal>(fine_nely(grid, sub_n));
  const PetscReal z = (static_cast<PetscReal>(k) + 0.5) /
                      static_cast<PetscReal>(fine_nelz(grid, sub_n));
  return density_at_normalized(x, y, z, grid, options);
}

// 判断细密度单元是否属于必须保持实体的硬约束区域。
// 这些区域通常对应固定端、加载端或连接孔附近，不允许优化器把它们挖空。
// 返回 PETSC_TRUE 时，后续 mask 会标成硬实体，并把密度固定为 1。
PetscBool hard_solid_fine_cell(PetscInt i, PetscInt j, PetscInt k,
                               const Grid &grid,
                               const DensityOptions &options,
                               PetscInt sub_n) {
  if (!options.use_control_arm_mask) return PETSC_FALSE;

  const PetscReal DL = domain_length(grid);
  const PetscReal DW = domain_width(grid);
  const PetscReal DH = domain_height(grid);
  const PetscReal fnx = static_cast<PetscReal>(fine_nelx(grid, sub_n));
  const PetscReal fny = static_cast<PetscReal>(fine_nely(grid, sub_n));
  const PetscReal fnz = static_cast<PetscReal>(fine_nelz(grid, sub_n));
  const PetscReal X = DL * (static_cast<PetscReal>(i) + 0.5) / fnx;
  const PetscReal Y = DW * (static_cast<PetscReal>(j) + 0.5) / fny;
  const PetscReal Z = DH * (static_cast<PetscReal>(k) + 0.5) / fnz;
  const PetscReal min_ld = PetscMin(DL, DH);
  const PetscReal min_lw = PetscMin(DL, DW);
  const PetscReal hmax = PetscMax(PetscMax(DL / fnx, DW / fny), DH / fnz);

  const PetscReal A[3] = {0.18 * DL, DW, 0.50 * DH};
  const PetscReal B[3] = {0.18 * DL, 0.0, 0.50 * DH};
  const PetscReal C[3] = {0.78 * DL, 0.50 * DW, 0.50 * DH};
  const PetscReal spring_center[3] = {0.50 * DL, 0.50 * DW, 0.50 * DH};

  const PetscReal rA_hole = 0.25 * min_ld;
  const PetscReal rB_hole = 0.25 * min_ld;
  const PetscReal rC_hole = 0.10 * min_lw;
  const PetscReal rA_keep = 0.30 * min_ld;
  const PetscReal rB_keep = 0.30 * min_ld;
  const PetscReal rC_keep = 0.13 * min_lw;
  const PetscReal rA_pad = 0.35 * min_ld;
  const PetscReal rB_pad = 0.35 * min_ld;

  const PetscReal rrA =
      PetscSqrtReal((X - A[0]) * (X - A[0]) + (Z - A[2]) * (Z - A[2]));
  const PetscReal rrB =
      PetscSqrtReal((X - B[0]) * (X - B[0]) + (Z - B[2]) * (Z - B[2]));
  const PetscReal rrC =
      PetscSqrtReal((X - C[0]) * (X - C[0]) + (Y - C[1]) * (Y - C[1]));
  const PetscBool ringA =
      (A[1] - Y <= rA_pad + hmax && rrA > rA_hole && rrA <= rA_keep)
          ? PETSC_TRUE
          : PETSC_FALSE;
  const PetscBool ringB =
      (Y - B[1] <= rB_pad + hmax && rrB > rB_hole && rrB <= rB_keep)
          ? PETSC_TRUE
          : PETSC_FALSE;
  const PetscBool ringC =
      (PetscAbsReal(Z - C[2]) <= PetscMax(0.25 * DH, 2.0 * hmax) &&
       rrC > rC_hole && rrC <= rC_keep)
          ? PETSC_TRUE
          : PETSC_FALSE;
  const PetscBool spring_mount =
      (Z >= 0.25 * DH && Z <= 0.75 * DH &&
       PetscAbsReal(X - spring_center[0]) <= 0.105 * DL / 2.0 &&
       PetscAbsReal(Y - spring_center[1]) <= 0.105 * DW / 2.0)
          ? PETSC_TRUE
          : PETSC_FALSE;

  return (ringA || ringB || ringC || spring_mount) ? PETSC_TRUE : PETSC_FALSE;
}

// 计算规则 H8 八节点六面体单元的 24x24 实体弹性刚度矩阵。
// hx/hy/hz 是单元物理尺寸，young/nu 是材料弹性参数。
// 这个矩阵既可作为普通 H8 单元刚度，也作为 EMsFEM 细胞积分的基础刚度。
void compute_h8_element_stiffness(PetscReal hx, PetscReal hy, PetscReal hz,
                                  PetscReal young_modulus, PetscReal *ke) {
  const PetscReal nu = 0.30;
  const PetscReal E = young_modulus;
  const PetscReal lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
  const PetscReal mu = E / (2.0 * (1.0 + nu));
  PetscReal D[6][6] = {};
  const PetscReal detJ = hx * hy * hz / 8.0;
  const PetscReal g = 1.0 / PetscSqrtReal(3.0);
  const PetscReal gps[2] = {-g, g};
  static const PetscReal xi_node[8] = {-1, 1, 1, -1, -1, 1, 1, -1};
  static const PetscReal eta_node[8] = {-1, -1, 1, 1, -1, -1, 1, 1};
  static const PetscReal zeta_node[8] = {-1, -1, -1, -1, 1, 1, 1, 1};

  for (PetscInt i = 0; i < kElementDofs * kElementDofs; ++i) ke[i] = 0.0;
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) D[i][j] = lambda;
    D[i][i] = lambda + 2.0 * mu;
  }
  D[3][3] = mu;
  D[4][4] = mu;
  D[5][5] = mu;

  for (PetscInt gz = 0; gz < 2; ++gz) {
    for (PetscInt gy = 0; gy < 2; ++gy) {
      for (PetscInt gx = 0; gx < 2; ++gx) {
        const PetscReal xi = gps[gx];
        const PetscReal eta = gps[gy];
        const PetscReal zeta = gps[gz];
        PetscReal B[6][kElementDofs] = {};
        for (PetscInt a = 0; a < 8; ++a) {
          const PetscReal dN_dxi =
              0.125 * xi_node[a] * (1.0 + eta * eta_node[a]) *
              (1.0 + zeta * zeta_node[a]);
          const PetscReal dN_deta =
              0.125 * eta_node[a] * (1.0 + xi * xi_node[a]) *
              (1.0 + zeta * zeta_node[a]);
          const PetscReal dN_dzeta =
              0.125 * zeta_node[a] * (1.0 + xi * xi_node[a]) *
              (1.0 + eta * eta_node[a]);
          const PetscReal dN_dx = dN_dxi * 2.0 / hx;
          const PetscReal dN_dy = dN_deta * 2.0 / hy;
          const PetscReal dN_dz = dN_dzeta * 2.0 / hz;
          const PetscInt col = 3 * a;
          B[0][col + 0] = dN_dx;
          B[1][col + 1] = dN_dy;
          B[2][col + 2] = dN_dz;
          B[3][col + 0] = dN_dy;
          B[3][col + 1] = dN_dx;
          B[4][col + 1] = dN_dz;
          B[4][col + 2] = dN_dy;
          B[5][col + 0] = dN_dz;
          B[5][col + 2] = dN_dx;
        }
        for (PetscInt a = 0; a < kElementDofs; ++a) {
          for (PetscInt b = 0; b < kElementDofs; ++b) {
            PetscReal value = 0.0;
            for (PetscInt r = 0; r < 6; ++r) {
              for (PetscInt s = 0; s < 6; ++s) value += B[r][a] * D[r][s] * B[s][b];
            }
            ke[kElementDofs * a + b] += value * detJ;
          }
        }
      }
    }
  }
}

// 构造粗 H8 单元内部细节点上的三线性形函数。
// shape 存的是“细节点自由度到粗单元 24 个自由度”的映射矩阵。
// 当 ANN 没有覆盖内部节点，或需要边界一致性时，这个三线性形函数是基准形函数。
void fill_trilinear_shape(PetscInt sub_n, std::vector<PetscReal> *shape) {
  const PetscInt nn = sub_n + 1;
  shape->assign(static_cast<std::size_t>(3 * nn * nn * nn) * kElementDofs, 0.0);
  for (PetscInt iz = 0; iz <= sub_n; ++iz) {
    const PetscReal z = static_cast<PetscReal>(iz) / static_cast<PetscReal>(sub_n);
    for (PetscInt iy = 0; iy <= sub_n; ++iy) {
      const PetscReal y = static_cast<PetscReal>(iy) / static_cast<PetscReal>(sub_n);
      for (PetscInt ix = 0; ix <= sub_n; ++ix) {
        const PetscReal x = static_cast<PetscReal>(ix) / static_cast<PetscReal>(sub_n);
        const PetscReal n[8] = {
            (1.0 - x) * (1.0 - y) * (1.0 - z),
            x * (1.0 - y) * (1.0 - z),
            x * y * (1.0 - z),
            (1.0 - x) * y * (1.0 - z),
            (1.0 - x) * (1.0 - y) * z,
            x * (1.0 - y) * z,
            x * y * z,
            (1.0 - x) * y * z};
        const PetscInt node = fine_node_id(ix, iy, iz, sub_n);
        for (PetscInt c = 0; c < 3; ++c) {
          const PetscInt row = 3 * node + c;
          for (PetscInt a = 0; a < 8; ++a) {
            (*shape)[static_cast<std::size_t>(row) * kElementDofs +
                     static_cast<std::size_t>(3 * a + c)] = n[a];
          }
        }
      }
    }
  }
}

// 用 ANN 预测结果覆盖粗单元内部细节点的形函数。
// 边界细节点仍保留三线性形函数，以保证相邻粗单元之间的位移连续性。
// inside_shape 只包含内部节点，函数负责把它放回完整 shape 数组的正确位置。
void overwrite_inside_shape(PetscInt sub_n,
                            const std::vector<PetscReal> &inside_shape,
                            std::vector<PetscReal> *shape) {
  PetscInt row_in = 0;
  for (PetscInt iz = 1; iz < sub_n; ++iz) {
    for (PetscInt iy = 1; iy < sub_n; ++iy) {
      for (PetscInt ix = 1; ix < sub_n; ++ix) {
        const PetscInt node = fine_node_id(ix, iy, iz, sub_n);
        for (PetscInt c = 0; c < 3; ++c) {
          const PetscInt row = 3 * node + c;
          std::copy(inside_shape.begin() +
                        static_cast<std::ptrdiff_t>(row_in * kElementDofs),
                    inside_shape.begin() +
                        static_cast<std::ptrdiff_t>((row_in + 1) * kElementDofs),
                    shape->begin() + static_cast<std::ptrdiff_t>(row * kElementDofs));
          ++row_in;
        }
      }
    }
  }
}

// 根据一个粗单元内部的细材料分布构造多尺度形函数。
// 流程是先生成三线性基准形函数，再调用 ANN 预测内部形函数并覆盖内部节点。
// timing 不为空时会累计 ANN 预测耗时，便于分析 K-cache 重建瓶颈。
PetscErrorCode build_shape_from_material(EmSfemAnnContext *ctx,
                                         const std::vector<PetscReal> &material,
                                         std::vector<PetscReal> *shape,
                                         EmsCacheTiming *timing = nullptr) {
  PetscReal min_value = PETSC_MAX_REAL;
  PetscReal max_value = -PETSC_MAX_REAL;
  for (const PetscReal value : material) {
    min_value = PetscMin(min_value, value);
    max_value = PetscMax(max_value, value);
  }

  fill_trilinear_shape(ctx->ems_options.sub_n, shape);
  if (max_value >= 1.0 - 1.0e-12 && min_value >= 1.0 - 1.0e-12) {
    return 0;
  }

  std::vector<PetscReal> inside_shape;
  PetscLogDouble predict_start = 0.0, predict_end = 0.0;
  if (timing != nullptr) PetscCall(PetscTime(&predict_start));
  PetscCall(ctx->ann.predict_inside_shape(material, &inside_shape));
  if (timing != nullptr) {
    PetscCall(PetscTime(&predict_end));
    timing->ann_shape_predict_s +=
        static_cast<PetscReal>(predict_end - predict_start);
  }
  overwrite_inside_shape(ctx->ems_options.sub_n, inside_shape, shape);
  return 0;
}

// 读取粗单元 (ex,ey,ez) 内第 (fx,fy,fz) 个细胞的 SIMP 后材料刚度比例。
// 当前版本从 ctx->fine_density 按全局细单元编号取密度，再通过 simp_scale 转成材料比例。
// 这个函数主要用于非优化或旧路径，优化路径通常从 rho_phys 的 local vector 直接采样。
PetscReal material_at_fine_cell(PetscInt ex, PetscInt ey, PetscInt ez,
                                PetscInt fx, PetscInt fy, PetscInt fz,
                                const EmSfemAnnContext &ctx) {
  const PetscReal sub = static_cast<PetscReal>(ctx.ems_options.sub_n);
  const PetscReal x = (static_cast<PetscReal>(ex) +
                       (static_cast<PetscReal>(fx) + 0.5) / sub) /
                      static_cast<PetscReal>(ctx.grid.nx - 1);
  const PetscReal y = (static_cast<PetscReal>(ey) +
                       (static_cast<PetscReal>(fy) + 0.5) / sub) /
                      static_cast<PetscReal>(ctx.grid.ny - 1);
  const PetscReal z = (static_cast<PetscReal>(ez) +
                       (static_cast<PetscReal>(fz) + 0.5) / sub) /
                      static_cast<PetscReal>(ctx.grid.nz - 1);
  const PetscReal rho = density_at_normalized(x, y, z, ctx.grid, ctx.density_options);
  return ctx.density_options.emin +
         (1.0 - ctx.density_options.emin) *
             PetscPowReal(PetscMax(0.0, PetscMin(1.0, rho)),
                          ctx.density_options.penal);
}

// 用多尺度形函数把细胞刚度积分/凝聚成粗单元 24x24 刚度矩阵。
// material[fine_id] 是每个细胞的材料比例，shape 是细节点到粗自由度的形函数映射。
// 本质上是在做 ke += B_shape^T * fine_ke * B_shape 的离散累加。
void accumulate_shape_stiffness(const PetscReal *fine_ke,
                                const std::vector<PetscReal> &shape,
                                const std::vector<PetscReal> &material,
                                PetscInt sub_n,
                                PetscReal *ke) {
  for (PetscInt i = 0; i < kElementDofs * kElementDofs; ++i) ke[i] = 0.0;

  PetscInt fine_id = 0;
  for (PetscInt fz = 0; fz < sub_n; ++fz) {
    for (PetscInt fy = 0; fy < sub_n; ++fy) {
      for (PetscInt fx = 0; fx < sub_n; ++fx, ++fine_id) {
        PetscReal B[kElementDofs][kElementDofs] = {};
        PetscReal tmp[kElementDofs][kElementDofs] = {};
        for (PetscInt local_node = 0; local_node < 8; ++local_node) {
          PetscInt ix = 0, iy = 0, iz = 0;
          h8_node_coords(fx, fy, fz, local_node, &ix, &iy, &iz);
          const PetscInt fine_node = fine_node_id(ix, iy, iz, sub_n);
          for (PetscInt c = 0; c < 3; ++c) {
            const PetscInt src_row = 3 * fine_node + c;
            const PetscInt dst_row = 3 * local_node + c;
            for (PetscInt col = 0; col < kElementDofs; ++col) {
              B[dst_row][col] =
                  shape[static_cast<std::size_t>(src_row) * kElementDofs +
                        static_cast<std::size_t>(col)];
            }
          }
        }

        for (PetscInt p = 0; p < kElementDofs; ++p) {
          for (PetscInt b = 0; b < kElementDofs; ++b) {
            PetscReal value = 0.0;
            for (PetscInt q = 0; q < kElementDofs; ++q) {
              value += fine_ke[kElementDofs * p + q] * B[q][b];
            }
            tmp[p][b] = value;
          }
        }
        for (PetscInt a = 0; a < kElementDofs; ++a) {
          for (PetscInt b = 0; b < kElementDofs; ++b) {
            PetscReal value = 0.0;
            for (PetscInt p = 0; p < kElementDofs; ++p) value += B[p][a] * tmp[p][b];
            ke[kElementDofs * a + b] +=
                material[static_cast<std::size_t>(fine_id)] * value;
          }
        }
      }
    }
  }
}

// 计算一个粗单元内部每个细胞对单元应变能的贡献。
// ue 是粗单元 24 个位移自由度，energies[fine_id] 输出每个细胞的能量密度贡献。
// 灵敏度计算会用这些细胞能量乘以 SIMP 导数，得到细密度设计变量的梯度。
PetscErrorCode compute_fine_cell_energies(EmSfemAnnContext *ctx,
                                          const std::vector<PetscReal> &material,
                                          const PetscReal ue[kElementDofs],
                                          PetscReal *energies) {
  const PetscInt sub_n = ctx->ems_options.sub_n;
  std::vector<PetscReal> shape;
  PetscCall(build_shape_from_material(ctx, material, &shape));

  PetscInt fine_id = 0;
  for (PetscInt fz = 0; fz < sub_n; ++fz) {
    for (PetscInt fy = 0; fy < sub_n; ++fy) {
      for (PetscInt fx = 0; fx < sub_n; ++fx, ++fine_id) {
        PetscReal q[kElementDofs] = {};
        for (PetscInt local_node = 0; local_node < 8; ++local_node) {
          PetscInt ix = 0, iy = 0, iz = 0;
          h8_node_coords(fx, fy, fz, local_node, &ix, &iy, &iz);
          const PetscInt fine_node = fine_node_id(ix, iy, iz, sub_n);
          for (PetscInt c = 0; c < 3; ++c) {
            const PetscInt row = 3 * local_node + c;
            const PetscInt src_row = 3 * fine_node + c;
            for (PetscInt b = 0; b < kElementDofs; ++b) {
              q[row] += shape[static_cast<std::size_t>(src_row) *
                                  kElementDofs +
                              static_cast<std::size_t>(b)] *
                        ue[b];
            }
          }
        }

        PetscReal energy = 0.0;
        for (PetscInt p = 0; p < kElementDofs; ++p) {
          PetscReal tmp = 0.0;
          for (PetscInt qcol = 0; qcol < kElementDofs; ++qcol) {
            tmp += ctx->fine_ke[kElementDofs * p + qcol] * q[qcol];
          }
          energy += q[p] * tmp;
        }
        energies[fine_id] = energy;
      }
    }
  }
  return 0;
}

// 根据一个粗单元内部 sub_n^3 个细胞的材料比例计算 EMsFEM 等效刚度矩阵。
// 全实体或近全空时走快速路径；一般情况先构造 ANN 多尺度形函数，再积分得到 ke。
// 输出 ke 是粗 H8 单元层面的 24x24 刚度矩阵，后续 MatMult 和辅助矩阵都会用它。
PetscErrorCode compute_ems_element_matrix_from_material(
    EmSfemAnnContext *ctx,
    const std::vector<PetscReal> &material,
    PetscReal *ke,
    EmsCacheTiming *timing = nullptr) {
  const PetscInt sub_n = ctx->ems_options.sub_n;
  PetscReal sum = 0.0;
  PetscReal min_value = PETSC_MAX_REAL;
  PetscReal max_value = -PETSC_MAX_REAL;

  for (const PetscReal value : material) {
    sum += value;
    min_value = PetscMin(min_value, value);
    max_value = PetscMax(max_value, value);
  }

  const PetscReal mean = sum / static_cast<PetscReal>(material.size());
  if (max_value >= 1.0 - 1.0e-12 && min_value >= 1.0 - 1.0e-12) {
    std::copy(ctx->solid_ke, ctx->solid_ke + kElementDofs * kElementDofs, ke);
    return 0;
  }
  if (mean <= 1.01 * ctx->density_options.emin) {
    for (PetscInt i = 0; i < kElementDofs * kElementDofs; ++i) {
      ke[i] = ctx->solid_ke[i] * ctx->density_options.emin;
    }
    return 0;
  }

  std::vector<PetscReal> shape;
  PetscCall(build_shape_from_material(ctx, material, &shape, timing));
  PetscLogDouble assembly_start = 0.0, assembly_end = 0.0;
  if (timing != nullptr) PetscCall(PetscTime(&assembly_start));
  accumulate_shape_stiffness(ctx->fine_ke, shape, material, sub_n, ke);

  for (PetscInt a = 0; a < kElementDofs; ++a) {
    for (PetscInt b = a + 1; b < kElementDofs; ++b) {
      const PetscReal sym =
          0.5 * (ke[kElementDofs * a + b] + ke[kElementDofs * b + a]);
      ke[kElementDofs * a + b] = sym;
      ke[kElementDofs * b + a] = sym;
    }
  }
  if (timing != nullptr) {
    PetscCall(PetscTime(&assembly_end));
    timing->matrix_assembly_s +=
        static_cast<PetscReal>(assembly_end - assembly_start);
  }
  return 0;
}

// 从 ctx->fine_density 采样指定粗单元内部的细材料，并计算对应的 EMsFEM 单元刚度矩阵。
// 这是按粗单元编号直接计算 ke 的便捷封装。
// 优化主循环中更常用 rebuild_local_k_cache_from_rho，因为它从当前 rho_phys 并行向量采样。
PetscErrorCode compute_ems_element_matrix(EmSfemAnnContext *ctx,
                                          PetscInt ex,
                                          PetscInt ey,
                                          PetscInt ez,
                                          PetscReal *ke) {
  const PetscInt sub_n = ctx->ems_options.sub_n;
  std::vector<PetscReal> material;
  material.reserve(static_cast<std::size_t>(sub_n * sub_n * sub_n));

  for (PetscInt fz = 0; fz < sub_n; ++fz) {
    for (PetscInt fy = 0; fy < sub_n; ++fy) {
      for (PetscInt fx = 0; fx < sub_n; ++fx) {
        material.push_back(material_at_fine_cell(ex, ey, ez, fx, fy, fz, *ctx));
      }
    }
  }
  PetscCall(compute_ems_element_matrix_from_material(ctx, material, ke));
  return 0;
}

// 判断粗单元 (ex,ey,ez) 是否落在当前 rank 的 K-cache 盒子里。
// MatMult 只能直接使用本 rank 已缓存的单元刚度；不在盒子内就返回 false。
// 这个检查用于避免错误索引 k_cache。
PetscBool cache_contains(const ElementCacheBox &box,
                         PetscInt ex,
                         PetscInt ey,
                         PetscInt ez) {
  return (ex >= box.xs && ex < box.xs + box.xm &&
          ey >= box.ys && ey < box.ys + box.ym &&
          ez >= box.zs && ez < box.zs + box.zm)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

// 返回粗单元 (ex,ey,ez) 在本地 K-cache 中对应的 24x24 矩阵指针。
// 如果缓存关闭或单元不在 cache_box 内，返回 nullptr。
// 返回的指针直接指向 ctx->k_cache 内部连续存储区域，不需要释放。
PetscReal *cached_element_matrix(EmSfemAnnContext *ctx,
                                 PetscInt ex,
                                 PetscInt ey,
                                 PetscInt ez) {
  const ElementCacheBox &box = ctx->cache_box;
  if (!ctx->ems_options.cache_element_matrices || !cache_contains(box, ex, ey, ez)) {
    return nullptr;
  }
  const std::size_t lx = static_cast<std::size_t>(ex - box.xs);
  const std::size_t ly = static_cast<std::size_t>(ey - box.ys);
  const std::size_t lz = static_cast<std::size_t>(ez - box.zs);
  const std::size_t idx =
      (lz * static_cast<std::size_t>(box.ym) + ly) *
          static_cast<std::size_t>(box.xm) +
      lx;
  return ctx->k_cache.data() + idx * kElementDofs * kElementDofs;
}

// 根据 ctx->fine_density 为当前 rank 需要的粗单元建立本地 K-cache。
// cache_box 会覆盖本 rank 拥有节点周围的一层粗单元，保证 matrix-free MatMult 能本地取到 ke。
// 这个版本从 ctx->fine_density 采样，主要服务旧的 solve/postsolve 路径。
PetscErrorCode build_local_k_cache(EmSfemAnnContext *ctx) {
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(DMDAGetCorners(ctx->da, &xs, &ys, &zs, &xm, &ym, &zm));
  ElementCacheBox box;
  box.xs = PetscMax(0, xs - 1);
  box.ys = PetscMax(0, ys - 1);
  box.zs = PetscMax(0, zs - 1);
  const PetscInt xe = PetscMin(ctx->grid.nx - 1, xs + xm);
  const PetscInt ye = PetscMin(ctx->grid.ny - 1, ys + ym);
  const PetscInt ze = PetscMin(ctx->grid.nz - 1, zs + zm);
  box.xm = PetscMax(0, xe - box.xs);
  box.ym = PetscMax(0, ye - box.ys);
  box.zm = PetscMax(0, ze - box.zs);
  ctx->cache_box = box;

  const std::size_t elems =
      static_cast<std::size_t>(box.xm) * static_cast<std::size_t>(box.ym) *
      static_cast<std::size_t>(box.zm);
  const double gib =
      static_cast<double>(elems) * static_cast<double>(kElementDofs * kElementDofs) *
      sizeof(PetscReal) / 1073741824.0;
  PetscCheck(ctx->ems_options.cache_gib_limit <= 0.0 ||
                 gib <= ctx->ems_options.cache_gib_limit,
             PETSC_COMM_WORLD, PETSC_ERR_MEM,
             "Local EMsFEM K cache would use %.3g GiB, above -ems_cache_gib_limit %.3g. Use -ems_cache_element_matrices false or increase the limit.",
             gib, static_cast<double>(ctx->ems_options.cache_gib_limit));

  ctx->k_cache.assign(elems * kElementDofs * kElementDofs, 0.0);
  for (PetscInt ez = box.zs; ez < box.zs + box.zm; ++ez) {
    for (PetscInt ey = box.ys; ey < box.ys + box.ym; ++ey) {
      for (PetscInt ex = box.xs; ex < box.xs + box.xm; ++ex) {
        PetscReal *ke = cached_element_matrix(ctx, ex, ey, ez);
        PetscCall(compute_ems_element_matrix(ctx, ex, ey, ez, ke));
      }
    }
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "EMsFEM ANN local-K cache built: max %.3f GiB/rank before solve\n",
                        gib));
  return 0;
}

// SIMP 材料插值：把密度 rho 转成有效刚度比例。
// emin 给空材料保留极小刚度，penal 用来惩罚中间密度，使优化结果趋向 0/1。
// 返回值会乘到单元刚度上。
PetscReal simp_scale(PetscReal rho, const DensityOptions &opts) {
  rho = PetscMax(1.0e-12, PetscMin(1.0, rho));
  return opts.emin + (1.0 - opts.emin) * PetscPowReal(rho, opts.penal);
}

// SIMP 插值对密度 rho 的导数。
// 灵敏度分析需要 dK/drho，因此细胞能量会乘以这个导数得到目标函数梯度。
// rho 会被限制在合理范围内，避免 0 附近出现数值问题。
PetscReal simp_derivative(PetscReal rho, const DensityOptions &opts) {
  rho = PetscMax(1.0e-12, PetscMin(1.0, rho));
  return (1.0 - opts.emin) * opts.penal * PetscPowReal(rho, opts.penal - 1.0);
}

// 计算密度过滤器的锥形权重。
// 距离越近权重越大，超过 radius 的邻居权重为 0。
// dx/dy/dz 是细网格索引偏移，radius 是细网格单位下的过滤半径。
PetscReal cone_weight(PetscInt dx, PetscInt dy, PetscInt dz, PetscReal radius) {
  const PetscReal dist =
      PetscSqrtReal(static_cast<PetscReal>(dx * dx + dy * dy + dz * dz));
  return PetscMax(0.0, radius - dist);
}

PetscInt ceil_div(PetscInt a, PetscInt b) {
  return (a + b - 1) / b;
}

// 自动或手动选择 EMsFEM 的三维 MPI 进程分区 px/py/pz。
// 如果用户传了 -ems_dm_px/-ems_dm_py/-ems_dm_pz，就严格检查后采用；
// 否则枚举所有 px*py*pz=ranks 的分区，用局部块大小、ghost 开销和长宽比评分选最优。
PetscErrorCode choose_ems_process_grid(const Grid &grid,
                                       PetscInt sub_n,
                                       PetscInt fine_stencil_width,
                                       PetscMPIInt ranks,
                                       PetscInt *px,
                                       PetscInt *py,
                                       PetscInt *pz) {
  const PetscInt fnx = fine_nelx(grid, sub_n);
  const PetscInt fny = fine_nely(grid, sub_n);
  const PetscInt fnz = fine_nelz(grid, sub_n);
  const PetscInt stencil = PetscMax(1, fine_stencil_width);
  PetscInt opt_px = 0, opt_py = 0, opt_pz = 0;
  PetscBool set_px = PETSC_FALSE, set_py = PETSC_FALSE, set_pz = PETSC_FALSE;

  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-ems_dm_px", &opt_px, &set_px));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-ems_dm_py", &opt_py, &set_py));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr, "-ems_dm_pz", &opt_pz, &set_pz));
  if (set_px || set_py || set_pz) {
    PetscCheck(set_px && set_py && set_pz, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_INCOMP,
               "Set -ems_dm_px, -ems_dm_py, and -ems_dm_pz together");
    PetscCheck(opt_px > 0 && opt_py > 0 && opt_pz > 0, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_OUTOFRANGE,
               "EMsFEM process-grid dimensions must be positive");
    PetscCheck(opt_px * opt_py * opt_pz == static_cast<PetscInt>(ranks),
               PETSC_COMM_WORLD, PETSC_ERR_ARG_INCOMP,
               "EMsFEM process grid product must equal MPI ranks");
    PetscCheck(grid.nx >= opt_px && grid.ny >= opt_py && grid.nz >= opt_pz,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "EMsFEM process grid has more ranks than coarse nodes in at least one direction");
    PetscCheck(fnx >= opt_px * stencil && fny >= opt_py * stencil &&
                   fnz >= opt_pz * stencil,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
               "EMsFEM process grid needs each local fine-density block at least as thick as the fine filter/cache stencil in x, y, and z");
    *px = opt_px;
    *py = opt_py;
    *pz = opt_pz;
    return 0;
  }

  PetscBool found = PETSC_FALSE;
  PetscReal best_score = PETSC_MAX_REAL;
  PetscInt best_px = 1, best_py = 1, best_pz = static_cast<PetscInt>(ranks);

  for (PetscMPIInt tx = 1; tx <= ranks; ++tx) {
    if (ranks % tx != 0) continue;
    const PetscMPIInt rem = ranks / tx;
    for (PetscMPIInt ty = 1; ty <= rem; ++ty) {
      if (rem % ty != 0) continue;
      const PetscMPIInt tz = rem / ty;
      if (grid.nx < static_cast<PetscInt>(tx) ||
          grid.ny < static_cast<PetscInt>(ty) ||
          grid.nz < static_cast<PetscInt>(tz)) {
        continue;
      }
      if (fnx < static_cast<PetscInt>(tx) * stencil ||
          fny < static_cast<PetscInt>(ty) * stencil ||
          fnz < static_cast<PetscInt>(tz) * stencil) {
        continue;
      }

      const PetscReal lx =
          static_cast<PetscReal>(ceil_div(fnx, static_cast<PetscInt>(tx)));
      const PetscReal ly =
          static_cast<PetscReal>(ceil_div(fny, static_cast<PetscInt>(ty)));
      const PetscReal lz =
          static_cast<PetscReal>(ceil_div(fnz, static_cast<PetscInt>(tz)));
      const PetscReal local_cells = lx * ly * lz;
      const PetscReal halo_ratio =
          (lx + 2.0 * stencil) * (ly + 2.0 * stencil) *
          (lz + 2.0 * stencil) / PetscMax(1.0, local_cells);
      const PetscReal local_min = PetscMin(lx, PetscMin(ly, lz));
      const PetscReal local_max = PetscMax(lx, PetscMax(ly, lz));
      const PetscReal aspect = local_max / PetscMax(1.0, local_min);
      const PetscReal score =
          local_cells * (1.0 + 0.10 * (halo_ratio - 1.0)) *
          (1.0 + 0.02 * (aspect - 1.0));

      if (!found || score < best_score) {
        found = PETSC_TRUE;
        best_score = score;
        best_px = static_cast<PetscInt>(tx);
        best_py = static_cast<PetscInt>(ty);
        best_pz = static_cast<PetscInt>(tz);
      }
    }
  }

  PetscCheck(found, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
             "EMsFEM optimizer could not find a 3D process grid where each local fine-density block covers the filter/cache stencil; reduce ranks/filter radius or increase nx/ny/nz");
  *px = best_px;
  *py = best_py;
  *pz = best_pz;
  return 0;
}

// 把某一方向的“粗节点 ownership”转换成“粗单元 ownership”和“细密度 ownership”。
// node_lens 由 uda 给出；elem_lens 让粗单元和节点分区对齐；fine_lens 再乘 sub_n。
// 这样 fda 的细密度分区边界会落在粗单元边界上，避免 K-cache 采样跨 rank 错位。
void ems_ownership_from_nodal_range(const PetscInt *node_lens,
                                    PetscInt p,
                                    PetscInt n_nodes,
                                    PetscInt sub_n,
                                    std::vector<PetscInt> *elem_lens,
                                    std::vector<PetscInt> *fine_lens) {
  elem_lens->assign(static_cast<std::size_t>(p), 0);
  fine_lens->assign(static_cast<std::size_t>(p), 0);
  PetscInt node_start = 0;
  for (PetscInt r = 0; r < p; ++r) {
    const PetscInt node_end = node_start + node_lens[r];
    const PetscInt elem_start = PetscMin(node_start, n_nodes - 1);
    const PetscInt elem_end = PetscMin(node_end, n_nodes - 1);
    const PetscInt nelem = PetscMax(0, elem_end - elem_start);
    (*elem_lens)[static_cast<std::size_t>(r)] = nelem;
    (*fine_lens)[static_cast<std::size_t>(r)] = nelem * sub_n;
    node_start = node_end;
  }
}

// 从已经 setup 好的粗位移 DMDA 读取实际分区，并推导 fda/cda 需要的 ownership 数组。
// PETSc 可能通过 DMSetFromOptions 调整 uda，所以这里重新读取实际 px/py/pz。
// 输出的 ex/ey/ez 和 fx/fy/fz 会用于创建与 uda 对齐的粗单元/细密度 DMDA。
PetscErrorCode ems_get_aligned_ownership(DM uda,
                                         const Grid &grid,
                                         PetscInt sub_n,
                                         PetscInt *px,
                                         PetscInt *py,
                                         PetscInt *pz,
                                         std::vector<PetscInt> *ex_lens,
                                         std::vector<PetscInt> *ey_lens,
                                         std::vector<PetscInt> *ez_lens,
                                         std::vector<PetscInt> *fx_lens,
                                         std::vector<PetscInt> *fy_lens,
                                         std::vector<PetscInt> *fz_lens) {
  const PetscInt *ux = nullptr;
  const PetscInt *uy = nullptr;
  const PetscInt *uz = nullptr;
  PetscCall(DMDAGetInfo(uda, nullptr, nullptr, nullptr, nullptr,
                        px, py, pz, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr));
  PetscCall(DMDAGetOwnershipRanges(uda, &ux, &uy, &uz));
  ems_ownership_from_nodal_range(ux, *px, grid.nx, sub_n, ex_lens, fx_lens);
  ems_ownership_from_nodal_range(uy, *py, grid.ny, sub_n, ey_lens, fy_lens);
  ems_ownership_from_nodal_range(uz, *pz, grid.nz, sub_n, ez_lens, fz_lens);
  return 0;
}

// 运行时检查 fda 的 ghost 区是否覆盖当前 K-cache 需要读取的细密度范围。
// 如果 fine_stencil 或 ownership 设置不够，这里会直接报错，避免静默越界读 rho。
// stage 用于在错误信息里标明是哪个计算阶段触发检查。
PetscErrorCode check_ems_fda_ghost_covers_element_box(DM fda,
                                                      const ElementCacheBox &box,
                                                      PetscInt sub_n,
                                                      const char *stage) {
  PetscInt fxs = 0, fys = 0, fzs = 0, fxm = 0, fym = 0, fzm = 0;
  PetscCall(DMDAGetGhostCorners(fda, &fxs, &fys, &fzs, &fxm, &fym, &fzm));
  const PetscInt need_x0 = box.xs * sub_n;
  const PetscInt need_y0 = box.ys * sub_n;
  const PetscInt need_z0 = box.zs * sub_n;
  const PetscInt need_x1 = (box.xs + box.xm) * sub_n;
  const PetscInt need_y1 = (box.ys + box.ym) * sub_n;
  const PetscInt need_z1 = (box.zs + box.zm) * sub_n;
  PetscCheck(need_x0 >= fxs && need_x1 <= fxs + fxm &&
                 need_y0 >= fys && need_y1 <= fys + fym &&
                 need_z0 >= fzs && need_z1 <= fzs + fzm,
             PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
             "%s: fda ghost does not cover EMsFEM fine cells: need x=[%lld,%lld) y=[%lld,%lld) z=[%lld,%lld), ghost x=[%lld,%lld) y=[%lld,%lld) z=[%lld,%lld)",
             stage,
             static_cast<long long>(need_x0),
             static_cast<long long>(need_x1),
             static_cast<long long>(need_y0),
             static_cast<long long>(need_y1),
             static_cast<long long>(need_z0),
             static_cast<long long>(need_z1),
             static_cast<long long>(fxs),
             static_cast<long long>(fxs + fxm),
             static_cast<long long>(fys),
             static_cast<long long>(fys + fym),
             static_cast<long long>(fzs),
             static_cast<long long>(fzs + fzm));
  return 0;
}

// 运行时检查 uda 的 ghost 节点是否覆盖当前粗单元盒子需要的位移节点范围。
// 一个 H8 粗单元需要 8 个角节点，因此单元盒子的节点范围比单元范围右端多一层。
// 该检查用于保护 MatMult 和灵敏度计算中的局部位移访问。
PetscErrorCode check_ems_uda_ghost_covers_element_box(DM uda,
                                                      const ElementCacheBox &box,
                                                      const char *stage) {
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(DMDAGetGhostCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));
  const PetscInt need_x0 = box.xs;
  const PetscInt need_y0 = box.ys;
  const PetscInt need_z0 = box.zs;
  const PetscInt need_x1 = box.xs + box.xm + 1;
  const PetscInt need_y1 = box.ys + box.ym + 1;
  const PetscInt need_z1 = box.zs + box.zm + 1;
  PetscCheck(need_x0 >= xs && need_x1 <= xs + xm &&
                 need_y0 >= ys && need_y1 <= ys + ym &&
                 need_z0 >= zs && need_z1 <= zs + zm,
             PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
             "%s: uda ghost does not cover EMsFEM coarse element nodes: need x=[%lld,%lld) y=[%lld,%lld) z=[%lld,%lld), ghost x=[%lld,%lld) y=[%lld,%lld) z=[%lld,%lld)",
             stage,
             static_cast<long long>(need_x0),
             static_cast<long long>(need_x1),
             static_cast<long long>(need_y0),
             static_cast<long long>(need_y1),
             static_cast<long long>(need_z0),
             static_cast<long long>(need_z1),
             static_cast<long long>(xs),
             static_cast<long long>(xs + xm),
             static_cast<long long>(ys),
             static_cast<long long>(ys + ym),
             static_cast<long long>(zs),
             static_cast<long long>(zs + zm));
  return 0;
}

// 创建 EMsFEM 优化所需的两个核心 DMDA：uda 和 fda。
// uda 管理粗节点位移场，每个节点 3 个自由度；fda 管理细密度场，每个细胞 1 个密度。
// 关键点是先创建 uda，再根据 uda 的节点 ownership 推导 fda 的细密度 ownership，保证两者分区对齐。
PetscErrorCode create_ems_opt_dms(const Grid &grid,
                                  PetscInt sub_n,
                                  PetscInt fine_stencil_width,
                                  DM *uda,
                                  DM *fda) {
  PetscMPIInt ranks = 1;
  const PetscInt fstencil = PetscMax(1, fine_stencil_width);
  PetscInt px = 1, py = 1, pz = 1;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &ranks));
  PetscCall(check_ems_fine_grid_size(grid, sub_n));
  PetscCall(choose_ems_process_grid(grid, sub_n, fstencil, ranks,
                                    &px, &py, &pz));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "EMsFEM optimizer DMDA process grid: px=%lld py=%lld pz=%lld, fine_stencil=%lld\n",
                        static_cast<long long>(px),
                        static_cast<long long>(py),
                        static_cast<long long>(pz),
                        static_cast<long long>(fstencil)));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         grid.nx, grid.ny, grid.nz, px, py, pz, 3, 1,
                         nullptr, nullptr, nullptr, uda));
  PetscCall(DMSetFromOptions(*uda));
  PetscCall(DMSetUp(*uda));

  std::vector<PetscInt> ex_lens, ey_lens, ez_lens;
  std::vector<PetscInt> fx_lens, fy_lens, fz_lens;
  PetscCall(ems_get_aligned_ownership(*uda, grid, sub_n, &px, &py, &pz,
                                      &ex_lens, &ey_lens, &ez_lens,
                                      &fx_lens, &fy_lens, &fz_lens));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         fine_nelx(grid, sub_n), fine_nely(grid, sub_n),
                         fine_nelz(grid, sub_n), px, py, pz, 1, fstencil,
                         fx_lens.data(), fy_lens.data(), fz_lens.data(), fda));
  PetscCall(DMSetUp(*fda));
  return 0;
}

// 在细密度 DMDA 上创建设计域 mask 和初始密度 rho。
// mask=0 表示非设计/空域，mask=1 表示可设计区域，mask=2 表示强制实体区域。
// 初始 rho 在设计域内取体积分数，在强制实体处取 1，在 void 区取 void_density。
PetscErrorCode create_fine_mask_and_density(DM fda,
                                            const Grid &grid,
                                            const DensityOptions &density_options,
                                            const OptimizerOptions &opt,
                                            PetscInt sub_n,
                                            Vec *mask,
                                            Vec *rho) {
  PetscScalar ***m = nullptr;
  PetscScalar ***r = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(DMCreateGlobalVector(fda, mask));
  PetscCall(VecDuplicate(*mask, rho));
  PetscCall(DMDAVecGetArray(fda, *mask, &m));
  PetscCall(DMDAVecGetArray(fda, *rho, &r));
  PetscCall(DMDAGetCorners(fda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscBool design =
            fine_cell_density(i, j, k, grid, density_options, sub_n) >=
                    density_options.mask_threshold
                ? PETSC_TRUE
                : PETSC_FALSE;
        const PetscBool hard_solid =
            hard_solid_fine_cell(i, j, k, grid, density_options, sub_n);
        if (hard_solid) {
          m[k][j][i] = 2.0;
          r[k][j][i] = 1.0;
        } else {
          m[k][j][i] = design ? 1.0 : 0.0;
          r[k][j][i] = design ? opt.volfrac : density_options.void_density;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArray(fda, *mask, &m));
  PetscCall(DMDAVecRestoreArray(fda, *rho, &r));
  return 0;
}

// 预计算密度过滤器的分母，也就是每个细胞邻域内有效 mask 权重之和。
// 分母只和 mask 与过滤半径有关，优化迭代中可重复使用，避免每步重复计算。
// radius<=0 时不做过滤，分母统一设为 1。
PetscErrorCode compute_filter_denominator(DM eda, Vec mask, PetscReal radius,
                                          Vec denom) {
  if (radius <= 0.0) {
    PetscCall(VecSet(denom, 1.0));
    return 0;
  }
  Vec local_mask = nullptr;
  PetscScalar ***m = nullptr, ***d = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  const PetscInt r = PetscMax(0, static_cast<PetscInt>(PetscCeilReal(radius)));
  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecGetArray(eda, denom, &d));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        PetscReal sum = 0.0;
        for (PetscInt dz = -r; dz <= r; ++dz) {
          const PetscInt kk = k + dz;
          if (kk < 0 || kk >= ez) continue;
          for (PetscInt dy = -r; dy <= r; ++dy) {
            const PetscInt jj = j + dy;
            if (jj < 0 || jj >= ey) continue;
            for (PetscInt dx = -r; dx <= r; ++dx) {
              const PetscInt ii = i + dx;
              if (ii < 0 || ii >= ex) continue;
              const PetscReal w = cone_weight(dx, dy, dz, radius);
              if (w > 0.0 && PetscRealPart(m[kk][jj][ii]) > 0.5) sum += w;
            }
          }
        }
        d[k][j][i] = PetscMax(sum, 1.0e-12);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecRestoreArray(eda, denom, &d));
  PetscCall(VecDestroy(&local_mask));
  return 0;
}

// 对设计密度 rho_design 做锥形密度过滤，得到物理密度 rho_phys。
// 过滤只在 mask 有效区域内进行；硬实体保持 1，非设计 void 保持 void_density。
// 需要通过 DMGlobalToLocal 同步 ghost 区，因为过滤会读取邻居细胞密度。
PetscErrorCode apply_density_filter(DM eda, Vec rho_design, Vec mask,
                                    Vec denom, PetscReal radius,
                                    PetscReal void_density, Vec rho_phys) {
  if (radius <= 0.0) {
    PetscScalar ***r = nullptr, ***m = nullptr, ***out = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscCall(DMDAVecGetArrayRead(eda, rho_design, &r));
    PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
    PetscCall(DMDAVecGetArray(eda, rho_phys, &out));
    PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (PetscRealPart(m[k][j][i]) > 1.5) out[k][j][i] = 1.0;
          else if (PetscRealPart(m[k][j][i]) <= 0.5) out[k][j][i] = void_density;
          else out[k][j][i] = r[k][j][i];
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(eda, rho_design, &r));
    PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
    PetscCall(DMDAVecRestoreArray(eda, rho_phys, &out));
    return 0;
  }
  Vec local_rho = nullptr, local_mask = nullptr;
  PetscScalar ***r = nullptr, ***m = nullptr, ***den = nullptr, ***out = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  const PetscInt rr = PetscMax(0, static_cast<PetscInt>(PetscCeilReal(radius)));
  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCall(DMCreateLocalVector(eda, &local_rho));
  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, rho_design, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(eda, rho_design, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalBegin(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayRead(eda, local_rho, &r));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecGetArrayRead(eda, denom, &den));
  PetscCall(DMDAVecGetArray(eda, rho_phys, &out));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (PetscRealPart(m[k][j][i]) > 1.5) {
          out[k][j][i] = 1.0;
          continue;
        }
        if (PetscRealPart(m[k][j][i]) <= 0.5) {
          out[k][j][i] = void_density;
          continue;
        }
        PetscReal sum = 0.0;
        for (PetscInt dz = -rr; dz <= rr; ++dz) {
          const PetscInt kk = k + dz;
          if (kk < 0 || kk >= ez) continue;
          for (PetscInt dy = -rr; dy <= rr; ++dy) {
            const PetscInt jj = j + dy;
            if (jj < 0 || jj >= ey) continue;
            for (PetscInt dx = -rr; dx <= rr; ++dx) {
              const PetscInt ii = i + dx;
              if (ii < 0 || ii >= ex) continue;
              if (PetscRealPart(m[kk][jj][ii]) <= 0.5) continue;
              const PetscReal w = cone_weight(dx, dy, dz, radius);
              if (w > 0.0) sum += w * PetscRealPart(r[kk][jj][ii]);
            }
          }
        }
        out[k][j][i] = sum / PetscRealPart(den[k][j][i]);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, local_rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecRestoreArrayRead(eda, denom, &den));
  PetscCall(DMDAVecRestoreArray(eda, rho_phys, &out));
  PetscCall(VecDestroy(&local_rho));
  PetscCall(VecDestroy(&local_mask));
  return 0;
}

// 根据当前优化迭代步数计算 Heaviside 投影的 beta。
// beta 会按 interval 逐步放大，直到 beta_max，用于让密度从灰度逐渐变成更接近 0/1。
// 如果未启用 Heaviside，则该函数的结果通常不会被使用。
PetscReal heaviside_beta_for_iter(const OptimizerOptions &opt, PetscInt iter) {
  if (!opt.heaviside_projection) return 0.0;
  PetscReal beta = PetscMax(1.0e-12, opt.heaviside_beta_initial);
  if (opt.heaviside_beta_interval <= 0) {
    if (opt.max_iter <= 1 || opt.heaviside_beta_max <= beta) {
      return PetscMin(beta, opt.heaviside_beta_max);
    }
    const PetscReal ratio = opt.heaviside_beta_max / beta;
    const PetscInt stages = PetscMax(
        static_cast<PetscInt>(std::ceil(std::log(ratio) / std::log(2.0))),
        static_cast<PetscInt>(1));
    const PetscReal progress =
        static_cast<PetscReal>(PetscMax(iter - 1, 0)) *
        static_cast<PetscReal>(stages + 1) /
        static_cast<PetscReal>(PetscMax(opt.max_iter, 1));
    const PetscInt stage = PetscMin(
        stages, static_cast<PetscInt>(std::floor(progress)));
    for (PetscInt s = 0; s < stage; ++s) beta *= 2.0;
    return PetscMin(beta, opt.heaviside_beta_max);
  }
  if (opt.heaviside_beta_interval > 0 && iter > 1) {
    const PetscInt stage = (iter - 1) / opt.heaviside_beta_interval;
    for (PetscInt s = 0; s < stage; ++s) beta *= 2.0;
  }
  return PetscMin(beta, opt.heaviside_beta_max);
}

// 平滑 Heaviside 投影函数。
// eta 是阈值位置，beta 控制曲线陡峭程度；beta 越大，中间密度越快被推向 0 或 1。
// 返回值用于把过滤后的密度转换为投影后的物理密度。
PetscReal smooth_heaviside_value(PetscReal x, PetscReal beta, PetscReal eta) {
  const PetscReal den =
      PetscTanhReal(beta * eta) + PetscTanhReal(beta * (1.0 - eta));
  if (den <= 0.0) return x;
  return (PetscTanhReal(beta * eta) + PetscTanhReal(beta * (x - eta))) / den;
}

// 平滑 Heaviside 投影对输入密度 x 的导数。
// 灵敏度反传时需要乘以这个导数，把投影后目标函数梯度传回过滤密度。
// beta 较大时导数会在 eta 附近集中，优化会更接近硬阈值行为。
PetscReal smooth_heaviside_derivative(PetscReal x, PetscReal beta,
                                      PetscReal eta) {
  const PetscReal den =
      PetscTanhReal(beta * eta) + PetscTanhReal(beta * (1.0 - eta));
  if (den <= 0.0) return 1.0;
  const PetscReal t = PetscTanhReal(beta * (x - eta));
  return beta * (1.0 - t * t) / den;
}

// 对过滤后的密度执行 Heaviside 投影。
// 可设计区域使用 smooth_heaviside_value，硬实体和 void 区域保持固定值。
// 输出 projected 通常作为本步用于刚度计算的物理密度。
PetscErrorCode apply_heaviside_projection(DM fda, Vec filtered, Vec mask,
                                          const OptimizerOptions &opt,
                                          PetscReal beta,
                                          PetscReal void_density,
                                          Vec projected) {
  if (!opt.heaviside_projection) {
    PetscCall(VecCopy(filtered, projected));
    return 0;
  }
  PetscScalar ***src = nullptr, ***m = nullptr, ***dst = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(DMDAVecGetArrayRead(fda, filtered, &src));
  PetscCall(DMDAVecGetArrayRead(fda, mask, &m));
  PetscCall(DMDAVecGetArray(fda, projected, &dst));
  PetscCall(DMDAGetCorners(fda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (mask_value > 1.5) {
          dst[k][j][i] = 1.0;
        } else if (mask_value <= 0.5) {
          dst[k][j][i] = void_density;
        } else {
          const PetscReal value = smooth_heaviside_value(
              PetscRealPart(src[k][j][i]), beta, opt.heaviside_eta);
          dst[k][j][i] = PetscMax(void_density, PetscMin(1.0, value));
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(fda, filtered, &src));
  PetscCall(DMDAVecRestoreArrayRead(fda, mask, &m));
  PetscCall(DMDAVecRestoreArray(fda, projected, &dst));
  return 0;
}

// 将投影后密度上的目标函数梯度反传到过滤密度上。
// 可设计区域乘以 Heaviside 导数；硬实体/void 区域不参与设计更新。
// 输出 dc_filtered 后续还要经过密度过滤器的伴随反传。
PetscErrorCode apply_heaviside_sensitivity(DM fda, Vec dc_projected,
                                           Vec filtered, Vec mask,
                                           const OptimizerOptions &opt,
                                           PetscReal beta, Vec dc_filtered) {
  if (!opt.heaviside_projection) {
    PetscCall(VecCopy(dc_projected, dc_filtered));
    return 0;
  }
  PetscScalar ***dc = nullptr, ***rho = nullptr, ***m = nullptr, ***out = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(DMDAVecGetArrayRead(fda, dc_projected, &dc));
  PetscCall(DMDAVecGetArrayRead(fda, filtered, &rho));
  PetscCall(DMDAVecGetArrayRead(fda, mask, &m));
  PetscCall(DMDAVecGetArray(fda, dc_filtered, &out));
  PetscCall(DMDAGetCorners(fda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        if (mask_value <= 0.5 || mask_value > 1.5) {
          out[k][j][i] = 0.0;
        } else {
          const PetscReal deriv = smooth_heaviside_derivative(
              PetscRealPart(rho[k][j][i]), beta, opt.heaviside_eta);
          out[k][j][i] = PetscRealPart(dc[k][j][i]) * deriv;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(fda, dc_projected, &dc));
  PetscCall(DMDAVecRestoreArrayRead(fda, filtered, &rho));
  PetscCall(DMDAVecRestoreArrayRead(fda, mask, &m));
  PetscCall(DMDAVecRestoreArray(fda, dc_filtered, &out));
  return 0;
}

// 密度过滤器的伴随灵敏度反传。
// 正向过滤是 rho_phys_i = sum_j w_ij rho_j / denom_i；反向要把 dc_phys_i 分配回邻域 j。
// 该函数得到设计变量上的梯度 dc_design，并保持硬实体/void 区域不参与更新。
PetscErrorCode apply_sensitivity_filter_adjoint(DM eda, Vec dc_phys, Vec mask,
                                                Vec denom, PetscReal radius,
                                                Vec dc_design) {
  if (radius <= 0.0) {
    PetscScalar ***src = nullptr, ***m = nullptr, ***dst = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscCall(DMDAVecGetArrayRead(eda, dc_phys, &src));
    PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
    PetscCall(DMDAVecGetArray(eda, dc_design, &dst));
    PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          dst[k][j][i] = PetscRealPart(m[k][j][i]) > 1.5 ? 0.0 : src[k][j][i];
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(eda, dc_phys, &src));
    PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
    PetscCall(DMDAVecRestoreArray(eda, dc_design, &dst));
    return 0;
  }
  Vec local_dc = nullptr, local_mask = nullptr, local_denom = nullptr;
  PetscScalar ***dc = nullptr, ***m = nullptr, ***den = nullptr, ***out = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ex = 0, ey = 0, ez = 0;
  const PetscInt rr = PetscMax(0, static_cast<PetscInt>(PetscCeilReal(radius)));
  PetscCall(DMDAGetInfo(eda, nullptr, &ex, &ey, &ez, nullptr, nullptr, nullptr,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCall(DMCreateLocalVector(eda, &local_dc));
  PetscCall(DMCreateLocalVector(eda, &local_mask));
  PetscCall(DMCreateLocalVector(eda, &local_denom));
  PetscCall(DMGlobalToLocalBegin(eda, dc_phys, INSERT_VALUES, local_dc));
  PetscCall(DMGlobalToLocalEnd(eda, dc_phys, INSERT_VALUES, local_dc));
  PetscCall(DMGlobalToLocalBegin(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(eda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalBegin(eda, denom, INSERT_VALUES, local_denom));
  PetscCall(DMGlobalToLocalEnd(eda, denom, INSERT_VALUES, local_denom));
  PetscCall(DMDAVecGetArrayRead(eda, local_dc, &dc));
  PetscCall(DMDAVecGetArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecGetArrayRead(eda, local_denom, &den));
  PetscCall(DMDAVecGetArray(eda, dc_design, &out));
  PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (PetscRealPart(m[k][j][i]) > 1.5) {
          out[k][j][i] = 0.0;
          continue;
        }
        if (PetscRealPart(m[k][j][i]) <= 0.5) {
          out[k][j][i] = 0.0;
          continue;
        }
        PetscReal sum = 0.0;
        for (PetscInt dz = -rr; dz <= rr; ++dz) {
          const PetscInt kk = k + dz;
          if (kk < 0 || kk >= ez) continue;
          for (PetscInt dy = -rr; dy <= rr; ++dy) {
            const PetscInt jj = j + dy;
            if (jj < 0 || jj >= ey) continue;
            for (PetscInt dx = -rr; dx <= rr; ++dx) {
              const PetscInt ii = i + dx;
              if (ii < 0 || ii >= ex) continue;
              if (PetscRealPart(m[kk][jj][ii]) <= 0.5) continue;
              const PetscReal w = cone_weight(dx, dy, dz, radius);
              if (w > 0.0) {
                sum += w * PetscRealPart(dc[kk][jj][ii]) /
                       PetscRealPart(den[kk][jj][ii]);
              }
            }
          }
        }
        out[k][j][i] = sum;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(eda, local_dc, &dc));
  PetscCall(DMDAVecRestoreArrayRead(eda, local_mask, &m));
  PetscCall(DMDAVecRestoreArrayRead(eda, local_denom, &den));
  PetscCall(DMDAVecRestoreArray(eda, dc_design, &out));
  PetscCall(VecDestroy(&local_dc));
  PetscCall(VecDestroy(&local_mask));
  PetscCall(VecDestroy(&local_denom));
  return 0;
}

// 设置体积约束对每个设计变量的导数 dv。
// 可设计区域 dv=1，硬实体和 void 区域 dv=0，因为它们不由优化器更新。
// OC 更新中会用 dc/dv 的比值决定密度增减方向。
PetscErrorCode set_active_volume_vector(DM fda, Vec mask, Vec dv) {
  PetscScalar ***m = nullptr, ***v = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscCall(DMDAVecGetArrayRead(fda, mask, &m));
  PetscCall(DMDAVecGetArray(fda, dv, &v));
  PetscCall(DMDAGetCorners(fda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mv = PetscRealPart(m[k][j][i]);
        v[k][j][i] = (mv > 0.5 && mv <= 1.5) ? 1.0 : 0.0;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(fda, mask, &m));
  PetscCall(DMDAVecRestoreArray(fda, dv, &v));
  return 0;
}

// 统计全局可设计细胞数量。
// 每个 rank 只遍历自己拥有的 fda 范围，然后用 MPI/PETSc 全局求和。
// 结果用于把目标体积分数换算成可设计区域的平均密度约束。
PetscErrorCode design_mask_sum(DM fda, Vec mask, PetscReal *global_design_sum) {
  PetscScalar ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_design_sum = 0.0;
  PetscCall(DMDAVecGetArrayRead(fda, mask, &m));
  PetscCall(DMDAGetCorners(fda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mv = PetscRealPart(m[k][j][i]);
        if (mv > 0.5 && mv <= 1.5) local_design_sum += 1.0;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(fda, mask, &m));
  PetscCallMPI(MPI_Allreduce(&local_design_sum, global_design_sum, 1,
                             MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  return 0;
}

// 计算可设计区域内当前密度的全局平均值。
// 只统计 mask=1 的细胞，硬实体和 void 区域不计入可设计体积分数。
// 该值常用于日志输出，判断优化是否满足体积约束。
PetscErrorCode compute_design_mean(DM fda, Vec rho, Vec mask,
                                   PetscReal *global_mean) {
  PetscScalar ***r = nullptr, ***m = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_sum = 0.0, local_count = 0.0;
  PetscReal global_sum = 0.0, global_count = 0.0;
  PetscCall(DMDAVecGetArrayRead(fda, rho, &r));
  PetscCall(DMDAVecGetArrayRead(fda, mask, &m));
  PetscCall(DMDAGetCorners(fda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal mv = PetscRealPart(m[k][j][i]);
        if (mv > 0.5 && mv <= 1.5) {
          local_sum += PetscRealPart(r[k][j][i]);
          local_count += 1.0;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(fda, rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(fda, mask, &m));
  PetscCallMPI(MPI_Allreduce(&local_sum, &global_sum, 1, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_count, &global_count, 1, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  *global_mean = global_count > 0.0 ? global_sum / global_count : 0.0;
  return 0;
}

// 对细密度场执行 z 方向拔模闭合/可制造性处理。
// 该操作沿 z 方向传播密度约束，减少不符合拔模方向的孤立或倒扣特征。
// 同时会输出 sensitivity_scale，供灵敏度回传时近似考虑该闭合操作的影响。
PetscErrorCode apply_z_draft_closure(DM fda,
                                     Vec mask,
                                     PetscReal void_density,
                                     Vec rho_phys) {
  MPI_Comm zcol_comm = MPI_COMM_NULL;
  Vec local_rho = nullptr, local_mask = nullptr;
  PetscScalar ***r = nullptr, ***m = nullptr, ***out = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt npx = 1, npy = 1, npz = 1;
  const PetscInt *x_ownership = nullptr;
  const PetscInt *y_ownership = nullptr;
  const PetscInt *z_ownership = nullptr;
  PetscCall(DMCreateLocalVector(fda, &local_rho));
  PetscCall(DMCreateLocalVector(fda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(fda, rho_phys, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(fda, rho_phys, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalBegin(fda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(fda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayRead(fda, local_rho, &r));
  PetscCall(DMDAVecGetArrayRead(fda, local_mask, &m));
  PetscCall(DMDAVecGetArray(fda, rho_phys, &out));
  PetscCall(DMDAGetCorners(fda, &xs, &ys, &zs, &xm, &ym, &zm));
  PetscCall(DMDAGetInfo(fda, nullptr, nullptr, nullptr, nullptr, &npx, &npy, &npz,
                        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr));
  PetscCall(DMDAGetOwnershipRanges(fda, &x_ownership, &y_ownership,
                                   &z_ownership));

  auto ownership_index = [](const PetscInt *ranges, PetscInt nproc,
                            PetscInt start) {
    PetscInt offset = 0;
    for (PetscInt p = 0; p < nproc; ++p) {
      if (start == offset) return p;
      offset += ranges[p];
    }
    return nproc;
  };
  const PetscInt xproc = ownership_index(x_ownership, npx, xs);
  const PetscInt yproc = ownership_index(y_ownership, npy, ys);
  const PetscInt zproc = ownership_index(z_ownership, npz, zs);
  PetscCheck(xproc < npx && yproc < npy && zproc < npz, PETSC_COMM_WORLD,
             PETSC_ERR_PLIB,
             "Could not identify EMsFEM fine-density DMDA process coordinates");
  PetscCallMPI(MPI_Comm_split(PETSC_COMM_WORLD,
                              static_cast<PetscMPIInt>(xproc + npx * yproc),
                              static_cast<PetscMPIInt>(zproc),
                              &zcol_comm));
  PetscMPIInt zrank = 0, zsize = 1;
  PetscCallMPI(MPI_Comm_rank(zcol_comm, &zrank));
  PetscCallMPI(MPI_Comm_size(zcol_comm, &zsize));

  const PetscInt plane = xm * ym;
  PetscMPIInt mpi_plane = static_cast<PetscMPIInt>(plane);
  std::vector<PetscReal> local_values(static_cast<std::size_t>(plane) *
                                      static_cast<std::size_t>(zm), 0.0);
  std::vector<PetscReal> prefix_values(local_values.size(), 0.0);
  std::vector<PetscReal> local_max(static_cast<std::size_t>(plane), 0.0);
  std::vector<PetscReal> below(static_cast<std::size_t>(plane), 0.0);
  std::vector<PetscReal> above(static_cast<std::size_t>(plane), 0.0);

  auto col_id = [=](PetscInt i, PetscInt j) {
    return (i - xs) + xm * (j - ys);
  };
  auto val_id = [=](PetscInt i, PetscInt j, PetscInt k) {
    return static_cast<std::size_t>(col_id(i, j)) +
           static_cast<std::size_t>(plane) *
               static_cast<std::size_t>(k - zs);
  };

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        PetscReal value = PetscRealPart(r[k][j][i]);
        if (PetscRealPart(m[k][j][i]) > 1.5) value = 1.0;
        if (PetscRealPart(m[k][j][i]) <= 0.5) value = void_density;
        local_values[val_id(i, j, k)] = value;
        local_max[static_cast<std::size_t>(col_id(i, j))] =
            PetscMax(local_max[static_cast<std::size_t>(col_id(i, j))], value);
      }
    }
  }

  if (zsize > 1) {
    PetscCallMPI(MPI_Exscan(local_max.data(), below.data(), mpi_plane,
                            MPIU_REAL, MPI_MAX, zcol_comm));
    if (zrank == 0) std::fill(below.begin(), below.end(), 0.0);

    MPI_Comm reverse_comm = MPI_COMM_NULL;
    PetscCallMPI(MPI_Comm_split(zcol_comm, 0, zsize - 1 - zrank,
                                &reverse_comm));
    PetscCallMPI(MPI_Exscan(local_max.data(), above.data(), mpi_plane,
                            MPIU_REAL, MPI_MAX, reverse_comm));
    if (zrank == zsize - 1) std::fill(above.begin(), above.end(), 0.0);
    PetscCallMPI(MPI_Comm_free(&reverse_comm));
  }

  for (PetscInt j = ys; j < ys + ym; ++j) {
    for (PetscInt i = xs; i < xs + xm; ++i) {
      const PetscInt cid = col_id(i, j);
      PetscReal running = below[static_cast<std::size_t>(cid)];
      for (PetscInt k = zs; k < zs + zm; ++k) {
        const std::size_t id = val_id(i, j, k);
        running = PetscMax(running, local_values[id]);
        prefix_values[id] = running;
      }
      running = above[static_cast<std::size_t>(cid)];
      for (PetscInt k = zs + zm - 1; k >= zs; --k) {
        const std::size_t id = val_id(i, j, k);
        running = PetscMax(running, local_values[id]);
        PetscReal value =
            PetscMax(local_values[id], PetscMin(prefix_values[id], running));
        if (PetscRealPart(m[k][j][i]) > 1.5) value = 1.0;
        if (PetscRealPart(m[k][j][i]) <= 0.5) value = void_density;
        out[k][j][i] = PetscMax(void_density, PetscMin(1.0, value));
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(fda, local_rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(fda, local_mask, &m));
  PetscCall(DMDAVecRestoreArray(fda, rho_phys, &out));
  PetscCall(VecDestroy(&local_rho));
  PetscCall(VecDestroy(&local_mask));
  PetscCallMPI(MPI_Comm_free(&zcol_comm));
  return 0;
}

// 根据当前物理密度 rho_phys 重建本 rank 的 EMsFEM 单元刚度缓存。
// 先把 rho_phys 同步到带 ghost 的 local_rho，再为 cache_box 内每个粗单元采样 sub_n^3 个细胞。
// 每个单元会通过 ANN 多尺度形函数计算 24x24 ke，供后续 KSP 的多次 MatMult 重复使用。
PetscErrorCode rebuild_local_k_cache_from_rho(EmSfemAnnContext *ctx,
                                              DM fda,
                                              Vec rho_phys,
                                              EmsCacheTiming *timing = nullptr) {
  Vec local_rho = nullptr;
  PetscScalar ***rho = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt ncache = 0;
  const PetscInt sub_n = ctx->ems_options.sub_n;
  EmsCacheTiming local_timing;
  PetscLogDouble total_start = 0.0, total_end = 0.0;
  PetscCall(PetscTime(&total_start));
  PetscCall(DMCreateLocalVector(fda, &local_rho));
  PetscCall(DMGlobalToLocalBegin(fda, rho_phys, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(fda, rho_phys, INSERT_VALUES, local_rho));
  PetscCall(DMDAVecGetArrayRead(fda, local_rho, &rho));
  PetscCall(DMDAGetCorners(ctx->da, &xs, &ys, &zs, &xm, &ym, &zm));
  ElementCacheBox box;
  box.xs = PetscMax(0, xs - 1);
  box.ys = PetscMax(0, ys - 1);
  box.zs = PetscMax(0, zs - 1);
  const PetscInt xe = PetscMin(ctx->grid.nx - 1, xs + xm);
  const PetscInt ye = PetscMin(ctx->grid.ny - 1, ys + ym);
  const PetscInt ze = PetscMin(ctx->grid.nz - 1, zs + zm);
  box.xm = PetscMax(0, xe - box.xs);
  box.ym = PetscMax(0, ye - box.ys);
  box.zm = PetscMax(0, ze - box.zs);
  ctx->cache_box = box;
  PetscCall(check_ems_fda_ghost_covers_element_box(
      fda, box, sub_n, "EMsFEM K-cache rebuild"));
  const std::size_t elems =
      static_cast<std::size_t>(box.xm) * static_cast<std::size_t>(box.ym) *
      static_cast<std::size_t>(box.zm);
  const double gib =
      static_cast<double>(elems) * static_cast<double>(kElementDofs * kElementDofs) *
      sizeof(PetscReal) / 1073741824.0;
  PetscCheck(ctx->ems_options.cache_gib_limit <= 0.0 ||
                 gib <= ctx->ems_options.cache_gib_limit,
             PETSC_COMM_WORLD, PETSC_ERR_MEM,
             "Local EMsFEM optimizer K cache would use %.3g GiB, above -ems_cache_gib_limit %.3g.",
             gib, static_cast<double>(ctx->ems_options.cache_gib_limit));
  ctx->k_cache.assign(elems * kElementDofs * kElementDofs, 0.0);
  for (PetscInt ez = box.zs; ez < box.zs + box.zm; ++ez) {
    for (PetscInt ey = box.ys; ey < box.ys + box.ym; ++ey) {
      for (PetscInt ex = box.xs; ex < box.xs + box.xm; ++ex) {
        PetscReal *ke = cached_element_matrix(ctx, ex, ey, ez);
        std::vector<PetscReal> material;
        material.reserve(static_cast<std::size_t>(sub_n * sub_n * sub_n));
        PetscLogDouble material_start = 0.0, material_end = 0.0;
        PetscCall(PetscTime(&material_start));
        for (PetscInt fz = 0; fz < sub_n; ++fz) {
          for (PetscInt fy = 0; fy < sub_n; ++fy) {
            for (PetscInt fx = 0; fx < sub_n; ++fx) {
              const PetscReal rv =
                  PetscRealPart(rho[ez * sub_n + fz][ey * sub_n + fy]
                                      [ex * sub_n + fx]);
              material.push_back(simp_scale(rv, ctx->density_options));
            }
          }
        }
        PetscCall(PetscTime(&material_end));
        local_timing.material_sampling_s +=
            static_cast<PetscReal>(material_end - material_start);
        PetscCall(compute_ems_element_matrix_from_material(ctx, material, ke,
                                                           &local_timing));
        ++ncache;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayRead(fda, local_rho, &rho));
  PetscCall(VecDestroy(&local_rho));
  PetscCall(PetscTime(&total_end));
  local_timing.total_s = static_cast<PetscReal>(total_end - total_start);
  if (timing != nullptr) {
    PetscReal local_values[4] = {local_timing.total_s,
                                 local_timing.material_sampling_s,
                                 local_timing.ann_shape_predict_s,
                                 local_timing.matrix_assembly_s};
    PetscReal global_values[4] = {};
    PetscCallMPI(MPI_Allreduce(local_values, global_values, 4, MPIU_REAL,
                               MPI_MAX, PETSC_COMM_WORLD));
    timing->total_s = global_values[0];
    timing->material_sampling_s = global_values[1];
    timing->ann_shape_predict_s = global_values[2];
    timing->matrix_assembly_s = global_values[3];
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "EMsFEM ANN fine-density K cache rebuilt: local coarse elements=%lld fine subcells/element=%lld max %.3f GiB/rank time[s]: total=%.3g material=%.3g ann_shape=%.3g ems_ke=%.3g\n",
                        static_cast<long long>(ncache),
                        static_cast<long long>(sub_n * sub_n * sub_n),
                        gib,
                        static_cast<double>(timing != nullptr ? timing->total_s : local_timing.total_s),
                        static_cast<double>(timing != nullptr ? timing->material_sampling_s : local_timing.material_sampling_s),
                        static_cast<double>(timing != nullptr ? timing->ann_shape_predict_s : local_timing.ann_shape_predict_s),
                        static_cast<double>(timing != nullptr ? timing->matrix_assembly_s : local_timing.matrix_assembly_s)));
  return 0;
}

// 把粗节点索引 (i,j,k) 转换成实际物理坐标 (x,y,z)。
// 控制臂几何、固定端选择、载荷节点筛选和力矩分配都会用这个坐标。
// 坐标尺度由 Grid 中的物理高度和几何比例函数决定。
void node_xyz(PetscInt i, PetscInt j, PetscInt k,
              const Grid &grid,
              PetscReal *x,
              PetscReal *y,
              PetscReal *z) {
  *x = domain_length(grid) * static_cast<PetscReal>(i) /
       static_cast<PetscReal>(grid.nx - 1);
  *y = domain_width(grid) * static_cast<PetscReal>(j) /
       static_cast<PetscReal>(grid.ny - 1);
  *z = domain_height(grid) * static_cast<PetscReal>(k) /
       static_cast<PetscReal>(grid.nz - 1);
}

// 判断控制臂算例中某个粗节点是否属于固定约束区域。
// 通常固定区域对应车身连接端或安装孔附近，所有 3 个位移自由度会被约束。
// 该函数只负责几何判断，真正施加边界条件在矩阵乘法/载荷或 KSP 约束处理中完成。
PetscBool control_arm_fixed_node(PetscInt i, PetscInt j, PetscInt k,
                                 const Grid &grid) {
  PetscReal x = 0.0, y = 0.0, z = 0.0;
  node_xyz(i, j, k, grid, &x, &y, &z);
  const PetscReal DL = domain_length(grid);
  const PetscReal DW = domain_width(grid);
  const PetscReal DH = domain_height(grid);
  const PetscReal cx = 0.78 * DL;
  const PetscReal cy = 0.50 * DW;
  const PetscReal cz = 0.50 * DH;
  const PetscReal r = 0.10 * PetscMin(DL, DW);
  const PetscReal hmax = PetscMax(PetscMax(DL / (grid.nx - 1),
                                           DW / (grid.ny - 1)),
                                  DH / (grid.nz - 1));
  const PetscReal tol = 0.75 * hmax;
  const PetscReal axial = PetscMax(0.15 * PetscMin(DL, DW), 2.0 * hmax);
  const PetscReal rr = PetscSqrtReal((x - cx) * (x - cx) + (y - cy) * (y - cy));
  return (PetscAbsReal(rr - r) <= tol && PetscAbsReal(z - cz) <= axial)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

// 简单测试算例的固定节点判断。
// 这里通常把 x=0 平面固定，用于非控制臂几何的小规模验证。
PetscBool simple_fixed_node(PetscInt i) {
  return i == 0 ? PETSC_TRUE : PETSC_FALSE;
}

PetscBool benchmark_is_torsion(const char *benchmark_case) {
  return std::strcmp(benchmark_case, "torsion") == 0 ? PETSC_TRUE : PETSC_FALSE;
}

// 根据当前几何模式统一判断节点是否固定。
// 控制臂模式调用 control_arm_fixed_node，简单模式调用 simple_fixed_node。
// 这样矩阵乘法和载荷装配不需要关心具体几何类型。
PetscBool is_fixed_node(PetscInt i, PetscInt j, PetscInt k,
                        const EmSfemAnnContext &ctx) {
  return ctx.ems_options.control_arm_bc
             ? control_arm_fixed_node(i, j, k, ctx.grid)
             : simple_fixed_node(i);
}

// 为弹性辅助矩阵附加近零空间 near-nullspace。
// 三维弹性问题的刚体平移和转动模态会影响 AMG/GAMG 的粗网格构造；
// 给矩阵附加这些近零空间向量，可以显著改善弹性类预条件器的收敛性。
PetscErrorCode attach_ems_elasticity_near_nullspace(DM uda,
                                                    const EmSfemAnnContext &ctx,
                                                    Mat P) {
  PetscBool attach = PETSC_TRUE;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr,
                                "-ems_attach_elasticity_near_nullspace",
                                &attach, nullptr));
  if (!attach) return 0;

  Vec basis[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const Grid &grid = ctx.grid;
  const PetscReal cx = 0.5 * domain_length(grid);
  const PetscReal cy = 0.5 * domain_width(grid);
  const PetscReal cz = 0.5 * domain_height(grid);

  for (PetscInt m = 0; m < 6; ++m) {
    PetscCall(DMCreateGlobalVector(uda, &basis[m]));
    PetscCall(VecSet(basis[m], 0.0));
  }

  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt m = 0; m < 6; ++m) {
    PetscScalar ****v = nullptr;
    PetscCall(DMDAVecGetArrayDOF(uda, basis[m], &v));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (is_fixed_node(i, j, k, ctx)) continue;
          PetscReal x = 0.0, y = 0.0, z = 0.0;
          node_xyz(i, j, k, grid, &x, &y, &z);
          x -= cx;
          y -= cy;
          z -= cz;
          switch (m) {
          case 0:
            v[k][j][i][0] = 1.0;
            break;
          case 1:
            v[k][j][i][1] = 1.0;
            break;
          case 2:
            v[k][j][i][2] = 1.0;
            break;
          case 3:
            v[k][j][i][1] = -z;
            v[k][j][i][2] = y;
            break;
          case 4:
            v[k][j][i][0] = z;
            v[k][j][i][2] = -x;
            break;
          case 5:
            v[k][j][i][0] = -y;
            v[k][j][i][1] = x;
            break;
          }
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayDOF(uda, basis[m], &v));
  }

  for (PetscInt i = 0; i < 6; ++i) {
    for (PetscInt j = 0; j < i; ++j) {
      PetscScalar dot = 0.0;
      PetscCall(VecDot(basis[i], basis[j], &dot));
      PetscCall(VecAXPY(basis[i], -dot, basis[j]));
    }
    PetscReal norm = 0.0;
    PetscCall(VecNorm(basis[i], NORM_2, &norm));
    PetscCheck(norm > PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "Elasticity near-nullspace mode %lld is numerically zero",
               static_cast<long long>(i));
    PetscCall(VecScale(basis[i], 1.0 / norm));
  }

  MatNullSpace near_nullspace = nullptr;
  PetscCall(MatNullSpaceCreate(PETSC_COMM_WORLD, PETSC_FALSE, 6, basis,
                               &near_nullspace));
  PetscCall(MatSetNearNullSpace(P, near_nullspace));
  PetscCall(MatNullSpaceDestroy(&near_nullspace));
  for (PetscInt m = 0; m < 6; ++m) PetscCall(VecDestroy(&basis[m]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "EMsFEM ANN auxiliary matrix: attached 3D elasticity near-nullspace for GAMG\n"));
  return 0;
}

// 判断粗节点是否位于控制臂加载/连接孔附近。
// 该选择函数用于把集中力或力矩分配到一组孔周节点，而不是单个节点上。
// 返回值基于节点物理坐标与孔中心/半径的几何关系。
PetscBool select_hole_node(PetscInt i, PetscInt j, PetscInt k,
                           const Grid &grid,
                           PetscInt which) {
  PetscReal x = 0.0, y = 0.0, z = 0.0;
  node_xyz(i, j, k, grid, &x, &y, &z);
  const PetscReal DL = domain_length(grid);
  const PetscReal DW = domain_width(grid);
  const PetscReal DH = domain_height(grid);
  const PetscReal hmax = PetscMax(PetscMax(DL / (grid.nx - 1),
                                           DW / (grid.ny - 1)),
                                  DH / (grid.nz - 1));
  const PetscReal tol = 0.75 * hmax;
  PetscReal cx = 0.18 * DL;
  PetscReal cy = which == 1 ? DW : 0.0;
  const PetscReal cz = 0.50 * DH;
  const PetscReal r = 0.25 * PetscMin(DL, DH);
  const PetscReal axial = PetscMax(0.35 * PetscMin(DL, DH), 2.0 * hmax);
  const PetscReal rr = PetscSqrtReal((x - cx) * (x - cx) + (z - cz) * (z - cz));
  return (PetscAbsReal(rr - r) <= tol && PetscAbsReal(y - cy) <= axial)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

// 判断粗节点是否位于弹簧/衬套加载区域。
// 多载荷控制臂算例会用不同的选择函数构造不同工况的载荷分布。
// 这里同样只做几何筛选，不直接写入力向量。
PetscBool select_spring_node(PetscInt i, PetscInt j, PetscInt k,
                             const Grid &grid) {
  PetscReal x = 0.0, y = 0.0, z = 0.0;
  node_xyz(i, j, k, grid, &x, &y, &z);
  const PetscReal DL = domain_length(grid);
  const PetscReal DW = domain_width(grid);
  const PetscReal DH = domain_height(grid);
  return (PetscAbsReal(x - 0.50 * DL) <= 0.105 * DL / 2.0 &&
          PetscAbsReal(y - 0.50 * DW) <= 0.105 * DW / 2.0 &&
          z >= 0.25 * DH && z <= 0.75 * DH)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

// 向 6x6 wrench 分配矩阵中累加一个节点的力/力矩贡献关系。
// 目标是求出一组节点力，使它们合力和合力矩满足指定 wrench。
// x/y/z 是节点相对参考点的坐标，A 是正规方程矩阵。
void add_wrench_matrix(PetscReal x, PetscReal y, PetscReal z,
                       const PetscReal center[3],
                       PetscReal S[6][6]) {
  const PetscReal rx = x - center[0];
  const PetscReal ry = y - center[1];
  const PetscReal rz = z - center[2];
  const PetscReal C[6][3] = {
      {1.0, 0.0, 0.0},
      {0.0, 1.0, 0.0},
      {0.0, 0.0, 1.0},
      {0.0, -rz, ry},
      {rz, 0.0, -rx},
      {-ry, rx, 0.0}};
  for (PetscInt a = 0; a < 6; ++a) {
    for (PetscInt b = 0; b < 6; ++b) {
      for (PetscInt c = 0; c < 3; ++c) S[a][b] += C[a][c] * C[b][c];
    }
  }
}

// 用简单高斯消元求解 6x6 线性系统 A x = b。
// 这里系统规模固定且很小，用手写求解器避免引入额外 LAPACK 调用。
// 主要用于把目标合力/力矩分配成节点力的中间计算。
void solve_6x6(PetscReal A[6][6], const PetscReal b[6], PetscReal x[6]) {
  PetscReal aug[6][7] = {};
  for (PetscInt i = 0; i < 6; ++i) {
    for (PetscInt j = 0; j < 6; ++j) aug[i][j] = A[i][j];
    aug[i][i] += 1.0e-12;
    aug[i][6] = b[i];
  }
  for (PetscInt col = 0; col < 6; ++col) {
    PetscInt pivot = col;
    for (PetscInt row = col + 1; row < 6; ++row) {
      if (PetscAbsReal(aug[row][col]) > PetscAbsReal(aug[pivot][col])) pivot = row;
    }
    if (pivot != col) {
      for (PetscInt j = col; j < 7; ++j) std::swap(aug[col][j], aug[pivot][j]);
    }
    const PetscReal diag = aug[col][col];
    if (PetscAbsReal(diag) < 1.0e-30) {
      x[col] = 0.0;
      continue;
    }
    for (PetscInt j = col; j < 7; ++j) aug[col][j] /= diag;
    for (PetscInt row = 0; row < 6; ++row) {
      if (row == col) continue;
      const PetscReal f = aug[row][col];
      for (PetscInt j = col; j < 7; ++j) aug[row][j] -= f * aug[col][j];
    }
  }
  for (PetscInt i = 0; i < 6; ++i) x[i] = aug[i][6];
}

// 根据已求得的 wrench 分配系数，计算某个节点上应施加的三维力。
// 该力场在所有被选节点上累加后，会匹配目标合力和目标力矩。
// x/y/z 是节点相对载荷参考点的位置。
void wrench_force_at_node(PetscReal x, PetscReal y, PetscReal z,
                          const PetscReal center[3],
                          const PetscReal q[6],
                          PetscReal f[3]) {
  const PetscReal rx = x - center[0];
  const PetscReal ry = y - center[1];
  const PetscReal rz = z - center[2];
  f[0] = q[0] + rz * q[4] - ry * q[5];
  f[1] = q[1] - rz * q[3] + rx * q[5];
  f[2] = q[2] + ry * q[3] - rx * q[4];
}

// 为控制臂孔/弹簧节点构造满足目标合力和力矩的分布式载荷向量。
// 函数先收集被选节点的几何矩阵，再解 6x6 系统得到节点力分配系数。
// 输出 q 是与 uda 匹配的全局力向量，可直接作为 KSP 右端项。
PetscErrorCode control_arm_wrench_q(DM da,
                                    const Grid &grid,
                                    PetscInt which,
                                    const PetscReal wrench[6],
                                    PetscReal q[6],
                                    PetscInt *global_count) {
  PetscScalar ****dummy = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_s[36] = {};
  PetscReal global_s[36] = {};
  PetscInt local_count = 0;
  PetscReal center[3] = {0.18 * domain_length(grid),
                         which == 1 ? domain_width(grid) : 0.0,
                         0.50 * domain_height(grid)};

  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (!select_hole_node(i, j, k, grid, which)) continue;
        PetscReal x = 0.0, y = 0.0, z = 0.0;
        PetscReal S[6][6] = {};
        node_xyz(i, j, k, grid, &x, &y, &z);
        add_wrench_matrix(x, y, z, center, S);
        for (PetscInt a = 0; a < 6; ++a) {
          for (PetscInt b = 0; b < 6; ++b) local_s[6 * a + b] += S[a][b];
        }
        ++local_count;
      }
    }
  }
  PetscCallMPI(MPI_Allreduce(local_s, global_s, 36, MPIU_REAL, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_count, global_count, 1, MPIU_INT, MPI_SUM,
                             PETSC_COMM_WORLD));
  (void)dummy;
  PetscReal Sg[6][6] = {};
  for (PetscInt a = 0; a < 6; ++a) {
    for (PetscInt b = 0; b < 6; ++b) Sg[a][b] = global_s[6 * a + b];
  }
  solve_6x6(Sg, wrench, q);
  return 0;
}

// 根据控制臂工况编号填充全局载荷向量。
// 不同 load_case 会选择不同加载区域和力/力矩组合，用于多工况优化。
// 该函数封装了控制臂专用的载荷生成逻辑。
PetscErrorCode fill_control_arm_load(DM da,
                                     const Grid &grid,
                                     const EmSfemAnnOptions &options,
                                     PetscReal load_scale,
                                     Vec b) {
  // Keep the dominant vertical/spring case, but avoid letting one load case
  // completely drive the topology into a single force path.
  static const PetscReal weights[3] = {0.25, 0.50, 0.25};
  static const PetscReal p1_f[3][3] = {
      {91.0, 3597.0, 10230.0},
      {-814.0, 2716.0, 22726.0},
      {276.0, 5569.0, 16943.0}};
  static const PetscReal p1_t[3][3] = {
      {-6309.0, -37360.0, 13369.0},
      {-29734.0, 16033.0, -3062.0},
      {-14823.0, 58723.0, -19636.0}};
  static const PetscReal p2_f[3][3] = {
      {2534.0, 10162.0, -8288.0},
      {-1790.0, 3503.0, -16398.0},
      {-3488.0, -4632.0, -7851.0}};
  // Bushing/load-introduction moment at the B ring.  The previous pure-force
  // B load allowed the optimizer to keep only a very narrow resultant-force
  // path.  These moderate moments represent braking/cornering torsion and
  // force the B ring to introduce load over a broader local region.
  static const PetscReal p2_t[3][3] = {
      {4200.0, -6200.0, 3600.0},
      {-7600.0, 9200.0, -5200.0},
      {5400.0, 6800.0, 6400.0}};
  static const PetscReal spring_mag[3] = {8000.0, 16000.0, 10000.0};

  PetscScalar ****bg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscInt cases[3] = {0, 1, 2};
  PetscInt ncases = 3;
  if (options.load_case >= 1 && options.load_case <= 3) {
    cases[0] = options.load_case - 1;
    ncases = 1;
  }

  PetscCall(VecSet(b, 0.0));
  PetscCall(DMDAVecGetArrayDOF(da, b, &bg));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt case_it = 0; case_it < ncases; ++case_it) {
    const PetscInt cidx = cases[case_it];
    const PetscReal case_weight = ncases == 1 ? 1.0 : weights[cidx];
    PetscReal wrench1[6] = {p1_f[cidx][0], p1_f[cidx][1], p1_f[cidx][2],
                            p1_t[cidx][0], p1_t[cidx][1], p1_t[cidx][2]};
    PetscReal wrench2[6] = {p2_f[cidx][0], p2_f[cidx][1], p2_f[cidx][2],
                            p2_t[cidx][0], p2_t[cidx][1], p2_t[cidx][2]};
    PetscReal q1[6] = {};
    PetscReal q2[6] = {};
    PetscInt n1 = 0, n2 = 0;
    PetscCall(control_arm_wrench_q(da, grid, 1, wrench1, q1, &n1));
    PetscCall(control_arm_wrench_q(da, grid, 2, wrench2, q2, &n2));
    PetscCheck(n1 > 0 && n2 > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "Control-arm load rings are empty. Increase grid resolution or adjust BC geometry.");

    PetscInt local_spring = 0, global_spring = 0;
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (select_spring_node(i, j, k, grid)) ++local_spring;
        }
      }
    }
    PetscCallMPI(MPI_Allreduce(&local_spring, &global_spring, 1, MPIU_INT,
                               MPI_SUM, PETSC_COMM_WORLD));

    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (control_arm_fixed_node(i, j, k, grid)) continue;
          PetscReal x = 0.0, y = 0.0, z = 0.0;
          node_xyz(i, j, k, grid, &x, &y, &z);
          if (select_hole_node(i, j, k, grid, 1)) {
            PetscReal center[3] = {0.18 * domain_length(grid), domain_width(grid),
                                   0.50 * domain_height(grid)};
            PetscReal f[3] = {};
            wrench_force_at_node(x, y, z, center, q1, f);
            for (PetscInt d = 0; d < 3; ++d) {
              bg[k][j][i][d] += load_scale * case_weight * f[d];
            }
          }
          if (select_hole_node(i, j, k, grid, 2)) {
            PetscReal center[3] = {0.18 * domain_length(grid), 0.0,
                                   0.50 * domain_height(grid)};
            PetscReal f[3] = {};
            wrench_force_at_node(x, y, z, center, q2, f);
            for (PetscInt d = 0; d < 3; ++d) {
              bg[k][j][i][d] += load_scale * case_weight * f[d];
            }
          }
          if (options.include_spring_load && global_spring > 0 &&
              select_spring_node(i, j, k, grid)) {
            bg[k][j][i][2] += load_scale * case_weight *
                              (-spring_mag[cidx]) /
                              static_cast<PetscReal>(global_spring);
          }
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOF(da, b, &bg));
  return 0;
}

// 为简单悬臂梁/测试几何填充端部载荷。
// 通常在自由端某个节点或端面施加竖向力，用于小规模验证 matrix-free 和 KSP 流程。
// 控制臂生产算例一般不会走这个简单载荷路径。
PetscErrorCode fill_simple_tip_load(DM da,
                                    const Grid &grid,
                                    const char *benchmark_case,
                                    PetscReal load,
                                    Vec b) {
  PetscScalar ****bg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_torsion_denom = 0.0;
  PetscReal global_torsion_denom = 0.0;
  const PetscBool torsion = benchmark_is_torsion(benchmark_case);

  PetscCheck(torsion || std::strcmp(benchmark_case, "cantilever") == 0,
             PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "-benchmark_case must be cantilever or torsion");
  PetscCall(VecSet(b, 0.0));
  PetscCall(DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm));
  if (torsion) {
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (i != grid.nx - 1) continue;
          PetscReal x = 0.0, y = 0.0, z = 0.0;
          node_xyz(i, j, k, grid, &x, &y, &z);
          const PetscReal yc = y - 0.5 * domain_width(grid);
          const PetscReal zc = z - 0.5 * domain_height(grid);
          local_torsion_denom += yc * yc + zc * zc;
        }
      }
    }
    PetscCallMPI(MPI_Allreduce(&local_torsion_denom, &global_torsion_denom,
                               1, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
    PetscCheck(global_torsion_denom > PETSC_SMALL, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_WRONG,
               "Torsion benchmark needs a non-degenerate free-end face");
  }

  PetscCall(DMDAVecGetArrayDOF(da, b, &bg));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        if (i == grid.nx - 1) {
          if (torsion) {
            PetscReal x = 0.0, y = 0.0, z = 0.0;
            node_xyz(i, j, k, grid, &x, &y, &z);
            const PetscReal yc = y - 0.5 * domain_width(grid);
            const PetscReal zc = z - 0.5 * domain_height(grid);
            // 右端面切向力形成绕 x 轴的扭矩，用于无 mask 矩形域扭转梁。
            bg[k][j][i][1] += -load * zc / global_torsion_denom;
            bg[k][j][i][2] += load * yc / global_torsion_denom;
          } else {
            // 经典 3D 悬臂梁：右端面均布 -Z 方向载荷。
            bg[k][j][i][2] += -load /
                               static_cast<PetscReal>(grid.ny * grid.nz);
          }
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(da, b, &bg));
  return 0;
}

// EMsFEM ANN 的 matrix-free 矩阵向量乘法 y = A*x。
// PETSc KSP 每次需要 MatMult 时都会回调这个函数；它遍历本 rank 相关粗单元，
// 用缓存好的 24x24 ke 乘局部位移，再把单元贡献装配到 y。
PetscErrorCode emsfem_mult(Mat A, Vec x, Vec y) {
  EmSfemAnnContext *ctx = nullptr;
  PetscScalar ****xg = nullptr;
  PetscScalar ****yg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal temp_ke[kElementDofs * kElementDofs] = {};

  PetscCall(MatShellGetContext(A, &ctx));
  const Grid &g = ctx->grid;
  PetscCall(DMGlobalToLocalBegin(ctx->da, x, INSERT_VALUES, ctx->local_x));
  PetscCall(DMGlobalToLocalEnd(ctx->da, x, INSERT_VALUES, ctx->local_x));
  PetscCall(DMDAVecGetArrayDOFRead(ctx->da, ctx->local_x, &xg));
  PetscCall(DMDAVecGetArrayDOF(ctx->da, y, &yg));
  PetscCall(DMDAGetCorners(ctx->da, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        for (PetscInt c = 0; c < 3; ++c) {
          if (is_fixed_node(i, j, k, *ctx)) {
            yg[k][j][i][c] = xg[k][j][i][c];
            continue;
          }
          PetscScalar value = 0.0;
          const PetscInt ex0 = PetscMax(0, i - 1);
          const PetscInt ex1 = PetscMin(i, g.nx - 2);
          const PetscInt ey0 = PetscMax(0, j - 1);
          const PetscInt ey1 = PetscMin(j, g.ny - 2);
          const PetscInt ez0 = PetscMax(0, k - 1);
          const PetscInt ez1 = PetscMin(k, g.nz - 2);
          for (PetscInt ez = ez0; ez <= ez1; ++ez) {
            for (PetscInt ey = ey0; ey <= ey1; ++ey) {
              for (PetscInt ex = ex0; ex <= ex1; ++ex) {
                PetscReal *ke = cached_element_matrix(ctx, ex, ey, ez);
                if (!ke) {
                  PetscCall(compute_ems_element_matrix(ctx, ex, ey, ez, temp_ke));
                  ke = temp_ke;
                }
                const PetscInt row_node = h8_local_node(i - ex, j - ey, k - ez);
                const PetscInt row = 3 * row_node + c;
                for (PetscInt col_node = 0; col_node < 8; ++col_node) {
                  PetscInt ni = 0, nj = 0, nk = 0;
                  h8_node_coords(ex, ey, ez, col_node, &ni, &nj, &nk);
                  if (is_fixed_node(ni, nj, nk, *ctx)) continue;
                  for (PetscInt d = 0; d < 3; ++d) {
                    const PetscInt col = 3 * col_node + d;
                    value += ke[kElementDofs * row + col] * xg[nk][nj][ni][d];
                  }
                }
              }
            }
          }
          yg[k][j][i][c] = value;
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(ctx->da, ctx->local_x, &xg));
  PetscCall(DMDAVecRestoreArrayDOF(ctx->da, y, &yg));
  return 0;
}

// 近似提取 matrix-free EMsFEM 刚度矩阵的对角线。
// 对角线常用于 Jacobi 或 block-Jacobi 预条件，也可帮助 PETSc 估计缩放信息。
// 由于 A 没有显式存储，这里需要遍历缓存单元 ke 并累加本地节点的对角项。
PetscErrorCode emsfem_get_diagonal(Mat A, Vec diag_vec) {
  EmSfemAnnContext *ctx = nullptr;
  PetscScalar ****diag = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal temp_ke[kElementDofs * kElementDofs] = {};

  PetscCall(MatShellGetContext(A, &ctx));
  const Grid &g = ctx->grid;
  PetscCall(DMDAVecGetArrayDOF(ctx->da, diag_vec, &diag));
  PetscCall(DMDAGetCorners(ctx->da, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        for (PetscInt c = 0; c < 3; ++c) {
          if (is_fixed_node(i, j, k, *ctx)) {
            diag[k][j][i][c] = 1.0;
            continue;
          }
          PetscReal value = 0.0;
          const PetscInt ex0 = PetscMax(0, i - 1);
          const PetscInt ex1 = PetscMin(i, g.nx - 2);
          const PetscInt ey0 = PetscMax(0, j - 1);
          const PetscInt ey1 = PetscMin(j, g.ny - 2);
          const PetscInt ez0 = PetscMax(0, k - 1);
          const PetscInt ez1 = PetscMin(k, g.nz - 2);
          for (PetscInt ez = ez0; ez <= ez1; ++ez) {
            for (PetscInt ey = ey0; ey <= ey1; ++ey) {
              for (PetscInt ex = ex0; ex <= ex1; ++ex) {
                PetscReal *ke = cached_element_matrix(ctx, ex, ey, ez);
                if (!ke) {
                  PetscCall(compute_ems_element_matrix(ctx, ex, ey, ez, temp_ke));
                  ke = temp_ke;
                }
                const PetscInt node = h8_local_node(i - ex, j - ey, k - ez);
                const PetscInt row = 3 * node + c;
                value += ke[kElementDofs * row + row];
              }
            }
          }
          diag[k][j][i][c] = value;
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(ctx->da, diag_vec, &diag));
  return 0;
}

// 求一个 3x3 小矩阵的逆矩阵，并在病态时加入保护。
// 每个粗节点有 ux/uy/uz 三个自由度，block-Jacobi 需要反转对应的 3x3 对角块。
// 若块接近奇异，函数会使用安全的对角近似，避免预条件器产生 NaN。
void invert_3x3_block(const PetscReal a_in[9], PetscReal inv[9]) {
  PetscReal a[9] = {};
  PetscReal scale = 0.0;
  for (PetscInt i = 0; i < 3; ++i) {
    for (PetscInt j = 0; j < 3; ++j) {
      a[3 * i + j] = 0.5 * (a_in[3 * i + j] + a_in[3 * j + i]);
      scale = PetscMax(scale, PetscAbsReal(a[3 * i + j]));
    }
  }
  scale = PetscMax(scale, 1.0);
  const PetscReal shift = 1.0e-10 * scale;
  a[0] += shift;
  a[4] += shift;
  a[8] += shift;

  const PetscReal det =
      a[0] * (a[4] * a[8] - a[5] * a[7]) -
      a[1] * (a[3] * a[8] - a[5] * a[6]) +
      a[2] * (a[3] * a[7] - a[4] * a[6]);
  if (!std::isfinite(det) || PetscAbsReal(det) < 1.0e-24 * scale * scale * scale) {
    for (PetscInt i = 0; i < 9; ++i) inv[i] = 0.0;
    inv[0] = 1.0 / PetscMax(PetscAbsReal(a[0]), shift);
    inv[4] = 1.0 / PetscMax(PetscAbsReal(a[4]), shift);
    inv[8] = 1.0 / PetscMax(PetscAbsReal(a[8]), shift);
    return;
  }

  const PetscReal idet = 1.0 / det;
  inv[0] = (a[4] * a[8] - a[5] * a[7]) * idet;
  inv[1] = (a[2] * a[7] - a[1] * a[8]) * idet;
  inv[2] = (a[1] * a[5] - a[2] * a[4]) * idet;
  inv[3] = (a[5] * a[6] - a[3] * a[8]) * idet;
  inv[4] = (a[0] * a[8] - a[2] * a[6]) * idet;
  inv[5] = (a[2] * a[3] - a[0] * a[5]) * idet;
  inv[6] = (a[3] * a[7] - a[4] * a[6]) * idet;
  inv[7] = (a[1] * a[6] - a[0] * a[7]) * idet;
  inv[8] = (a[0] * a[4] - a[1] * a[3]) * idet;
}

// 构造 EMsFEM 自定义 block-Jacobi 预条件器的数据。
// 它从 matrix-free A 的对角块中为每个本地节点提取 3x3 块并求逆。
// 后续 PCApply 时只需要把每个节点的残差乘以这个小块逆矩阵。
PetscErrorCode setup_ems_block_jacobi(EmsBlockJacobiContext *pc_ctx) {
  PetscCheck(pc_ctx && pc_ctx->ems, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Missing EMsFEM block-Jacobi context");
  EmSfemAnnContext *ctx = pc_ctx->ems;
  const Grid &g = ctx->grid;
  PetscReal temp_ke[kElementDofs * kElementDofs] = {};

  PetscCall(DMDAGetCorners(ctx->da, &pc_ctx->xs, &pc_ctx->ys, &pc_ctx->zs,
                           &pc_ctx->xm, &pc_ctx->ym, &pc_ctx->zm));
  pc_ctx->inverse_blocks.assign(
      static_cast<std::size_t>(pc_ctx->xm) *
          static_cast<std::size_t>(pc_ctx->ym) *
          static_cast<std::size_t>(pc_ctx->zm) * 9,
      0.0);

  for (PetscInt k = pc_ctx->zs; k < pc_ctx->zs + pc_ctx->zm; ++k) {
    for (PetscInt j = pc_ctx->ys; j < pc_ctx->ys + pc_ctx->ym; ++j) {
      for (PetscInt i = pc_ctx->xs; i < pc_ctx->xs + pc_ctx->xm; ++i) {
        const std::size_t block_id =
            static_cast<std::size_t>(i - pc_ctx->xs) +
            static_cast<std::size_t>(pc_ctx->xm) *
                (static_cast<std::size_t>(j - pc_ctx->ys) +
                 static_cast<std::size_t>(pc_ctx->ym) *
                     static_cast<std::size_t>(k - pc_ctx->zs));
        PetscReal block[9] = {};
        PetscReal *inv = pc_ctx->inverse_blocks.data() + 9 * block_id;
        if (is_fixed_node(i, j, k, *ctx)) {
          inv[0] = 1.0;
          inv[4] = 1.0;
          inv[8] = 1.0;
          continue;
        }

        const PetscInt ex0 = PetscMax(0, i - 1);
        const PetscInt ex1 = PetscMin(i, g.nx - 2);
        const PetscInt ey0 = PetscMax(0, j - 1);
        const PetscInt ey1 = PetscMin(j, g.ny - 2);
        const PetscInt ez0 = PetscMax(0, k - 1);
        const PetscInt ez1 = PetscMin(k, g.nz - 2);
        for (PetscInt ez = ez0; ez <= ez1; ++ez) {
          for (PetscInt ey = ey0; ey <= ey1; ++ey) {
            for (PetscInt ex = ex0; ex <= ex1; ++ex) {
              PetscReal *ke = cached_element_matrix(ctx, ex, ey, ez);
              if (!ke) {
                PetscCall(compute_ems_element_matrix(ctx, ex, ey, ez, temp_ke));
                ke = temp_ke;
              }
              const PetscInt node = h8_local_node(i - ex, j - ey, k - ez);
              for (PetscInt c = 0; c < 3; ++c) {
                for (PetscInt d = 0; d < 3; ++d) {
                  block[3 * c + d] +=
                      ke[kElementDofs * (3 * node + c) + (3 * node + d)];
                }
              }
            }
          }
        }
        invert_3x3_block(block, inv);
      }
    }
  }
  return 0;
}

// 应用自定义 block-Jacobi 预条件器 y = B^{-1} x。
// 每个节点的三个位移分量作为一个 3x3 小块一起处理，比普通 Jacobi 更适合弹性问题。
// 该函数会被 PETSc PCShell 在 KSP 迭代中反复调用。
PetscErrorCode ems_block_jacobi_apply(PC pc, Vec x, Vec y) {
  EmsBlockJacobiContext *pc_ctx = nullptr;
  PetscScalar ****xg = nullptr;
  PetscScalar ****yg = nullptr;
  PetscCall(PCShellGetContext(pc, &pc_ctx));
  PetscCheck(pc_ctx && pc_ctx->ems, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Missing EMsFEM block-Jacobi context");

  PetscCall(DMDAVecGetArrayDOFRead(pc_ctx->ems->da, x, &xg));
  PetscCall(DMDAVecGetArrayDOF(pc_ctx->ems->da, y, &yg));
  for (PetscInt k = pc_ctx->zs; k < pc_ctx->zs + pc_ctx->zm; ++k) {
    for (PetscInt j = pc_ctx->ys; j < pc_ctx->ys + pc_ctx->ym; ++j) {
      for (PetscInt i = pc_ctx->xs; i < pc_ctx->xs + pc_ctx->xm; ++i) {
        const std::size_t block_id =
            static_cast<std::size_t>(i - pc_ctx->xs) +
            static_cast<std::size_t>(pc_ctx->xm) *
                (static_cast<std::size_t>(j - pc_ctx->ys) +
                 static_cast<std::size_t>(pc_ctx->ym) *
                     static_cast<std::size_t>(k - pc_ctx->zs));
        const PetscReal *inv = pc_ctx->inverse_blocks.data() + 9 * block_id;
        for (PetscInt c = 0; c < 3; ++c) {
          yg[k][j][i][c] = inv[3 * c + 0] * xg[k][j][i][0] +
                           inv[3 * c + 1] * xg[k][j][i][1] +
                           inv[3 * c + 2] * xg[k][j][i][2];
        }
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFRead(pc_ctx->ems->da, x, &xg));
  PetscCall(DMDAVecRestoreArrayDOF(pc_ctx->ems->da, y, &yg));
  return 0;
}

// 创建与 uda 分区对齐的粗单元密度 DMDA cda。
// cda 的网格尺寸是粗单元数 (nx-1, ny-1, nz-1)，每个单元一个平均密度。
// 低阶辅助矩阵会用 cda 上的粗密度来近似真实 EMsFEM 刚度。
PetscErrorCode create_ems_coarse_density_dm(DM uda, const Grid &grid, DM *cda) {
  PetscInt px = 1, py = 1, pz = 1;
  std::vector<PetscInt> ex_lens, ey_lens, ez_lens;
  std::vector<PetscInt> fx_lens, fy_lens, fz_lens;
  PetscCall(ems_get_aligned_ownership(uda, grid, 1, &px, &py, &pz,
                                      &ex_lens, &ey_lens, &ez_lens,
                                      &fx_lens, &fy_lens, &fz_lens));
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DMDA_STENCIL_BOX, grid.nx - 1, grid.ny - 1,
                         grid.nz - 1, px, py, pz, 1, 2,
                         ex_lens.data(), ey_lens.data(), ez_lens.data(), cda));
  PetscCall(DMSetUp(*cda));
  return 0;
}

// 从细密度 fine_rho 平均得到每个粗单元的粗密度 coarse_rho。
// 对每个粗单元，函数读取其内部 sub_n^3 个细胞密度并取平均。
// 结果用于 aux_laplacian/aux_elastic 等低阶辅助预条件矩阵。
PetscErrorCode build_ems_coarse_density(DM cda, DM fda, Vec fine_rho,
                                        const Grid &grid, PetscInt sub_n,
                                        Vec coarse_rho) {
  Vec local_rho = nullptr;
  PetscScalar ***rho = nullptr;
  PetscScalar ***coarse = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const PetscScalar inv_cells =
      1.0 / static_cast<PetscScalar>(sub_n * sub_n * sub_n);

  PetscCall(DMCreateLocalVector(fda, &local_rho));
  PetscCall(DMGlobalToLocalBegin(fda, fine_rho, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(fda, fine_rho, INSERT_VALUES, local_rho));
  PetscCall(DMDAVecGetArrayRead(fda, local_rho, &rho));
  PetscCall(DMDAVecGetArray(cda, coarse_rho, &coarse));
  PetscCall(DMDAGetCorners(cda, &xs, &ys, &zs, &xm, &ym, &zm));
  ElementCacheBox box;
  box.xs = xs;
  box.ys = ys;
  box.zs = zs;
  box.xm = xm;
  box.ym = ym;
  box.zm = zm;
  PetscCall(check_ems_fda_ghost_covers_element_box(
      fda, box, sub_n, "EMsFEM coarse-density build"));

  for (PetscInt ez = zs; ez < zs + zm; ++ez) {
    for (PetscInt ey = ys; ey < ys + ym; ++ey) {
      for (PetscInt ex = xs; ex < xs + xm; ++ex) {
        PetscScalar sum = 0.0;
        for (PetscInt fz = 0; fz < sub_n; ++fz) {
          const PetscInt k = ez * sub_n + fz;
          for (PetscInt fy = 0; fy < sub_n; ++fy) {
            const PetscInt j = ey * sub_n + fy;
            for (PetscInt fx = 0; fx < sub_n; ++fx) {
              const PetscInt i = ex * sub_n + fx;
              sum += rho[k][j][i];
            }
          }
        }
        coarse[ez][ey][ex] = sum * inv_cells;
      }
    }
  }

  PetscCall(DMDAVecRestoreArray(cda, coarse_rho, &coarse));
  PetscCall(DMDAVecRestoreArrayRead(fda, local_rho, &rho));
  PetscCall(VecDestroy(&local_rho));
  (void)grid;
  return 0;
}

// 计算某个粗节点周围相邻粗单元密度的平均值。
// 辅助矩阵装配节点/边刚度时需要一个局部材料比例，这里用周围单元平均来平滑近似。
// 函数会自动跳过越界单元。
PetscReal ems_average_adjacent_coarse_density(PetscScalar ***rho,
                                              const Grid &grid,
                                              PetscInt i, PetscInt j,
                                              PetscInt k) {
  PetscReal sum = 0.0;
  PetscInt count = 0;
  const PetscInt ex0 = PetscMax(0, i - 1);
  const PetscInt ex1 = PetscMin(i, grid.nx - 2);
  const PetscInt ey0 = PetscMax(0, j - 1);
  const PetscInt ey1 = PetscMin(j, grid.ny - 2);
  const PetscInt ez0 = PetscMax(0, k - 1);
  const PetscInt ez1 = PetscMin(k, grid.nz - 2);
  for (PetscInt ez = ez0; ez <= ez1; ++ez) {
    for (PetscInt ey = ey0; ey <= ey1; ++ey) {
      for (PetscInt ex = ex0; ex <= ex1; ++ex) {
        sum += PetscRealPart(rho[ez][ey][ex]);
        ++count;
      }
    }
  }
  return count > 0 ? sum / static_cast<PetscReal>(count) : 0.0;
}

// 装配标量 Laplacian 风格的辅助预条件矩阵。
// 该矩阵比真实弹性矩阵简单，主要捕捉密度导致的局部刚度分布，用于较轻量的预条件。
// 它不是物理精确刚度矩阵，而是给 KSP 提供更容易构造的 P 矩阵。
PetscErrorCode assemble_ems_aux_laplacian_matrix(DM uda, DM cda,
                                                Vec coarse_rho,
                                                const EmSfemAnnContext &ctx,
                                                Mat P) {
  Vec local_rho = nullptr;
  PetscScalar ***rg = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const Grid &grid = ctx.grid;
  const PetscInt di[6] = {-1, 1, 0, 0, 0, 0};
  const PetscInt dj[6] = {0, 0, -1, 1, 0, 0};
  const PetscInt dk[6] = {0, 0, 0, 0, -1, 1};
  const PetscReal hx =
      domain_length(grid) / static_cast<PetscReal>(PetscMax(1, grid.nx - 1));
  const PetscReal hy =
      domain_width(grid) / static_cast<PetscReal>(PetscMax(1, grid.ny - 1));
  const PetscReal hz =
      domain_height(grid) / static_cast<PetscReal>(PetscMax(1, grid.nz - 1));
  const PetscReal wx = hy * hz / PetscMax(hx, PETSC_SMALL);
  const PetscReal wy = hx * hz / PetscMax(hy, PETSC_SMALL);
  const PetscReal wz = hx * hy / PetscMax(hz, PETSC_SMALL);
  const PetscReal weight[6] = {wx, wx, wy, wy, wz, wz};

  PetscCall(MatZeroEntries(P));
  PetscCall(DMCreateLocalVector(cda, &local_rho));
  PetscCall(DMGlobalToLocalBegin(cda, coarse_rho, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(cda, coarse_rho, INSERT_VALUES, local_rho));
  PetscCall(DMDAVecGetArrayRead(cda, local_rho, &rg));
  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        const PetscReal rho0 =
            ems_average_adjacent_coarse_density(rg, grid, i, j, k);
        for (PetscInt c = 0; c < 3; ++c) {
          MatStencil row;
          MatStencil cols[7];
          PetscScalar vals[7];
          PetscInt ncols = 0;
          PetscReal diag = 0.0;
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = c;

          if (is_fixed_node(i, j, k, ctx)) {
            cols[0] = row;
            vals[0] = 1.0;
            PetscCall(MatSetValuesStencil(P, 1, &row, 1, cols, vals,
                                           ADD_VALUES));
            continue;
          }

          for (PetscInt q = 0; q < 6; ++q) {
            const PetscInt ii = i + di[q];
            const PetscInt jj = j + dj[q];
            const PetscInt kk = k + dk[q];
            if (ii < 0 || ii >= grid.nx || jj < 0 || jj >= grid.ny ||
                kk < 0 || kk >= grid.nz) {
              continue;
            }
            const PetscReal rho1 =
                ems_average_adjacent_coarse_density(rg, grid, ii, jj, kk);
            const PetscReal kij =
                weight[q] * edge_stiffness(rho0, rho1, ctx.density_options);
            diag += kij;
            if (!is_fixed_node(ii, jj, kk, ctx)) {
              cols[ncols].i = ii;
              cols[ncols].j = jj;
              cols[ncols].k = kk;
              cols[ncols].c = c;
              vals[ncols] = -kij;
              ++ncols;
            }
          }

          cols[ncols] = row;
          vals[ncols] = PetscMax(diag, PETSC_SMALL);
          ++ncols;
          PetscCall(MatSetValuesStencil(P, 1, &row, ncols, cols, vals,
                                         ADD_VALUES));
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(cda, local_rho, &rg));
  PetscCall(VecDestroy(&local_rho));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  return 0;
}

// 创建并装配 Laplacian 辅助矩阵。
// 先根据 uda 创建稀疏矩阵结构，再调用 assemble_ems_aux_laplacian_matrix 填值。
// 返回的 P 可交给 PETSc 的 GAMG/HYPRE 等预条件器。
PetscErrorCode create_ems_aux_laplacian_matrix(DM uda, DM cda,
                                               Vec coarse_rho,
                                               const EmSfemAnnContext &ctx,
                                               Mat *P) {
  PetscCall(DMCreateMatrix(uda, P));
  PetscCall(MatSetOption(*P, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(assemble_ems_aux_laplacian_matrix(uda, cda, coarse_rho, ctx, *P));
  return 0;
}

// 装配低阶弹性辅助矩阵。
// 它用粗单元平均密度和标准 H8 弹性刚度近似真实 EMsFEM ANN 刚度，
// 既保留弹性耦合结构，又比完整 ANN 等效矩阵更便宜。
PetscErrorCode assemble_ems_aux_elasticity_matrix(DM uda, DM cda,
                                                 Vec coarse_rho,
                                                 const EmSfemAnnContext &ctx,
                                                 Mat P) {
  Vec local_rho = nullptr;
  PetscScalar ***rg = nullptr;
  PetscReal ke[kElementDofs * kElementDofs] = {};
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  const Grid &grid = ctx.grid;
  const PetscReal hx =
      domain_length(grid) / static_cast<PetscReal>(PetscMax(1, grid.nx - 1));
  const PetscReal hy =
      domain_width(grid) / static_cast<PetscReal>(PetscMax(1, grid.ny - 1));
  const PetscReal hz =
      domain_height(grid) / static_cast<PetscReal>(PetscMax(1, grid.nz - 1));
  compute_h8_element_stiffness(hx, hy, hz,
                               ctx.density_options.young_modulus, ke);

  PetscCall(MatZeroEntries(P));
  PetscCall(DMCreateLocalVector(cda, &local_rho));
  PetscCall(DMGlobalToLocalBegin(cda, coarse_rho, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(cda, coarse_rho, INSERT_VALUES, local_rho));
  PetscCall(DMDAVecGetArrayRead(cda, local_rho, &rg));
  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        for (PetscInt c = 0; c < 3; ++c) {
          MatStencil row;
          MatStencil cols[8 * 8 * 3];
          PetscScalar vals[8 * 8 * 3];
          PetscInt ncols = 0;
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = c;

          if (is_fixed_node(i, j, k, ctx)) {
            cols[0] = row;
            vals[0] = 1.0;
            PetscCall(MatSetValuesStencil(P, 1, &row, 1, cols, vals,
                                           ADD_VALUES));
            continue;
          }

          const PetscInt ex0 = PetscMax(0, i - 1);
          const PetscInt ex1 = PetscMin(i, grid.nx - 2);
          const PetscInt ey0 = PetscMax(0, j - 1);
          const PetscInt ey1 = PetscMin(j, grid.ny - 2);
          const PetscInt ez0 = PetscMax(0, k - 1);
          const PetscInt ez1 = PetscMin(k, grid.nz - 2);
          for (PetscInt ez = ez0; ez <= ez1; ++ez) {
            for (PetscInt ey = ey0; ey <= ey1; ++ey) {
              for (PetscInt ex = ex0; ex <= ex1; ++ex) {
                const PetscInt row_node = h8_local_node(i - ex, j - ey, k - ez);
                const PetscInt erow = 3 * row_node + c;
                const PetscReal scale =
                    simp_scale(PetscRealPart(rg[ez][ey][ex]),
                               ctx.density_options);
                for (PetscInt col_node = 0; col_node < 8; ++col_node) {
                  PetscInt ni = 0, nj = 0, nk = 0;
                  h8_node_coords(ex, ey, ez, col_node, &ni, &nj, &nk);
                  if (is_fixed_node(ni, nj, nk, ctx)) continue;
                  for (PetscInt d = 0; d < 3; ++d) {
                    const PetscInt ecol = 3 * col_node + d;
                    cols[ncols].i = ni;
                    cols[ncols].j = nj;
                    cols[ncols].k = nk;
                    cols[ncols].c = d;
                    vals[ncols] = scale * ke[kElementDofs * erow + ecol];
                    ++ncols;
                  }
                }
              }
            }
          }

          PetscCall(MatSetValuesStencil(P, 1, &row, ncols, cols, vals,
                                         ADD_VALUES));
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayRead(cda, local_rho, &rg));
  PetscCall(VecDestroy(&local_rho));
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  return 0;
}

// 创建并装配低阶弹性辅助矩阵 P。
// 装配完成后会附加弹性近零空间，帮助 GAMG/HYPRE 更好识别刚体模态。
// 该矩阵常作为 aux_elastic_gamg/aux_elastic_hypre 的预条件输入。
PetscErrorCode create_ems_aux_elasticity_matrix(DM uda, DM cda,
                                                Vec coarse_rho,
                                                const EmSfemAnnContext &ctx,
                                                Mat *P) {
  PetscCall(DMCreateMatrix(uda, P));
  PetscCall(MatSetBlockSize(*P, 3));
  PetscCall(MatSetOption(*P, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(assemble_ems_aux_elasticity_matrix(uda, cda, coarse_rho, ctx, *P));
  PetscCall(attach_ems_elasticity_near_nullspace(uda, ctx, *P));
  return 0;
}

// 装配基于 ANN 等效单元刚度的辅助矩阵。
// 它把当前 K-cache 中的 EMsFEM 24x24 ke 显式装配成稀疏矩阵 P，
// 预条件质量通常更接近真实 A，但内存和装配成本也更高。
PetscErrorCode assemble_ems_aux_ann_matrix(DM uda, EmSfemAnnContext *ctx,
                                           Mat P) {
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal temp_ke[kElementDofs * kElementDofs] = {};
  const Grid &grid = ctx->grid;

  PetscCall(MatZeroEntries(P));
  PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));
  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        for (PetscInt c = 0; c < 3; ++c) {
          MatStencil row;
          MatStencil cols[8 * 8 * 3];
          PetscScalar vals[8 * 8 * 3];
          PetscInt ncols = 0;
          row.i = i;
          row.j = j;
          row.k = k;
          row.c = c;

          if (is_fixed_node(i, j, k, *ctx)) {
            cols[0] = row;
            vals[0] = 1.0;
            PetscCall(MatSetValuesStencil(P, 1, &row, 1, cols, vals,
                                           ADD_VALUES));
            continue;
          }

          const PetscInt ex0 = PetscMax(0, i - 1);
          const PetscInt ex1 = PetscMin(i, grid.nx - 2);
          const PetscInt ey0 = PetscMax(0, j - 1);
          const PetscInt ey1 = PetscMin(j, grid.ny - 2);
          const PetscInt ez0 = PetscMax(0, k - 1);
          const PetscInt ez1 = PetscMin(k, grid.nz - 2);
          for (PetscInt ez = ez0; ez <= ez1; ++ez) {
            for (PetscInt ey = ey0; ey <= ey1; ++ey) {
              for (PetscInt ex = ex0; ex <= ex1; ++ex) {
                PetscReal *ke = cached_element_matrix(ctx, ex, ey, ez);
                if (!ke) {
                  PetscCall(compute_ems_element_matrix(ctx, ex, ey, ez,
                                                       temp_ke));
                  ke = temp_ke;
                }
                const PetscInt row_node = h8_local_node(i - ex, j - ey, k - ez);
                const PetscInt erow = 3 * row_node + c;
                for (PetscInt col_node = 0; col_node < 8; ++col_node) {
                  PetscInt ni = 0, nj = 0, nk = 0;
                  h8_node_coords(ex, ey, ez, col_node, &ni, &nj, &nk);
                  if (is_fixed_node(ni, nj, nk, *ctx)) continue;
                  for (PetscInt d = 0; d < 3; ++d) {
                    const PetscInt ecol = 3 * col_node + d;
                    cols[ncols].i = ni;
                    cols[ncols].j = nj;
                    cols[ncols].k = nk;
                    cols[ncols].c = d;
                    vals[ncols] = ke[kElementDofs * erow + ecol];
                    ++ncols;
                  }
                }
              }
            }
          }

          PetscCall(MatSetValuesStencil(P, 1, &row, ncols, cols, vals,
                                         ADD_VALUES));
        }
      }
    }
  }

  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));
  return 0;
}

// 创建并装配 ANN 辅助矩阵 P。
// 该 P 使用当前设计下的 EMsFEM 等效刚度，适合交给 AMG 类预条件器。
// 当自由度很大时，这一步可能成为内存和 setup 时间的主要来源。
PetscErrorCode create_ems_aux_ann_matrix(DM uda, EmSfemAnnContext *ctx,
                                         Mat *P) {
  PetscCall(DMCreateMatrix(uda, P));
  PetscCall(MatSetBlockSize(*P, 3));
  PetscCall(MatSetOption(*P, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatSetOption(*P, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE));
  PetscCall(assemble_ems_aux_ann_matrix(uda, ctx, *P));
  PetscCall(attach_ems_elasticity_near_nullspace(uda, *ctx, *P));
  return 0;
}

// 判断给定 EMsFEM 预条件器类型是否需要显式辅助矩阵 P。
// 返回 true 的类型会在 KSPSetOperators(ksp, A, P) 中使用 P 供 PC 构造。
// block_jacobi 等 shell 预条件器不需要额外辅助矩阵。
PetscBool ems_pc_type_uses_aux_matrix(const char *ems_pc_type) {
  return (std::strcmp(ems_pc_type, "aux_gamg") == 0 ||
          std::strcmp(ems_pc_type, "aux_hypre") == 0 ||
          std::strcmp(ems_pc_type, "aux_elastic_gamg") == 0 ||
          std::strcmp(ems_pc_type, "aux_elastic_hypre") == 0 ||
          std::strcmp(ems_pc_type, "aux_ann_gamg") == 0 ||
          std::strcmp(ems_pc_type, "aux_ann_hypre") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

// 判断预条件器类型是否使用低阶辅助矩阵。
// 低阶辅助矩阵通常来自粗密度平均，而不是完整 ANN 等效单元刚度。
// 这类 P 内存更省，但对真实 A 的近似会弱一些。
PetscBool ems_pc_type_uses_low_order_aux_matrix(const char *ems_pc_type) {
  return (std::strcmp(ems_pc_type, "aux_gamg") == 0 ||
          std::strcmp(ems_pc_type, "aux_hypre") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

// 判断预条件器类型是否使用弹性辅助矩阵。
// 弹性辅助矩阵保留三维弹性耦合和刚体模态信息，适合 GAMG/HYPRE。
PetscBool ems_pc_type_uses_elastic_aux_matrix(const char *ems_pc_type) {
  return (std::strcmp(ems_pc_type, "aux_elastic_gamg") == 0 ||
          std::strcmp(ems_pc_type, "aux_elastic_hypre") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

// 判断预条件器类型是否需要先构造粗单元平均密度 cda/coarse_rho。
// aux_laplacian 和 aux_elastic 路径都需要粗密度；aux_ann 直接用 K-cache，不需要。
PetscBool ems_pc_type_uses_coarse_density_aux_matrix(const char *ems_pc_type) {
  return (ems_pc_type_uses_low_order_aux_matrix(ems_pc_type) ||
          ems_pc_type_uses_elastic_aux_matrix(ems_pc_type))
             ? PETSC_TRUE
             : PETSC_FALSE;
}

// 判断预条件器类型是否使用 ANN 等效刚度辅助矩阵。
// aux_ann_* 会把当前 EMsFEM 单元 ke 显式装配成 P，因此更接近真实 matrix-free A。
PetscBool ems_pc_type_uses_ann_aux_matrix(const char *ems_pc_type) {
  return (std::strcmp(ems_pc_type, "aux_ann_gamg") == 0 ||
          std::strcmp(ems_pc_type, "aux_ann_hypre") == 0)
             ? PETSC_TRUE
             : PETSC_FALSE;
}

// 读取命令行选项 -ems_pc_type，并返回该类型是否需要辅助矩阵。
// 如果用户没有指定，默认使用 aux_ann_gamg。
// ems_pc_type 是输出字符串缓冲区，uses_aux_matrix 是输出布尔标记。
PetscErrorCode get_ems_pc_type_option(char *ems_pc_type,
                                      size_t ems_pc_type_size,
                                      PetscBool *uses_aux_matrix) {
  PetscBool has_ems_pc_type = PETSC_FALSE;
  PetscCall(PetscStrncpy(ems_pc_type, "aux_ann_gamg", ems_pc_type_size));
  PetscCall(PetscOptionsGetString(nullptr, nullptr, "-ems_pc_type",
                                  ems_pc_type, ems_pc_type_size,
                                  &has_ems_pc_type));
  (void)has_ems_pc_type;
  *uses_aux_matrix = ems_pc_type_uses_aux_matrix(ems_pc_type);
  return 0;
}

// 根据 ems_pc_type 配置 KSP 的预条件器 PC。
// 这里会选择 GAMG/HYPRE、block-Jacobi 或无预条件，并设置对应的 PC 矩阵与参数。
// A 是真实 matrix-free 算子，P 是可选的显式辅助矩阵。
PetscErrorCode setup_ems_preconditioner(KSP ksp, Mat A, Mat P, DM uda, DM cda,
                                        Vec coarse_rho,
                                        EmsBlockJacobiContext *pc_ctx,
                                        EmSfemAnnContext *ctx,
                                        const char *ems_pc_type) {
  PetscCheck(ctx != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "Missing EMsFEM ANN context");
  if (ems_pc_type_uses_low_order_aux_matrix(ems_pc_type)) {
    PetscCheck(cda != nullptr && coarse_rho != nullptr, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_NULL,
               "Auxiliary EMsFEM preconditioner needs coarse density state");
    PetscCall(assemble_ems_aux_laplacian_matrix(uda, cda, coarse_rho, *ctx, P));
  } else if (ems_pc_type_uses_elastic_aux_matrix(ems_pc_type)) {
    PetscCheck(cda != nullptr && coarse_rho != nullptr, PETSC_COMM_WORLD,
               PETSC_ERR_ARG_NULL,
               "Auxiliary EMsFEM elasticity preconditioner needs coarse density state");
    PetscCall(assemble_ems_aux_elasticity_matrix(uda, cda, coarse_rho, *ctx, P));
  } else if (ems_pc_type_uses_ann_aux_matrix(ems_pc_type)) {
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
               "ANN auxiliary EMsFEM preconditioner needs an assembled matrix");
    PetscCall(assemble_ems_aux_ann_matrix(uda, ctx, P));
  } else if (std::strcmp(ems_pc_type, "block_jacobi") == 0) {
    PetscCall(setup_ems_block_jacobi(pc_ctx));
  }
  PetscCall(KSPSetReusePreconditioner(ksp, PETSC_FALSE));
  PetscCall(KSPSetOperators(ksp, A, P ? P : A));
  return 0;
}

// 配置 EMsFEM 线性求解器 KSP。
// 函数设置真实算子 A、可选预条件矩阵 P、默认 KSP 类型/容差以及对应 PC。
// 最后调用 KSPSetFromOptions，允许命令行覆盖求解器和预条件器参数。
PetscErrorCode configure_ems_ksp(KSP ksp, Mat A, Mat P,
                                 EmsBlockJacobiContext *pc_ctx,
                                 const Grid &grid,
                                 const OptimizerOptions &optimizer_options,
                                 const char *ems_pc_type) {
  PetscBool allow_large_block_jacobi = PETSC_FALSE;
  PetscInt large_block_jacobi_dof_limit = 2000000;
  PetscCall(PetscOptionsGetBool(nullptr, nullptr,
                                "-ems_allow_large_block_jacobi",
                                &allow_large_block_jacobi, nullptr));
  PetscCall(PetscOptionsGetInt(nullptr, nullptr,
                               "-ems_large_block_jacobi_dof_limit",
                               &large_block_jacobi_dof_limit, nullptr));
  if (std::strcmp(ems_pc_type, "block_jacobi") == 0) {
    const PetscInt ndof = dof_count(grid);
    PetscCheck(allow_large_block_jacobi ||
                   ndof < large_block_jacobi_dof_limit,
               PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "Refusing EMsFEM ANN block_jacobi on %lld DOF. Use -ems_pc_type aux_hypre or set -ems_allow_large_block_jacobi true for diagnostics.",
               static_cast<long long>(ndof));
  }

  PetscCall(KSPSetOperators(ksp, A, P ? P : A));
  PetscCall(KSPSetType(ksp, ems_pc_type_uses_aux_matrix(ems_pc_type)
                                ? KSPFGMRES
                                : KSPCG));
  PetscBool has_gmres_restart = PETSC_FALSE;
  PetscCall(PetscOptionsHasName(nullptr, nullptr, "-ksp_gmres_restart",
                                &has_gmres_restart));
  PetscCall(KSPSetTolerances(ksp, optimizer_options.ksp_rtol, PETSC_DEFAULT,
                             PETSC_DEFAULT, optimizer_options.ksp_max_it));
  PetscCall(KSPSetFromOptions(ksp));
  if (ems_pc_type_uses_aux_matrix(ems_pc_type) && !has_gmres_restart) {
    PetscCall(KSPGMRESSetRestart(ksp, 100));
  }

  if (std::strcmp(ems_pc_type, "block_jacobi") == 0) {
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCSHELL));
    PetscCall(PCShellSetName(pc, "ems_node_block_jacobi"));
    PetscCall(PCShellSetContext(pc, pc_ctx));
    PetscCall(PCShellSetApply(pc, ems_block_jacobi_apply));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN preconditioner: node 3x3 block Jacobi\n"));
  } else if (std::strcmp(ems_pc_type, "jacobi") == 0) {
    PC pc = nullptr;
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCJACOBI));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN preconditioner: scalar PETSc Jacobi\n"));
  } else if (std::strcmp(ems_pc_type, "petsc") == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN preconditioner: using PETSc -pc_type options on the shell operator\n"));
  } else if (std::strcmp(ems_pc_type, "aux_gamg") == 0) {
    PC pc = nullptr;
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_gamg needs an assembled auxiliary matrix");
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMG));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN preconditioner: auxiliary low-order matrix with PETSc GAMG\n"));
  } else if (std::strcmp(ems_pc_type, "aux_hypre") == 0) {
    PC pc = nullptr;
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_hypre needs an assembled auxiliary matrix");
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCHYPRE));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN preconditioner: auxiliary low-order matrix with PETSc/HYPRE\n"));
  } else if (std::strcmp(ems_pc_type, "aux_elastic_gamg") == 0) {
    PC pc = nullptr;
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_elastic_gamg needs an assembled auxiliary elasticity matrix");
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMG));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN preconditioner: auxiliary H8 elasticity matrix with PETSc GAMG\n"));
  } else if (std::strcmp(ems_pc_type, "aux_elastic_hypre") == 0) {
    PC pc = nullptr;
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_elastic_hypre needs an assembled auxiliary elasticity matrix");
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCHYPRE));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN preconditioner: auxiliary H8 elasticity matrix with PETSc/HYPRE\n"));
  } else if (std::strcmp(ems_pc_type, "aux_ann_gamg") == 0) {
    PC pc = nullptr;
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_ann_gamg needs an assembled ANN auxiliary matrix");
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCGAMG));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN preconditioner: assembled ANN matrix with PETSc GAMG\n"));
  } else if (std::strcmp(ems_pc_type, "aux_ann_hypre") == 0) {
    PC pc = nullptr;
    PetscCheck(P != nullptr, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "aux_ann_hypre needs an assembled ANN auxiliary matrix");
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCHYPRE));
    PetscCall(PCSetFromOptions(pc));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN preconditioner: assembled ANN matrix with PETSc/HYPRE\n"));
  } else {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
               "-ems_pc_type must be block_jacobi, jacobi, petsc, aux_gamg, aux_hypre, aux_elastic_gamg, aux_elastic_hypre, aux_ann_gamg, or aux_ann_hypre");
  }
  return 0;
}

// 销毁 MatShell 绑定的 EMsFEM 上下文。
// PETSc 销毁 shell matrix 时会调用该函数，释放 new 出来的 EmSfemAnnContext。
// 这里只负责 shell matrix 的上下文，不销毁外部单独管理的 Vec/DM。
PetscErrorCode emsfem_destroy(Mat A) {
  EmSfemAnnContext *ctx = nullptr;
  PetscCall(MatShellGetContext(A, &ctx));
  if (ctx) {
    PetscCall(VecDestroy(&ctx->local_x));
    PetscCall(DMDestroy(&ctx->da));
    delete ctx;
  }
  return 0;
}

// 计算细密度设计变量上的柔度灵敏度。
// 对每个粗单元，先提取粗单元位移 ue，再计算内部每个细胞能量并乘以 SIMP 导数。
// 输出 dc_phys 是物理密度上的梯度，后续会经过 Heaviside 和过滤器伴随反传。
PetscErrorCode compute_ems_fine_sensitivity(EmSfemAnnContext *ctx,
                                            DM uda,
                                            DM fda,
                                            Vec u,
                                            Vec rho,
                                            Vec mask,
                                            Vec dc,
                                            PetscReal *volume_fraction) {
  Vec local_u = nullptr;
  Vec local_rho = nullptr, local_mask = nullptr;
  PetscScalar ****ug = nullptr;
  PetscScalar ***r = nullptr, ***m = nullptr, ***d = nullptr;
  PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
  PetscReal local_rho_sum = 0.0, local_mask_sum = 0.0;
  PetscReal global_rho_sum = 0.0, global_mask_sum = 0.0;
  const PetscInt sub_n = ctx->ems_options.sub_n;

  PetscCall(DMCreateLocalVector(uda, &local_u));
  PetscCall(DMGlobalToLocalBegin(uda, u, INSERT_VALUES, local_u));
  PetscCall(DMGlobalToLocalEnd(uda, u, INSERT_VALUES, local_u));
  PetscCall(DMCreateLocalVector(fda, &local_rho));
  PetscCall(DMCreateLocalVector(fda, &local_mask));
  PetscCall(DMGlobalToLocalBegin(fda, rho, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalEnd(fda, rho, INSERT_VALUES, local_rho));
  PetscCall(DMGlobalToLocalBegin(fda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMGlobalToLocalEnd(fda, mask, INSERT_VALUES, local_mask));
  PetscCall(DMDAVecGetArrayDOFRead(uda, local_u, &ug));
  PetscCall(DMDAVecGetArrayRead(fda, local_rho, &r));
  PetscCall(DMDAVecGetArrayRead(fda, local_mask, &m));
  PetscCall(DMDAVecGetArray(fda, dc, &d));
  PetscCall(DMDAGetCorners(fda, &xs, &ys, &zs, &xm, &ym, &zm));

  for (PetscInt k = zs; k < zs + zm; ++k) {
    for (PetscInt j = ys; j < ys + ym; ++j) {
      for (PetscInt i = xs; i < xs + xm; ++i) {
        d[k][j][i] = 0.0;
        const PetscReal mask_value = PetscRealPart(m[k][j][i]);
        const PetscReal rho_value = PetscRealPart(r[k][j][i]);
        const PetscReal design = (mask_value > 0.5 && mask_value <= 1.5) ? 1.0 : 0.0;
        local_mask_sum += design;
        local_rho_sum += design * rho_value;
      }
    }
  }

  const PetscInt ex_start = PetscMax(0, xs / sub_n);
  const PetscInt ey_start = PetscMax(0, ys / sub_n);
  const PetscInt ez_start = PetscMax(0, zs / sub_n);
  const PetscInt ex_end =
      PetscMin(ctx->grid.nx - 2, (xs + xm - 1) / sub_n);
  const PetscInt ey_end =
      PetscMin(ctx->grid.ny - 2, (ys + ym - 1) / sub_n);
  const PetscInt ez_end =
      PetscMin(ctx->grid.nz - 2, (zs + zm - 1) / sub_n);
  if (ex_start <= ex_end && ey_start <= ey_end && ez_start <= ez_end) {
    ElementCacheBox box;
    box.xs = ex_start;
    box.ys = ey_start;
    box.zs = ez_start;
    box.xm = ex_end - ex_start + 1;
    box.ym = ey_end - ey_start + 1;
    box.zm = ez_end - ez_start + 1;
    PetscCall(check_ems_fda_ghost_covers_element_box(
        fda, box, sub_n, "EMsFEM sensitivity"));
    PetscCall(check_ems_uda_ghost_covers_element_box(
        uda, box, "EMsFEM sensitivity"));
  }

  for (PetscInt ez = ez_start; ez <= ez_end; ++ez) {
    for (PetscInt ey = ey_start; ey <= ey_end; ++ey) {
      for (PetscInt ex = ex_start; ex <= ex_end; ++ex) {
        PetscReal ue[kElementDofs] = {};
        std::vector<PetscReal> energies(
            static_cast<std::size_t>(sub_n * sub_n * sub_n), 0.0);
        std::vector<PetscReal> material;
        material.reserve(static_cast<std::size_t>(sub_n * sub_n * sub_n));

        for (PetscInt fz = 0; fz < sub_n; ++fz) {
          for (PetscInt fy = 0; fy < sub_n; ++fy) {
            for (PetscInt fx = 0; fx < sub_n; ++fx) {
              const PetscInt fi = ex * sub_n + fx;
              const PetscInt fj = ey * sub_n + fy;
              const PetscInt fk = ez * sub_n + fz;
              material.push_back(simp_scale(PetscRealPart(r[fk][fj][fi]),
                                            ctx->density_options));
            }
          }
        }

        for (PetscInt node = 0; node < 8; ++node) {
          PetscInt ni = 0, nj = 0, nk = 0;
          h8_node_coords(ex, ey, ez, node, &ni, &nj, &nk);
          for (PetscInt c = 0; c < 3; ++c) {
            ue[3 * node + c] = is_fixed_node(ni, nj, nk, *ctx)
                                   ? 0.0
                                   : PetscRealPart(ug[nk][nj][ni][c]);
          }
        }

        PetscCall(compute_fine_cell_energies(ctx, material, ue, energies.data()));
        for (PetscInt fz = 0; fz < sub_n; ++fz) {
          for (PetscInt fy = 0; fy < sub_n; ++fy) {
            for (PetscInt fx = 0; fx < sub_n; ++fx) {
              const PetscInt fi = ex * sub_n + fx;
              const PetscInt fj = ey * sub_n + fy;
              const PetscInt fk = ez * sub_n + fz;
              if (fi < xs || fi >= xs + xm || fj < ys || fj >= ys + ym ||
                  fk < zs || fk >= zs + zm) {
                continue;
              }
              const PetscReal mask_value = PetscRealPart(m[fk][fj][fi]);
              if (mask_value <= 0.5 || mask_value > 1.5) {
                d[fk][fj][fi] = 0.0;
                continue;
              }
              const PetscReal rho_value = PetscRealPart(r[fk][fj][fi]);
              const PetscInt fid = fine_cell_id(fx, fy, fz, sub_n);
              const PetscReal dscale =
                  simp_derivative(rho_value, ctx->density_options);
              d[fk][fj][fi] = -dscale * energies[fid];
            }
          }
        }
      }
    }
  }

  PetscCall(DMDAVecRestoreArrayDOFRead(uda, local_u, &ug));
  PetscCall(DMDAVecRestoreArrayRead(fda, local_rho, &r));
  PetscCall(DMDAVecRestoreArrayRead(fda, local_mask, &m));
  PetscCall(DMDAVecRestoreArray(fda, dc, &d));
  PetscCall(VecDestroy(&local_u));
  PetscCall(VecDestroy(&local_rho));
  PetscCall(VecDestroy(&local_mask));
  PetscCallMPI(MPI_Allreduce(&local_rho_sum, &global_rho_sum, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&local_mask_sum, &global_mask_sum, 1, MPIU_REAL,
                             MPI_SUM, PETSC_COMM_WORLD));
  *volume_fraction = global_mask_sum > 0.0 ? global_rho_sum / global_mask_sum : 0.0;
  return 0;
}

// OC 更新中的单个设计变量试探密度公式。
// 根据目标梯度 dc、体积梯度 dv 和拉格朗日乘子 lambda 决定 rho 增减，
// 同时施加 move limit、上下界和 mask 约束。
PetscReal trial_rho(PetscReal rho, PetscReal dc, PetscReal dv, PetscReal mask,
                    PetscReal lambda, const OptimizerOptions &opt) {
  if (mask > 1.5) return 1.0;
  if (mask <= 0.5) return 0.0;
  if (dv <= 0.0) return opt.rho_min;
  const PetscReal ratio = PetscMax(1.0e-16, -dc / (lambda * dv));
  PetscReal value = rho * PetscSqrtReal(ratio);
  value = PetscMax(rho - opt.move, PetscMin(rho + opt.move, value));
  return PetscMax(opt.rho_min, PetscMin(1.0, value));
}

// 执行一次 Optimality Criteria 密度更新。
// 函数通过二分搜索体积约束的拉格朗日乘子，使更新后的可设计区域平均密度接近 volfrac。
// rho 会被原地更新；硬实体和 void 区域由 mask 保持不变。
PetscErrorCode oc_update(DM eda, Vec rho, Vec mask, Vec dc, Vec dv,
                         const DensityOptions &density_options,
                         const OptimizerOptions &opt,
                         PetscReal *max_change) {
  PetscReal l1 = 1.0e-16, l2 = 1.0e16;
  PetscReal design_sum = 0.0;
  PetscCall(design_mask_sum(eda, mask, &design_sum));
  const PetscReal target_volume_sum = opt.volfrac * design_sum;

  for (PetscInt it = 0; it < 80; ++it) {
    const PetscReal lambda = 0.5 * (l1 + l2);
    PetscScalar ***r = nullptr, ***m = nullptr, ***d = nullptr, ***v = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscReal local_sum = 0.0, global_sum = 0.0;
    PetscCall(DMDAVecGetArrayRead(eda, rho, &r));
    PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
    PetscCall(DMDAVecGetArrayRead(eda, dc, &d));
    PetscCall(DMDAVecGetArrayRead(eda, dv, &v));
    PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          const PetscReal candidate =
              trial_rho(PetscRealPart(r[k][j][i]), PetscRealPart(d[k][j][i]),
                        PetscRealPart(v[k][j][i]), PetscRealPart(m[k][j][i]),
                        lambda, opt);
          local_sum += PetscRealPart(v[k][j][i]) * candidate;
        }
      }
    }
    PetscCall(DMDAVecRestoreArrayRead(eda, rho, &r));
    PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
    PetscCall(DMDAVecRestoreArrayRead(eda, dc, &d));
    PetscCall(DMDAVecRestoreArrayRead(eda, dv, &v));
    PetscCallMPI(MPI_Allreduce(&local_sum, &global_sum, 1, MPIU_REAL, MPI_SUM,
                               PETSC_COMM_WORLD));
    if (global_sum > target_volume_sum) l1 = lambda;
    else l2 = lambda;
  }

  PetscReal local_change = 0.0, global_change = 0.0;
  {
    const PetscReal lambda = 0.5 * (l1 + l2);
    PetscScalar ***r = nullptr, ***m = nullptr, ***d = nullptr, ***v = nullptr;
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscCall(DMDAVecGetArray(eda, rho, &r));
    PetscCall(DMDAVecGetArrayRead(eda, mask, &m));
    PetscCall(DMDAVecGetArrayRead(eda, dc, &d));
    PetscCall(DMDAVecGetArrayRead(eda, dv, &v));
    PetscCall(DMDAGetCorners(eda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          const PetscReal old_value = PetscRealPart(r[k][j][i]);
          PetscReal new_value = density_options.void_density;
          if (PetscRealPart(m[k][j][i]) > 1.5) {
            new_value = 1.0;
          } else if (PetscRealPart(m[k][j][i]) > 0.5) {
            new_value = trial_rho(old_value, PetscRealPart(d[k][j][i]),
                                  PetscRealPart(v[k][j][i]),
                                  PetscRealPart(m[k][j][i]), lambda, opt);
          }
          r[k][j][i] = new_value;
          local_change = PetscMax(local_change, PetscAbsReal(new_value - old_value));
        }
      }
    }
    PetscCall(DMDAVecRestoreArray(eda, rho, &r));
    PetscCall(DMDAVecRestoreArrayRead(eda, mask, &m));
    PetscCall(DMDAVecRestoreArrayRead(eda, dc, &d));
    PetscCall(DMDAVecRestoreArrayRead(eda, dv, &v));
  }
  PetscCallMPI(MPI_Allreduce(&local_change, &global_change, 1, MPIU_REAL, MPI_MAX,
                             PETSC_COMM_WORLD));
  *max_change = global_change;
  return 0;
}

// 将 PETSc Vec 以二进制 PETSc viewer 格式写到 path。
// 这种 .petscbin 文件可用于 checkpoint、后处理脚本读取和本地 VTK 转换。
// 写出是并行 collective 操作，所有 rank 都要参与。
PetscErrorCode write_vec_binary(Vec v, const char *path) {
  PetscViewer viewer = nullptr;
  PetscCall(PetscViewerBinaryOpen(PetscObjectComm(reinterpret_cast<PetscObject>(v)),
                                  path, FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(v, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  return 0;
}

// 写出 EMsFEM 优化的密度 checkpoint。
// 根据 iter 生成中间步或 final 文件名，并保存设计密度、物理密度、mask 等结果。
// 这些文件用于断点查看、后处理成 VTK，或后续恢复/比较优化结果。
PetscErrorCode write_ems_checkpoint(const OptimizerOptions &opt, PetscInt iter,
                                    PetscBool final, Vec rho_design,
                                    Vec rho_phys, Vec mask) {
  char design_path[PETSC_MAX_PATH_LEN];
  char phys_path[PETSC_MAX_PATH_LEN];
  char mask_path[PETSC_MAX_PATH_LEN];
  if (!opt.write_checkpoint) return 0;
  if (final) {
    PetscCall(PetscSNPrintf(design_path, sizeof(design_path),
                            "%s_final_rho_design.petscbin",
                            opt.checkpoint_prefix));
    PetscCall(PetscSNPrintf(phys_path, sizeof(phys_path),
                            "%s_final_rho_phys.petscbin",
                            opt.checkpoint_prefix));
    PetscCall(PetscSNPrintf(mask_path, sizeof(mask_path),
                            "%s_final_mask.petscbin",
                            opt.checkpoint_prefix));
  } else {
    PetscCall(PetscSNPrintf(design_path, sizeof(design_path),
                            "%s_iter_%06lld_rho_design.petscbin",
                            opt.checkpoint_prefix,
                            static_cast<long long>(iter)));
    PetscCall(PetscSNPrintf(phys_path, sizeof(phys_path),
                            "%s_iter_%06lld_rho_phys.petscbin",
                            opt.checkpoint_prefix,
                            static_cast<long long>(iter)));
    PetscCall(PetscSNPrintf(mask_path, sizeof(mask_path),
                            "%s_iter_%06lld_mask.petscbin",
                            opt.checkpoint_prefix,
                            static_cast<long long>(iter)));
  }
  PetscCall(write_vec_binary(rho_design, design_path));
  PetscCall(write_vec_binary(rho_phys, phys_path));
  if (final || iter == 0) PetscCall(write_vec_binary(mask, mask_path));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Wrote EMsFEM ANN checkpoint: %s, %s\n",
                        design_path, phys_path));
  return 0;
}

// 写出某个载荷工况对应的位移向量 checkpoint。
// 多工况时文件名会带 case 编号，避免不同载荷的位移场互相覆盖。
// 后处理合成 VTK 时可以把密度和位移放到同一个可视化文件中。
PetscErrorCode write_ems_displacement_checkpoint(const OptimizerOptions &opt,
                                                 PetscBool multi_case,
                                                 PetscInt load_case,
                                                 Vec u) {
  char u_path[PETSC_MAX_PATH_LEN];
  if (!opt.write_checkpoint) return 0;
  if (multi_case) {
    PetscCall(PetscSNPrintf(u_path, sizeof(u_path),
                            "%s_final_u_case_%03lld.petscbin",
                            opt.checkpoint_prefix,
                            static_cast<long long>(load_case)));
  } else {
    PetscCall(PetscSNPrintf(u_path, sizeof(u_path),
                            "%s_final_u.petscbin",
                            opt.checkpoint_prefix));
  }
  PetscCall(write_vec_binary(u, u_path));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Wrote EMsFEM ANN displacement checkpoint: %s\n",
                        u_path));
  return 0;
}

// 写出优化过程摘要文本文件。
// 摘要包含网格、体积分数、最终柔度、KSP 迭代、收敛状态和各阶段耗时。
// 该文件便于不用打开完整 out 日志也能快速确认一次算例是否正常。
PetscErrorCode write_ems_summary(const char *output_prefix,
                                 const Grid &grid,
                                 const DensityOptions &density_options,
                                 const OptimizerOptions &opt,
                                 const EmSfemAnnOptions &ems,
                                 PetscInt final_iter,
                                 PetscReal compliance,
                                 PetscReal volume,
                                 PetscReal raw_design_volume,
                                 PetscReal change,
                                 const TimingTotals &timing) {
  char path[PETSC_MAX_PATH_LEN];
  PetscViewer viewer = nullptr;
  PetscLogDouble mem_current = 0.0, mem_peak = 0.0;
  PetscLogDouble mem_current_max = 0.0, mem_peak_max = 0.0;
  PetscCall(PetscSNPrintf(path, sizeof(path), "%s_opt_summary.txt", output_prefix));
  PetscCall(PetscMemoryGetCurrentUsage(&mem_current));
  PetscCall(PetscMemoryGetMaximumUsage(&mem_peak));
  PetscCallMPI(MPI_Allreduce(&mem_current, &mem_current_max, 1, MPI_DOUBLE, MPI_MAX,
                             PETSC_COMM_WORLD));
  PetscCallMPI(MPI_Allreduce(&mem_peak, &mem_peak_max, 1, MPI_DOUBLE, MPI_MAX,
                             PETSC_COMM_WORLD));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, path, &viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "mode=optimize\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "operator=emsfem_ann\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "nx=%lld\n", static_cast<long long>(grid.nx)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ny=%lld\n", static_cast<long long>(grid.ny)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "nz=%lld\n", static_cast<long long>(grid.nz)));
  PetscCall(PetscViewerASCIIPrintf(viewer,
                                   "domain_length_m=%.12e\ndomain_width_m=%.12e\ndomain_height_m=%.12e\n",
                                   static_cast<double>(domain_length(grid)),
                                   static_cast<double>(domain_width(grid)),
                                   static_cast<double>(domain_height(grid))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "dof=%lld\n", static_cast<long long>(dof_count(grid))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "sub_n=%lld\n", static_cast<long long>(ems.sub_n)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "fine_density_cells=%lld\n",
                                   static_cast<long long>(fine_cell_count(grid, ems.sub_n))));
  PetscCall(PetscViewerASCIIPrintf(viewer, "control_arm_bc=%d\n", static_cast<int>(ems.control_arm_bc)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "benchmark_case=%s\n", ems.benchmark_case));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_iter=%lld\n", static_cast<long long>(opt.max_iter)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "volfrac_target=%.12e\n", static_cast<double>(opt.volfrac)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "young_modulus=%.12e\n",
                                   static_cast<double>(density_options.young_modulus)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "filter_radius=%.12e\n", static_cast<double>(opt.filter_radius)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "move_initial=%.12e\n", static_cast<double>(opt.move)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "move_min=%.12e\n", static_cast<double>(opt.move_min)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "move_shrink=%.12e\n", static_cast<double>(opt.move_shrink)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "move_growth=%.12e\n", static_cast<double>(opt.move_growth)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_compliance_increase=%.12e\n",
                                   static_cast<double>(opt.max_compliance_increase)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "stability_guard=%d\n",
                                   static_cast<int>(opt.stability_guard)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "projected_volume_correction=%d\n",
                                   static_cast<int>(opt.projected_volume_correction)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "heaviside_projection=%d\n",
                                   static_cast<int>(opt.heaviside_projection)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "heaviside_eta=%.12e\n",
                                   static_cast<double>(opt.heaviside_eta)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "heaviside_beta_initial=%.12e\n",
                                   static_cast<double>(opt.heaviside_beta_initial)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "heaviside_beta_max=%.12e\n",
                                   static_cast<double>(opt.heaviside_beta_max)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "heaviside_beta_interval=%lld\n",
                                   static_cast<long long>(opt.heaviside_beta_interval)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_iter=%lld\n", static_cast<long long>(final_iter)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_compliance=%.12e\n", static_cast<double>(compliance)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_volume=%.12e\n", static_cast<double>(volume)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_raw_design_volume=%.12e\n", static_cast<double>(raw_design_volume)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "final_change=%.12e\n", static_cast<double>(change)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "sensitivity_note=fine_subcell_energy_frozen_ann_shape\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_total_topopt_s=%.12e\n", static_cast<double>(timing.total_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_setup_s=%.12e\n", static_cast<double>(timing.setup_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_ann_load_s=%.12e\n", static_cast<double>(timing.ann_load_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_density_filter_s=%.12e\n", static_cast<double>(timing.density_filter_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_draft_closure_s=%.12e\n", static_cast<double>(timing.draft_closure_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_ann_cache_s=%.12e\n", static_cast<double>(timing.ann_cache_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_fine_material_sampling_s=%.12e\n", static_cast<double>(timing.fine_material_sampling_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_ann_shape_predict_s=%.12e\n", static_cast<double>(timing.ann_shape_predict_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_ems_matrix_assembly_s=%.12e\n", static_cast<double>(timing.ems_matrix_assembly_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_initial_ann_cache_s=%.12e\n", static_cast<double>(timing.initial_ann_cache_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_preconditioner_setup_s=%.12e\n", static_cast<double>(timing.preconditioner_setup_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_load_assembly_s=%.12e\n", static_cast<double>(timing.load_assembly_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_ksp_solve_s=%.12e\n", static_cast<double>(timing.ksp_solve_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_sensitivity_s=%.12e\n", static_cast<double>(timing.sensitivity_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_sensitivity_filter_s=%.12e\n", static_cast<double>(timing.sensitivity_filter_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_mma_oc_update_s=%.12e\n", static_cast<double>(timing.optimizer_update_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_checkpoint_write_s=%.12e\n", static_cast<double>(timing.checkpoint_write_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_iteration_total_s=%.12e\n", static_cast<double>(timing.iteration_total_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_final_eval_s=%.12e\n", static_cast<double>(timing.final_eval_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "timing_vtk_write_s=%.12e\n", static_cast<double>(timing.vtk_write_s)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_current_memory_bytes=%.0f\n", static_cast<double>(mem_current_max)));
  PetscCall(PetscViewerASCIIPrintf(viewer, "max_peak_memory_bytes=%.0f\n", static_cast<double>(mem_peak_max)));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "EMsFEM ANN optimization summary: %s\n", path));
  return 0;
}

// 写出 EMsFEM 优化结果 VTK。
// 函数把细密度/掩码和可选位移场转换成可视化网格数据，供 ParaView 查看。
// 大规模全细网格写出会非常大，因此实际使用时要谨慎选择 stride 或专门后处理脚本。
PetscErrorCode write_ems_opt_vtk(DM uda, DM fda, const char *path,
                                 const Grid &grid, PetscInt sub_n, Vec u,
                                 Vec rho, Vec mask) {
  PetscMPIInt rank = 0;
  Vec u_nat = nullptr, rho_nat = nullptr, mask_nat = nullptr;
  Vec u_seq = nullptr, rho_seq = nullptr, mask_seq = nullptr;
  VecScatter su = nullptr, sr = nullptr, sm = nullptr;
  const PetscInt fnx = fine_nelx(grid, sub_n);
  const PetscInt fny = fine_nely(grid, sub_n);
  const PetscInt fnz = fine_nelz(grid, sub_n);
  const PetscInt fine_points = (fnx + 1) * (fny + 1) * (fnz + 1);
  const PetscInt fine_cells = fine_cell_count(grid, sub_n);
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(DMDACreateNaturalVector(uda, &u_nat));
  PetscCall(DMDACreateNaturalVector(fda, &rho_nat));
  PetscCall(DMDACreateNaturalVector(fda, &mask_nat));
  PetscCall(DMDAGlobalToNaturalBegin(uda, u, INSERT_VALUES, u_nat));
  PetscCall(DMDAGlobalToNaturalEnd(uda, u, INSERT_VALUES, u_nat));
  PetscCall(DMDAGlobalToNaturalBegin(fda, rho, INSERT_VALUES, rho_nat));
  PetscCall(DMDAGlobalToNaturalEnd(fda, rho, INSERT_VALUES, rho_nat));
  PetscCall(DMDAGlobalToNaturalBegin(fda, mask, INSERT_VALUES, mask_nat));
  PetscCall(DMDAGlobalToNaturalEnd(fda, mask, INSERT_VALUES, mask_nat));
  PetscCall(VecScatterCreateToZero(u_nat, &su, &u_seq));
  PetscCall(VecScatterCreateToZero(rho_nat, &sr, &rho_seq));
  PetscCall(VecScatterCreateToZero(mask_nat, &sm, &mask_seq));
  PetscCall(VecScatterBegin(su, u_nat, u_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(su, u_nat, u_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(sr, rho_nat, rho_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sr, rho_nat, rho_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterBegin(sm, mask_nat, mask_seq, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sm, mask_nat, mask_seq, INSERT_VALUES, SCATTER_FORWARD));

  if (rank == 0) {
    const PetscScalar *ua = nullptr, *ra = nullptr, *ma = nullptr;
    FILE *fp = std::fopen(path, "w");
    PetscCheck(fp != nullptr, PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
               "Cannot open EMsFEM optimizer VTK: %s", path);
    PetscCall(VecGetArrayRead(u_seq, &ua));
    PetscCall(VecGetArrayRead(rho_seq, &ra));
    PetscCall(VecGetArrayRead(mask_seq, &ma));
    std::fprintf(fp, "# vtk DataFile Version 3.0\n");
    std::fprintf(fp, "emsfem_ann_optimizer_fine_density\nASCII\n");
    std::fprintf(fp, "DATASET STRUCTURED_GRID\n");
    std::fprintf(fp, "DIMENSIONS %lld %lld %lld\n",
                 static_cast<long long>(fnx + 1),
                 static_cast<long long>(fny + 1),
                 static_cast<long long>(fnz + 1));
    std::fprintf(fp, "POINTS %lld double\n",
                 static_cast<long long>(fine_points));
    for (PetscInt k = 0; k <= fnz; ++k) {
      for (PetscInt j = 0; j <= fny; ++j) {
        for (PetscInt i = 0; i <= fnx; ++i) {
          const PetscReal x = domain_length(grid) *
                              static_cast<PetscReal>(i) /
                              static_cast<PetscReal>(fnx);
          const PetscReal y = domain_width(grid) *
                              static_cast<PetscReal>(j) /
                              static_cast<PetscReal>(fny);
          const PetscReal z = domain_height(grid) *
                              static_cast<PetscReal>(k) /
                              static_cast<PetscReal>(fnz);
          std::fprintf(fp, "%.17g %.17g %.17g\n",
                       static_cast<double>(x),
                       static_cast<double>(y),
                       static_cast<double>(z));
        }
      }
    }
    std::fprintf(fp, "POINT_DATA %lld\n", static_cast<long long>(fine_points));
    std::fprintf(fp, "SCALARS displacement_magnitude double 1\nLOOKUP_TABLE default\n");
    for (PetscInt fk = 0; fk <= fnz; ++fk) {
      const PetscInt ck = fk == fnz ? grid.nz - 2 : fk / sub_n;
      const PetscReal tz = fk == fnz ? 1.0 :
          static_cast<PetscReal>(fk % sub_n) / static_cast<PetscReal>(sub_n);
      for (PetscInt fj = 0; fj <= fny; ++fj) {
        const PetscInt cj = fj == fny ? grid.ny - 2 : fj / sub_n;
        const PetscReal ty = fj == fny ? 1.0 :
            static_cast<PetscReal>(fj % sub_n) / static_cast<PetscReal>(sub_n);
        for (PetscInt fi = 0; fi <= fnx; ++fi) {
          const PetscInt ci = fi == fnx ? grid.nx - 2 : fi / sub_n;
          const PetscReal tx = fi == fnx ? 1.0 :
              static_cast<PetscReal>(fi % sub_n) / static_cast<PetscReal>(sub_n);
          PetscReal disp[3] = {};
          for (PetscInt dz = 0; dz <= 1; ++dz) {
            const PetscReal wz = dz == 0 ? 1.0 - tz : tz;
            for (PetscInt dy = 0; dy <= 1; ++dy) {
              const PetscReal wy = dy == 0 ? 1.0 - ty : ty;
              for (PetscInt dx = 0; dx <= 1; ++dx) {
                const PetscReal wx = dx == 0 ? 1.0 - tx : tx;
                const PetscInt p =
                    (ci + dx) + grid.nx * ((cj + dy) + grid.ny * (ck + dz));
                const PetscReal w = wx * wy * wz;
                for (PetscInt c = 0; c < 3; ++c) {
                  disp[c] += w * PetscRealPart(ua[3 * p + c]);
                }
              }
            }
          }
          std::fprintf(fp, "%.17g\n",
                       static_cast<double>(PetscSqrtReal(
                           disp[0] * disp[0] + disp[1] * disp[1] +
                           disp[2] * disp[2])));
        }
      }
    }
    std::fprintf(fp, "CELL_DATA %lld\n", static_cast<long long>(fine_cells));
    std::fprintf(fp, "SCALARS rho double 1\nLOOKUP_TABLE default\n");
    for (PetscInt c = 0; c < fine_cells; ++c) {
      std::fprintf(fp, "%.17g\n", static_cast<double>(PetscRealPart(ra[c])));
    }
    std::fprintf(fp, "SCALARS rho_plot double 1\nLOOKUP_TABLE default\n");
    for (PetscInt c = 0; c < fine_cells; ++c) {
      const PetscReal mask_value = PetscRealPart(ma[c]);
      const PetscReal value =
          mask_value <= 0.5 ? 0.0 : PetscRealPart(ra[c]);
      std::fprintf(fp, "%.17g\n", static_cast<double>(value));
    }
    std::fprintf(fp, "SCALARS design_mask double 1\nLOOKUP_TABLE default\n");
    for (PetscInt c = 0; c < fine_cells; ++c) {
      std::fprintf(fp, "%.17g\n", static_cast<double>(PetscRealPart(ma[c])));
    }
    PetscCall(VecRestoreArrayRead(u_seq, &ua));
    PetscCall(VecRestoreArrayRead(rho_seq, &ra));
    PetscCall(VecRestoreArrayRead(mask_seq, &ma));
    std::fclose(fp);
  }

  PetscCall(VecScatterDestroy(&su));
  PetscCall(VecScatterDestroy(&sr));
  PetscCall(VecScatterDestroy(&sm));
  PetscCall(VecDestroy(&u_seq));
  PetscCall(VecDestroy(&rho_seq));
  PetscCall(VecDestroy(&mask_seq));
  PetscCall(VecDestroy(&u_nat));
  PetscCall(VecDestroy(&rho_nat));
  PetscCall(VecDestroy(&mask_nat));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Wrote EMsFEM ANN fine-density optimization VTK: %s\n", path));
  return 0;
}

// 返回控制臂多工况中某个载荷工况的权重。
// 优化目标通常是各工况柔度的加权和，权重用于平衡不同方向/大小的载荷贡献。
PetscReal control_arm_case_weight(PetscInt load_case) {
  static const PetscReal weights[3] = {0.25, 0.50, 0.25};
  if (load_case >= 1 && load_case <= 3) return weights[load_case - 1];
  return 1.0;
}

// 为优化器填充指定 load_case 的右端载荷向量 b。
// 控制臂模式走真实控制臂载荷；简单模式走端部载荷。
// 输出 b 会在每个优化迭代里交给 KSPSolve 求解对应位移场。
PetscErrorCode fill_optimizer_load(DM uda,
                                   const Grid &grid,
                                   const EmSfemAnnOptions &ems_options,
                                   PetscReal load,
                                   PetscInt load_case,
                                   Vec b) {
  if (ems_options.control_arm_bc) {
    EmSfemAnnOptions case_options = ems_options;
    case_options.load_case = load_case;
    PetscCall(fill_control_arm_load(uda, grid, case_options, load, b));
  } else {
    PetscCall(fill_simple_tip_load(uda, grid, ems_options.benchmark_case,
                                   load, b));
  }
  return 0;
}

} // namespace

// 创建单次 EMsFEM ANN 线性系统。
// 函数创建 uda、密度、载荷、matrix-free A、可选辅助 P 和 KSP，适合 solve/postsolve 路径。
// 优化主循环有自己的更完整初始化流程，但核心对象关系与这里一致。
PetscErrorCode create_emsfem_ann_system(const Grid &grid,
                                        const DensityOptions &density_options,
                                        const EmSfemAnnOptions &ems_options,
                                        PetscReal load_scale,
                                        Mat *A,
                                        Vec *u,
                                        Vec *b) {
  EmSfemAnnContext *ctx = new EmSfemAnnContext();
  PetscInt local_size = 0;
  PetscInt local_fixed = 0;
  PetscInt global_fixed = 0;

  ctx->grid = grid;
  ctx->density_options = density_options;
  ctx->ems_options = ems_options;
  PetscCall(ctx->ann.load(ems_options.ann_dir, ems_options.sub_n));

  const PetscReal hx = domain_length(grid) /
                       static_cast<PetscReal>((grid.nx - 1) * ems_options.sub_n);
  const PetscReal hy = domain_width(grid) /
                       static_cast<PetscReal>((grid.ny - 1) * ems_options.sub_n);
  const PetscReal hz = domain_height(grid) /
                       static_cast<PetscReal>((grid.nz - 1) * ems_options.sub_n);
  compute_h8_element_stiffness(hx, hy, hz, density_options.young_modulus,
                               ctx->fine_ke);
  {
    std::vector<PetscReal> shape;
    std::vector<PetscReal> material(static_cast<std::size_t>(
                                        ems_options.sub_n * ems_options.sub_n *
                                        ems_options.sub_n),
                                    1.0);
    fill_trilinear_shape(ems_options.sub_n, &shape);
    accumulate_shape_stiffness(ctx->fine_ke, shape, material, ems_options.sub_n,
                               ctx->solid_ke);
  }

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                         DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                         grid.nx, grid.ny, grid.nz,
                         PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE,
                         3, 1, nullptr, nullptr, nullptr, &ctx->da));
  PetscCall(DMSetFromOptions(ctx->da));
  PetscCall(DMSetUp(ctx->da));
  PetscCall(DMCreateGlobalVector(ctx->da, u));
  PetscCall(VecDuplicate(*u, b));
  PetscCall(DMCreateLocalVector(ctx->da, &ctx->local_x));
  PetscCall(VecGetLocalSize(*u, &local_size));

  if (ems_options.cache_element_matrices) {
    PetscCall(build_local_k_cache(ctx));
  }

  {
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscCall(DMDAGetCorners(ctx->da, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (is_fixed_node(i, j, k, *ctx)) ++local_fixed;
        }
      }
    }
  }
  PetscCallMPI(MPI_Allreduce(&local_fixed, &global_fixed, 1, MPIU_INT, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCheck(global_fixed > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "No fixed nodes were selected for the EMsFEM ANN operator");

  PetscCall(MatCreateShell(PETSC_COMM_WORLD, local_size, local_size,
                           dof_count(grid), dof_count(grid), ctx, A));
  PetscCall(MatShellSetOperation(*A, MATOP_MULT,
                                 reinterpret_cast<void (*)(void)>(emsfem_mult)));
  PetscCall(MatShellSetOperation(*A, MATOP_GET_DIAGONAL,
                                 reinterpret_cast<void (*)(void)>(emsfem_get_diagonal)));
  PetscCall(MatShellSetOperation(*A, MATOP_DESTROY,
                                 reinterpret_cast<void (*)(void)>(emsfem_destroy)));
  PetscCall(MatSetOption(*A, MAT_SYMMETRIC, PETSC_TRUE));

  if (ems_options.control_arm_bc) {
    PetscCall(fill_control_arm_load(ctx->da, grid, ems_options, load_scale, *b));
  } else {
    PetscCall(fill_simple_tip_load(ctx->da, grid, ems_options.benchmark_case,
                                   load_scale, *b));
  }
  PetscCall(VecSet(*u, 0.0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "EMsFEM ANN system ready: sub_n=%lld fixed_nodes=%lld control_arm_bc=%d benchmark_case=%s\n",
                        static_cast<long long>(ems_options.sub_n),
                        static_cast<long long>(global_fixed),
                        static_cast<int>(ems_options.control_arm_bc),
                        ems_options.benchmark_case));
  return 0;
}

// 将单次 EMsFEM ANN 求解结果写成 VTK。
// 从 MatShell 中取出上下文，使用其中的 uda/fda/密度和位移向量进行可视化输出。
// 主要服务 -mode solve 或 postsolve 后处理。
PetscErrorCode write_emsfem_ann_solution_vtk(Mat A,
                                             Vec displacement,
                                             const char *path,
                                             const Grid &grid,
                                             const DensityOptions &options) {
  EmSfemAnnContext *ctx = nullptr;
  Vec natural = nullptr;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(DMDACreateNaturalVector(ctx->da, &natural));
  PetscCall(DMDAGlobalToNaturalBegin(ctx->da, displacement, INSERT_VALUES, natural));
  PetscCall(DMDAGlobalToNaturalEnd(ctx->da, displacement, INSERT_VALUES, natural));
  PetscCall(write_structured_solution_vtk(natural, path, grid, options));
  PetscCall(VecDestroy(&natural));
  return 0;
}

// EMsFEM ANN 的 postsolve 模式入口。
// 它读取已有密度/结构文件，重建 K-cache 和线性系统，求解位移并写 checkpoint/VTK。
// 该模式用于优化完成后补算位移场、检查线性求解或生成可视化结果。
PetscErrorCode run_emsfem_ann_postsolve(const Grid &grid,
                                        const DensityOptions &density_options,
                                        const OptimizerOptions &optimizer_options,
                                        const EmSfemAnnOptions &ems_options,
                                        const char *density_file,
                                        const char *mask_file,
                                        const char *output_prefix) {
  PetscCheck(ems_options.cache_element_matrices, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "EMsFEM ANN postsolve currently requires -ems_cache_element_matrices true");
  PetscCheck(density_file != nullptr && density_file[0] != '\0',
             PETSC_COMM_WORLD, PETSC_ERR_ARG_NULL,
             "EMsFEM ANN postsolve needs -density_file <*_rho_phys.petscbin>");

  DM uda = nullptr, fda = nullptr, cda = nullptr;
  Vec rho_phys = nullptr, mask = nullptr, u = nullptr, b = nullptr;
  Vec coarse_rho = nullptr;
  Mat A = nullptr, P = nullptr;
  KSP ksp = nullptr;
  EmsBlockJacobiContext *pc_ctx = nullptr;
  EmSfemAnnContext *ctx = new EmSfemAnnContext();
  char ems_pc_type[64] = "aux_ann_gamg";
  PetscBool ems_uses_aux_matrix = PETSC_FALSE;
  PetscInt local_size = 0;
  PetscInt local_fixed = 0, global_fixed = 0;
  PetscReal compliance = 0.0;
  PetscReal rnorm_max = 0.0;
  PetscInt total_its = 0;
  PetscLogDouble t0 = 0.0;
  PetscReal setup_s = 0.0, solve_s = 0.0;
  OptimizerOptions write_options = optimizer_options;

  write_options.write_checkpoint = PETSC_TRUE;
  PetscCall(get_ems_pc_type_option(ems_pc_type, sizeof(ems_pc_type),
                                   &ems_uses_aux_matrix));
  const PetscReal fine_filter_radius =
      optimizer_options.filter_radius *
      static_cast<PetscReal>(ems_options.sub_n);
  // K-cache 只需要一层粗单元宽度的细密度 ghost；密度过滤需要覆盖过滤半径。
  const PetscInt fine_stencil =
      PetscMax(ems_options.sub_n,
               static_cast<PetscInt>(PetscCeilReal(fine_filter_radius)));

  PetscCall(PetscTime(&t0));
  PetscCall(create_ems_opt_dms(grid, ems_options.sub_n, fine_stencil,
                               &uda, &fda));
  ctx->da = uda;
  ctx->grid = grid;
  ctx->density_options = density_options;
  ctx->ems_options = ems_options;
  PetscCall(ctx->ann.load(ems_options.ann_dir, ems_options.sub_n));

  {
    const PetscReal hx =
        domain_length(grid) /
        static_cast<PetscReal>((grid.nx - 1) * ems_options.sub_n);
    const PetscReal hy =
        domain_width(grid) /
        static_cast<PetscReal>((grid.ny - 1) * ems_options.sub_n);
    const PetscReal hz =
        domain_height(grid) /
        static_cast<PetscReal>((grid.nz - 1) * ems_options.sub_n);
    compute_h8_element_stiffness(hx, hy, hz, density_options.young_modulus,
                                 ctx->fine_ke);
    std::vector<PetscReal> shape;
    std::vector<PetscReal> material(static_cast<std::size_t>(
                                        ems_options.sub_n * ems_options.sub_n *
                                        ems_options.sub_n),
                                    1.0);
    fill_trilinear_shape(ems_options.sub_n, &shape);
    accumulate_shape_stiffness(ctx->fine_ke, shape, material, ems_options.sub_n,
                               ctx->solid_ke);
  }

  PetscCall(DMCreateGlobalVector(fda, &rho_phys));
  PetscCall(load_dmda_natural_binary_checkpoint(fda, rho_phys, density_file));
  if (mask_file != nullptr && mask_file[0] != '\0') {
    PetscCall(DMCreateGlobalVector(fda, &mask));
    PetscCall(load_dmda_natural_binary_checkpoint(fda, mask, mask_file));
  }

  PetscCall(DMCreateGlobalVector(uda, &u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(DMCreateLocalVector(uda, &ctx->local_x));
  PetscCall(VecGetLocalSize(u, &local_size));
  {
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (is_fixed_node(i, j, k, *ctx)) ++local_fixed;
        }
      }
    }
  }
  PetscCallMPI(MPI_Allreduce(&local_fixed, &global_fixed, 1, MPIU_INT, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCheck(global_fixed > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "No fixed nodes were selected for EMsFEM ANN postsolve");

  {
    EmsCacheTiming cache_timing;
    PetscCall(rebuild_local_k_cache_from_rho(ctx, fda, rho_phys,
                                             &cache_timing));
  }
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, local_size, local_size,
                           dof_count(grid), dof_count(grid), ctx, &A));
  PetscCall(MatShellSetOperation(A, MATOP_MULT,
                                 reinterpret_cast<void (*)(void)>(emsfem_mult)));
  PetscCall(MatShellSetOperation(A, MATOP_GET_DIAGONAL,
                                 reinterpret_cast<void (*)(void)>(
                                     emsfem_get_diagonal)));
  PetscCall(MatShellSetOperation(A, MATOP_DESTROY,
                                 reinterpret_cast<void (*)(void)>(
                                     emsfem_destroy)));
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));

  if (ems_pc_type_uses_coarse_density_aux_matrix(ems_pc_type)) {
    PetscCall(create_ems_coarse_density_dm(uda, grid, &cda));
    PetscCall(DMCreateGlobalVector(cda, &coarse_rho));
    PetscCall(build_ems_coarse_density(cda, fda, rho_phys, grid,
                                       ems_options.sub_n, coarse_rho));
    if (ems_pc_type_uses_low_order_aux_matrix(ems_pc_type)) {
      PetscCall(create_ems_aux_laplacian_matrix(uda, cda, coarse_rho,
                                                *ctx, &P));
    } else {
      PetscCall(create_ems_aux_elasticity_matrix(uda, cda, coarse_rho,
                                                 *ctx, &P));
    }
  } else if (ems_pc_type_uses_ann_aux_matrix(ems_pc_type)) {
    PetscCall(create_ems_aux_ann_matrix(uda, ctx, &P));
  } else if (std::strcmp(ems_pc_type, "block_jacobi") == 0) {
    pc_ctx = new EmsBlockJacobiContext();
    pc_ctx->ems = ctx;
    PetscCall(setup_ems_block_jacobi(pc_ctx));
  }

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(configure_ems_ksp(ksp, A, P, pc_ctx, grid, optimizer_options,
                              ems_pc_type));
  PetscCall(elapsed_max_since(t0, &setup_s));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "EMsFEM ANN postsolve: density=%s mask=%s nx=%lld ny=%lld nz=%lld sub_n=%lld dof=%lld pc=%s fixed_nodes=%lld\n",
                        density_file,
                        (mask_file != nullptr && mask_file[0] != '\0')
                            ? mask_file
                            : "(none)",
                        static_cast<long long>(grid.nx),
                        static_cast<long long>(grid.ny),
                        static_cast<long long>(grid.nz),
                        static_cast<long long>(ems_options.sub_n),
                        static_cast<long long>(dof_count(grid)),
                        ems_pc_type,
                        static_cast<long long>(global_fixed)));

  PetscCall(PetscTime(&t0));
  {
    const PetscBool multi_case =
        (ems_options.control_arm_bc &&
         (ems_options.load_case <= 0 || ems_options.load_case > 3))
            ? PETSC_TRUE
            : PETSC_FALSE;
    const PetscInt first_case = multi_case ? 1 : ems_options.load_case;
    const PetscInt last_case = multi_case ? 3 : ems_options.load_case;
    for (PetscInt load_case = first_case; load_case <= last_case; ++load_case) {
      PetscScalar dot = 0.0;
      PetscInt case_its = 0;
      PetscReal case_rnorm = 0.0;
      const PetscReal weight =
          multi_case ? control_arm_case_weight(load_case) : 1.0;
      PetscCall(fill_optimizer_load(uda, grid, ems_options,
                                    optimizer_options.load, load_case, b));
      PetscCall(VecSet(u, 0.0));
      PetscCall(KSPSolve(ksp, b, u));
      PetscCall(KSPGetIterationNumber(ksp, &case_its));
      PetscCall(KSPGetResidualNorm(ksp, &case_rnorm));
      KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
      PetscCall(KSPGetConvergedReason(ksp, &reason));
      PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED,
                 "EMsFEM ANN postsolve did not converge for load_case=%lld",
                 static_cast<long long>(load_case));
      PetscCall(write_ems_displacement_checkpoint(write_options, multi_case,
                                                  load_case, u));
      PetscCall(VecDot(b, u, &dot));
      compliance += weight * PetscRealPart(dot);
      total_its += case_its;
      rnorm_max = PetscMax(rnorm_max, case_rnorm);
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "postsolve load_case=%lld ksp_it=%lld residual=%.3e compliance=%.6e\n",
                            static_cast<long long>(load_case),
                            static_cast<long long>(case_its),
                            static_cast<double>(case_rnorm),
                            static_cast<double>(PetscRealPart(dot))));
    }
  }
  PetscCall(elapsed_max_since(t0, &solve_s));

  {
    char summary_path[PETSC_MAX_PATH_LEN];
    PetscViewer viewer = nullptr;
    PetscCall(PetscSNPrintf(summary_path, sizeof(summary_path),
                            "%s_postsolve_summary.txt", output_prefix));
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, summary_path, &viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "mode=ems_ann_postsolve\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "density_file=%s\n",
                                     density_file));
    PetscCall(PetscViewerASCIIPrintf(
        viewer, "mask_file=%s\n",
        (mask_file != nullptr && mask_file[0] != '\0') ? mask_file : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "nx=%lld\n",
                                     static_cast<long long>(grid.nx)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ny=%lld\n",
                                     static_cast<long long>(grid.ny)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "nz=%lld\n",
                                     static_cast<long long>(grid.nz)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "sub_n=%lld\n",
                                     static_cast<long long>(ems_options.sub_n)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "dof=%lld\n",
                                     static_cast<long long>(dof_count(grid))));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ems_pc_type=%s\n",
                                     ems_pc_type));
    PetscCall(PetscViewerASCIIPrintf(viewer, "load_case=%lld\n",
                                     static_cast<long long>(
                                         ems_options.load_case)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ksp_iterations=%lld\n",
                                     static_cast<long long>(total_its)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ksp_residual=%.12e\n",
                                     static_cast<double>(rnorm_max)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "compliance=%.12e\n",
                                     static_cast<double>(compliance)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "setup_time_s=%.12e\n",
                                     static_cast<double>(setup_s)));
    PetscCall(PetscViewerASCIIPrintf(viewer, "solve_time_s=%.12e\n",
                                     static_cast<double>(solve_s)));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "EMsFEM ANN postsolve summary: %s\n",
                          summary_path));
  }

  PetscCall(KSPDestroy(&ksp));
  delete pc_ctx;
  pc_ctx = nullptr;
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  uda = nullptr;
  PetscCall(VecDestroy(&coarse_rho));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&mask));
  PetscCall(VecDestroy(&rho_phys));
  PetscCall(DMDestroy(&cda));
  PetscCall(DMDestroy(&fda));
  return 0;
}

// EMsFEM ANN 拓扑优化主入口。
// 该函数负责创建 DMDA、初始化密度/mask、配置 KSP/预条件器，并执行多步优化循环。
// 每步会过滤/投影密度、重建 K-cache、逐工况求解、累计柔度灵敏度、OC 更新并按需写 checkpoint。
PetscErrorCode run_emsfem_ann_optimizer(const Grid &grid,
                                        const DensityOptions &density_options,
                                        const OptimizerOptions &optimizer_options,
                                        const EmSfemAnnOptions &ems_options,
                                        const char *output_prefix,
                                        const char *final_vtk_file) {
  PetscCheck(ems_options.cache_element_matrices, PETSC_COMM_WORLD, PETSC_ERR_SUP,
             "EMsFEM ANN optimization currently requires -ems_cache_element_matrices true because KSP needs many MatMults per design iteration");

  DM uda = nullptr, fda = nullptr;
  Vec rho_design = nullptr, rho_filtered = nullptr, rho_phys = nullptr;
  Vec rho_last_good = nullptr;
  Vec mask = nullptr;
  Vec u = nullptr, b = nullptr;
  Vec dc_phys = nullptr, dc_case = nullptr, dc_projected = nullptr;
  Vec dc_design = nullptr;
  Vec dc_last_good = nullptr, dv_design = nullptr;
  Vec filter_denom = nullptr;
  Mat A = nullptr;
  Mat P = nullptr;
  KSP ksp = nullptr;
  DM cda = nullptr;
  Vec coarse_rho = nullptr;
  EmsBlockJacobiContext *pc_ctx = nullptr;
  PetscViewer hist = nullptr;
  char hist_path[PETSC_MAX_PATH_LEN];
  std::vector<ObjectiveVolumePoint> objective_history;
  PetscReal compliance = 0.0, volume = 0.0, change = 0.0;
  PetscReal accepted_compliance = PETSC_MAX_REAL;
  PetscReal current_move = optimizer_options.move;
  PetscReal last_good_effective_raw_volfrac = optimizer_options.volfrac;
  PetscBool have_last_good = PETSC_FALSE;
  PetscBool last_iteration_accepted = PETSC_TRUE;
  PetscInt reject_streak = 0;
  PetscInt final_iter = 0;
  PetscInt local_size = 0;
  PetscInt local_fixed = 0, global_fixed = 0;
  EmSfemAnnContext *ctx = new EmSfemAnnContext();
  TimingTotals timings;
  PetscLogDouble topopt_start = 0.0;
  PetscLogDouble setup_start = 0.0;
  char ems_pc_type[64] = "aux_ann_gamg";
  PetscBool ems_uses_aux_matrix = PETSC_FALSE;
  PetscCall(PetscTime(&topopt_start));
  PetscCall(PetscTime(&setup_start));
  PetscCall(get_ems_pc_type_option(ems_pc_type, sizeof(ems_pc_type),
                                   &ems_uses_aux_matrix));
  const PetscReal fine_filter_radius =
      optimizer_options.filter_radius *
      static_cast<PetscReal>(ems_options.sub_n);
  // K-cache 只需要一层粗单元宽度的细密度 ghost；密度过滤需要覆盖过滤半径。
  const PetscInt fine_stencil =
      PetscMax(ems_options.sub_n,
               static_cast<PetscInt>(PetscCeilReal(fine_filter_radius)));

  PetscCall(create_ems_opt_dms(grid, ems_options.sub_n, fine_stencil,
                               &uda, &fda));
  ctx->da = uda;
  ctx->grid = grid;
  ctx->density_options = density_options;
  ctx->ems_options = ems_options;
  {
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    PetscCall(PetscTime(&stage_start));
    PetscCall(ctx->ann.load(ems_options.ann_dir, ems_options.sub_n));
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.ann_load_s += elapsed;
  }

  const PetscReal hx = domain_length(grid) /
                       static_cast<PetscReal>((grid.nx - 1) * ems_options.sub_n);
  const PetscReal hy = domain_width(grid) /
                       static_cast<PetscReal>((grid.ny - 1) * ems_options.sub_n);
  const PetscReal hz = domain_height(grid) /
                       static_cast<PetscReal>((grid.nz - 1) * ems_options.sub_n);
  compute_h8_element_stiffness(hx, hy, hz, density_options.young_modulus,
                               ctx->fine_ke);
  {
    std::vector<PetscReal> shape;
    std::vector<PetscReal> material(static_cast<std::size_t>(
                                        ems_options.sub_n * ems_options.sub_n *
                                        ems_options.sub_n),
                                    1.0);
    fill_trilinear_shape(ems_options.sub_n, &shape);
    accumulate_shape_stiffness(ctx->fine_ke, shape, material, ems_options.sub_n,
                               ctx->solid_ke);
  }

  PetscCall(create_fine_mask_and_density(fda, grid, density_options,
                                         optimizer_options, ems_options.sub_n,
                                         &mask, &rho_design));
  PetscCall(VecDuplicate(rho_design, &rho_filtered));
  PetscCall(VecDuplicate(rho_design, &rho_phys));
  PetscCall(VecDuplicate(rho_design, &rho_last_good));
  PetscCall(VecDuplicate(rho_design, &filter_denom));
  PetscReal current_beta = heaviside_beta_for_iter(optimizer_options, 1);
  {
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    PetscCall(PetscTime(&stage_start));
    PetscCall(compute_filter_denominator(fda, mask, fine_filter_radius,
                                         filter_denom));
    PetscCall(apply_density_filter(fda, rho_design, mask, filter_denom,
                                   fine_filter_radius,
                                   density_options.void_density, rho_filtered));
    PetscCall(apply_heaviside_projection(fda, rho_filtered, mask,
                                         optimizer_options, current_beta,
                                         density_options.void_density,
                                         rho_phys));
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.density_filter_s += elapsed;
  }
  {
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    PetscCall(PetscTime(&stage_start));
    PetscCall(apply_z_draft_closure(fda, mask, density_options.void_density,
                                    rho_phys));
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.draft_closure_s += elapsed;
  }
  PetscCall(VecDuplicate(rho_design, &dv_design));
  PetscCall(set_active_volume_vector(fda, mask, dv_design));

  PetscCall(DMCreateGlobalVector(uda, &u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(DMCreateLocalVector(uda, &ctx->local_x));
  PetscCall(VecGetLocalSize(u, &local_size));
  PetscCall(DMCreateGlobalVector(fda, &dc_phys));
  PetscCall(VecDuplicate(dc_phys, &dc_case));
  PetscCall(VecDuplicate(dc_phys, &dc_projected));
  PetscCall(VecDuplicate(dc_phys, &dc_design));
  PetscCall(VecDuplicate(dc_phys, &dc_last_good));

  {
    PetscInt xs = 0, ys = 0, zs = 0, xm = 0, ym = 0, zm = 0;
    PetscCall(DMDAGetCorners(uda, &xs, &ys, &zs, &xm, &ym, &zm));
    for (PetscInt k = zs; k < zs + zm; ++k) {
      for (PetscInt j = ys; j < ys + ym; ++j) {
        for (PetscInt i = xs; i < xs + xm; ++i) {
          if (is_fixed_node(i, j, k, *ctx)) ++local_fixed;
        }
      }
    }
  }
  PetscCallMPI(MPI_Allreduce(&local_fixed, &global_fixed, 1, MPIU_INT, MPI_SUM,
                             PETSC_COMM_WORLD));
  PetscCheck(global_fixed > 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG,
             "No fixed nodes were selected for EMsFEM ANN optimization");

  {
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    EmsCacheTiming cache_timing;
    PetscCall(PetscTime(&stage_start));
    PetscCall(rebuild_local_k_cache_from_rho(ctx, fda, rho_phys,
                                             &cache_timing));
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.ann_cache_s += elapsed;
    timings.initial_ann_cache_s += elapsed;
    timings.fine_material_sampling_s += cache_timing.material_sampling_s;
    timings.ann_shape_predict_s += cache_timing.ann_shape_predict_s;
    timings.ems_matrix_assembly_s += cache_timing.matrix_assembly_s;
  }
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, local_size, local_size,
                           dof_count(grid), dof_count(grid), ctx, &A));
  PetscCall(MatShellSetOperation(A, MATOP_MULT,
                                 reinterpret_cast<void (*)(void)>(emsfem_mult)));
  PetscCall(MatShellSetOperation(A, MATOP_GET_DIAGONAL,
                                 reinterpret_cast<void (*)(void)>(emsfem_get_diagonal)));
  PetscCall(MatShellSetOperation(A, MATOP_DESTROY,
                                 reinterpret_cast<void (*)(void)>(emsfem_destroy)));
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));

  if (ems_pc_type_uses_coarse_density_aux_matrix(ems_pc_type)) {
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    PetscCall(PetscTime(&stage_start));
    PetscCall(create_ems_coarse_density_dm(uda, grid, &cda));
    PetscCall(DMCreateGlobalVector(cda, &coarse_rho));
    PetscCall(build_ems_coarse_density(cda, fda, rho_phys, grid,
                                       ems_options.sub_n, coarse_rho));
    if (ems_pc_type_uses_low_order_aux_matrix(ems_pc_type)) {
      PetscCall(create_ems_aux_laplacian_matrix(uda, cda, coarse_rho,
                                                *ctx, &P));
    } else {
      PetscCall(create_ems_aux_elasticity_matrix(uda, cda, coarse_rho,
                                                 *ctx, &P));
    }
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.preconditioner_setup_s += elapsed;
  } else if (ems_pc_type_uses_ann_aux_matrix(ems_pc_type)) {
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    PetscCall(PetscTime(&stage_start));
    PetscCall(create_ems_aux_ann_matrix(uda, ctx, &P));
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.preconditioner_setup_s += elapsed;
  } else if (std::strcmp(ems_pc_type, "block_jacobi") == 0) {
    pc_ctx = new EmsBlockJacobiContext();
    pc_ctx->ems = ctx;
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    PetscCall(PetscTime(&stage_start));
    PetscCall(setup_ems_block_jacobi(pc_ctx));
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.preconditioner_setup_s += elapsed;
  }
  {
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    PetscCall(PetscTime(&stage_start));
    PetscCall(write_ems_checkpoint(optimizer_options, 0, PETSC_FALSE,
                                   rho_design, rho_phys, mask));
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.checkpoint_write_s += elapsed;
  }

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(configure_ems_ksp(ksp, A, P, pc_ctx, grid, optimizer_options,
                              ems_pc_type));

  PetscCall(PetscSNPrintf(hist_path, sizeof(hist_path), "%s_history.csv",
                          output_prefix));
  PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, hist_path, &hist));
  PetscCall(PetscViewerASCIIPrintf(hist,
                                   "iter,compliance,volume,raw_design_volume,effective_raw_volfrac,projected_volume_gap,heaviside_beta,change,ksp_iterations,ksp_residual,effective_move,converged_cases,diverged_cases,accepted,iteration_total_time_s,density_filter_time_s,draft_closure_time_s,ann_cache_time_s,fine_material_sampling_time_s,ann_shape_predict_time_s,ems_matrix_assembly_time_s,preconditioner_setup_time_s,load_assembly_time_s,ksp_solve_time_s,sensitivity_time_s,sensitivity_filter_time_s,mma_oc_update_time_s,checkpoint_write_time_s,fem_total_time_s\n"));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                        "Optimize mode: emsfem_ann nx=%lld ny=%lld nz=%lld sub_n=%lld fine_density_cells=%lld max_iter=%lld volfrac=%g coarse_filter_radius=%g fine_filter_radius=%g heaviside=%d beta0=%g beta_max=%g beta_interval=%lld fixed_nodes=%lld\n",
                        static_cast<long long>(grid.nx),
                        static_cast<long long>(grid.ny),
                        static_cast<long long>(grid.nz),
                        static_cast<long long>(ems_options.sub_n),
                        static_cast<long long>(fine_cell_count(grid, ems_options.sub_n)),
                        static_cast<long long>(optimizer_options.max_iter),
                        static_cast<double>(optimizer_options.volfrac),
                        static_cast<double>(optimizer_options.filter_radius),
                        static_cast<double>(fine_filter_radius),
                        static_cast<int>(optimizer_options.heaviside_projection),
                        static_cast<double>(optimizer_options.heaviside_beta_initial),
                        static_cast<double>(optimizer_options.heaviside_beta_max),
                        static_cast<long long>(optimizer_options.heaviside_beta_interval),
                        static_cast<long long>(global_fixed)));
  PetscCall(elapsed_max_since(setup_start, &timings.setup_s));

  for (PetscInt iter = 1; iter <= optimizer_options.max_iter; ++iter) {
    PetscLogDouble iter_start = 0.0;
    PetscReal iter_total_s = 0.0;
    PetscReal iter_density_filter_s = 0.0;
    PetscReal iter_draft_closure_s = 0.0;
    PetscReal iter_ann_cache_s = 0.0;
    PetscReal iter_fine_material_sampling_s = 0.0;
    PetscReal iter_ann_shape_predict_s = 0.0;
    PetscReal iter_ems_matrix_assembly_s = 0.0;
    PetscReal iter_pc_setup_s = 0.0;
    PetscReal iter_load_assembly_s = 0.0;
    PetscReal iter_ksp_solve_s = 0.0;
    PetscReal iter_sensitivity_s = 0.0;
    PetscReal iter_sensitivity_filter_s = 0.0;
    PetscReal iter_optimizer_update_s = 0.0;
    PetscReal iter_checkpoint_s = 0.0;
    PetscCall(PetscTime(&iter_start));
    PetscInt its = 0;
    PetscInt converged_cases = 0;
    PetscInt diverged_cases = 0;
    PetscReal rnorm = 0.0;
    PetscReal rnorm_max = 0.0;
    current_beta = heaviside_beta_for_iter(optimizer_options, iter);
    {
      PetscLogDouble stage_start = 0.0;
      PetscCall(PetscTime(&stage_start));
      PetscCall(apply_density_filter(fda, rho_design, mask, filter_denom,
                                     fine_filter_radius,
                                     density_options.void_density, rho_filtered));
      PetscCall(apply_heaviside_projection(fda, rho_filtered, mask,
                                           optimizer_options, current_beta,
                                           density_options.void_density,
                                           rho_phys));
      PetscCall(elapsed_max_since(stage_start, &iter_density_filter_s));
      timings.density_filter_s += iter_density_filter_s;
    }
    {
      PetscLogDouble stage_start = 0.0;
      PetscCall(PetscTime(&stage_start));
      PetscCall(apply_z_draft_closure(fda, mask, density_options.void_density,
                                      rho_phys));
      PetscCall(elapsed_max_since(stage_start, &iter_draft_closure_s));
      timings.draft_closure_s += iter_draft_closure_s;
    }
    {
      PetscLogDouble stage_start = 0.0;
      EmsCacheTiming cache_timing;
      PetscCall(PetscTime(&stage_start));
      PetscCall(rebuild_local_k_cache_from_rho(ctx, fda, rho_phys,
                                               &cache_timing));
      PetscCall(elapsed_max_since(stage_start, &iter_ann_cache_s));
      timings.ann_cache_s += iter_ann_cache_s;
      iter_fine_material_sampling_s = cache_timing.material_sampling_s;
      iter_ann_shape_predict_s = cache_timing.ann_shape_predict_s;
      iter_ems_matrix_assembly_s = cache_timing.matrix_assembly_s;
      timings.fine_material_sampling_s += iter_fine_material_sampling_s;
      timings.ann_shape_predict_s += iter_ann_shape_predict_s;
      timings.ems_matrix_assembly_s += iter_ems_matrix_assembly_s;
    }
    {
      PetscLogDouble stage_start = 0.0;
      PetscCall(PetscTime(&stage_start));
      if (ems_pc_type_uses_coarse_density_aux_matrix(ems_pc_type)) {
        PetscCall(build_ems_coarse_density(cda, fda, rho_phys, grid,
                                           ems_options.sub_n, coarse_rho));
      }
      PetscCall(setup_ems_preconditioner(ksp, A, P, uda, cda, coarse_rho,
                                         pc_ctx, ctx, ems_pc_type));
      PetscCall(elapsed_max_since(stage_start, &iter_pc_setup_s));
      timings.preconditioner_setup_s += iter_pc_setup_s;
    }
    PetscCall(VecSet(dc_phys, 0.0));
    compliance = 0.0;
    const PetscBool multi_case =
        (ems_options.control_arm_bc &&
         (ems_options.load_case <= 0 || ems_options.load_case > 3))
            ? PETSC_TRUE
            : PETSC_FALSE;
    const PetscInt first_case = multi_case ? 1 : ems_options.load_case;
    const PetscInt last_case = multi_case ? 3 : ems_options.load_case;
    for (PetscInt load_case = first_case; load_case <= last_case; ++load_case) {
      PetscScalar dot = 0.0;
      PetscReal case_volume = 0.0;
      PetscInt case_its = 0;
      PetscReal case_rnorm = 0.0;
      const PetscReal weight = multi_case ? control_arm_case_weight(load_case) : 1.0;
      {
        PetscLogDouble stage_start = 0.0;
        PetscReal elapsed = 0.0;
        PetscCall(PetscTime(&stage_start));
        PetscCall(fill_optimizer_load(uda, grid, ems_options,
                                      optimizer_options.load, load_case, b));
        PetscCall(elapsed_max_since(stage_start, &elapsed));
        iter_load_assembly_s += elapsed;
        timings.load_assembly_s += elapsed;
      }
      PetscCall(VecSet(u, 0.0));
      {
        PetscLogDouble stage_start = 0.0;
        PetscReal elapsed = 0.0;
        PetscCall(PetscTime(&stage_start));
        PetscCall(KSPSolve(ksp, b, u));
        PetscCall(elapsed_max_since(stage_start, &elapsed));
        iter_ksp_solve_s += elapsed;
        timings.ksp_solve_s += elapsed;
      }
      PetscCall(KSPGetIterationNumber(ksp, &case_its));
      PetscCall(KSPGetResidualNorm(ksp, &case_rnorm));
      KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
      PetscCall(KSPGetConvergedReason(ksp, &reason));
      its += case_its;
      rnorm_max = PetscMax(rnorm_max, case_rnorm);
      if (reason > 0) {
        ++converged_cases;
      } else {
        ++diverged_cases;
        PetscCall(PetscPrintf(
            PETSC_COMM_WORLD,
            "EMsFEM ANN load_case=%lld linear solve did not converge; skipping invalid sensitivity for this case\n",
            static_cast<long long>(load_case)));
        PetscCheck(have_last_good, PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED,
                   "First EMsFEM ANN linear solve did not converge (%lld diverged load case(s)); refusing to update the design with an invalid displacement field",
                   static_cast<long long>(diverged_cases));
        continue;
      }
      PetscCall(VecDot(b, u, &dot));
      {
        PetscLogDouble stage_start = 0.0;
        PetscReal elapsed = 0.0;
        PetscCall(PetscTime(&stage_start));
        PetscCall(compute_ems_fine_sensitivity(ctx, uda, fda, u, rho_phys, mask,
                                               dc_case, &case_volume));
        PetscCall(elapsed_max_since(stage_start, &elapsed));
        iter_sensitivity_s += elapsed;
        timings.sensitivity_s += elapsed;
      }
      PetscCall(VecAXPY(dc_phys, weight, dc_case));
      compliance += weight * PetscRealPart(dot);
      volume = case_volume;
    }
    rnorm = rnorm_max;
    PetscCheck(!optimizer_options.stability_guard || diverged_cases == 0 ||
                   have_last_good,
               PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED,
               "First EMsFEM ANN linear solve did not converge (%lld diverged load case(s)); refusing to update the design with an invalid displacement field",
               static_cast<long long>(diverged_cases));
    {
      PetscLogDouble stage_start = 0.0;
      PetscCall(PetscTime(&stage_start));
      PetscCall(apply_heaviside_sensitivity(fda, dc_phys, rho_filtered, mask,
                                            optimizer_options, current_beta,
                                            dc_projected));
      PetscCall(apply_sensitivity_filter_adjoint(fda, dc_projected, mask, filter_denom,
                                                 fine_filter_radius,
                                                 dc_design));
      PetscCall(elapsed_max_since(stage_start, &iter_sensitivity_filter_s));
      timings.sensitivity_filter_s += iter_sensitivity_filter_s;
    }
    PetscBool accepted = PETSC_TRUE;
    PetscReal raw_design_volume = 0.0;
    PetscCall(compute_design_mean(fda, rho_design, mask, &raw_design_volume));
    const PetscReal projected_volume_gap =
        diverged_cases > 0 ? 0.0 : volume - raw_design_volume;
    PetscReal effective_raw_volfrac = optimizer_options.volfrac;
    if (optimizer_options.projected_volume_correction) {
      effective_raw_volfrac = optimizer_options.volfrac - projected_volume_gap;
      effective_raw_volfrac =
          PetscMax(optimizer_options.rho_min,
                   PetscMin(1.0, effective_raw_volfrac));
    }
    const PetscBool compliance_spike =
        (!optimizer_options.projected_volume_correction &&
         have_last_good && accepted_compliance < PETSC_MAX_REAL &&
         compliance > accepted_compliance *
                           (1.0 + optimizer_options.max_compliance_increase))
            ? PETSC_TRUE
            : PETSC_FALSE;
    const PetscBool unstable =
        (optimizer_options.stability_guard &&
         (diverged_cases > 0 || compliance_spike))
            ? PETSC_TRUE
            : PETSC_FALSE;
    if (unstable && have_last_good && diverged_cases > 0 &&
        optimizer_options.projected_volume_correction) {
      effective_raw_volfrac = last_good_effective_raw_volfrac;
    }
    OptimizerOptions step_options = optimizer_options;
    step_options.volfrac = effective_raw_volfrac;
    PetscLogDouble optimizer_update_start = 0.0;
    PetscCall(PetscTime(&optimizer_update_start));
    if (unstable && have_last_good) {
      accepted = PETSC_FALSE;
      ++reject_streak;
      current_move = PetscMax(optimizer_options.move_min,
                              current_move * optimizer_options.move_shrink);
      step_options.move = current_move;
      PetscCall(VecCopy(rho_last_good, rho_design));
      PetscCall(oc_update(fda, rho_design, mask, dc_last_good, dv_design,
                          density_options, step_options, &change));
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "it=%lld rejected candidate: diverged_cases=%lld compliance_spike=%d; restored previous accepted design and retried OC with move=%.3e\n",
          static_cast<long long>(iter),
          static_cast<long long>(diverged_cases),
          static_cast<int>(compliance_spike),
          static_cast<double>(current_move)));
    } else {
      step_options.move = current_move;
      PetscCall(VecCopy(rho_design, rho_last_good));
      PetscCall(VecCopy(dc_design, dc_last_good));
      accepted_compliance = compliance;
      last_good_effective_raw_volfrac = effective_raw_volfrac;
      have_last_good = PETSC_TRUE;
      reject_streak = 0;
      PetscCall(oc_update(fda, rho_design, mask, dc_design, dv_design,
                          density_options, step_options, &change));
      if (optimizer_options.stability_guard && diverged_cases == 0) {
        current_move = PetscMin(optimizer_options.move,
                                current_move * optimizer_options.move_growth);
      }
    }
    PetscCall(elapsed_max_since(optimizer_update_start, &iter_optimizer_update_s));
    timings.optimizer_update_s += iter_optimizer_update_s;
    if (optimizer_options.write_checkpoint &&
        optimizer_options.checkpoint_interval > 0 &&
        iter % optimizer_options.checkpoint_interval == 0) {
      PetscLogDouble stage_start = 0.0;
      PetscCall(PetscTime(&stage_start));
      PetscCall(apply_density_filter(fda, rho_design, mask, filter_denom,
                                     fine_filter_radius,
                                     density_options.void_density, rho_filtered));
      PetscCall(apply_heaviside_projection(fda, rho_filtered, mask,
                                           optimizer_options, current_beta,
                                           density_options.void_density,
                                           rho_phys));
      PetscCall(apply_z_draft_closure(fda, mask, density_options.void_density,
                                      rho_phys));
      PetscCall(write_ems_checkpoint(optimizer_options, iter, PETSC_FALSE,
                                     rho_design, rho_phys, mask));
      PetscCall(elapsed_max_since(stage_start, &iter_checkpoint_s));
      timings.checkpoint_write_s += iter_checkpoint_s;
    }
    PetscCall(elapsed_max_since(iter_start, &iter_total_s));
    timings.iteration_total_s += iter_total_s;
    const PetscReal iter_fem_total_s =
        iter_load_assembly_s + iter_ksp_solve_s + iter_sensitivity_s;
    final_iter = iter;
    PetscCall(PetscViewerASCIIPrintf(hist,
                                     "%lld,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%lld,%.12e,%.12e,%lld,%lld,%d,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e,%.12e\n",
                                     static_cast<long long>(iter),
                                     static_cast<double>(compliance),
                                     static_cast<double>(volume),
                                     static_cast<double>(raw_design_volume),
                                     static_cast<double>(effective_raw_volfrac),
                                     static_cast<double>(projected_volume_gap),
                                     static_cast<double>(current_beta),
                                     static_cast<double>(change),
                                     static_cast<long long>(its),
                                     static_cast<double>(rnorm),
                                     static_cast<double>(step_options.move),
                                     static_cast<long long>(converged_cases),
                                     static_cast<long long>(diverged_cases),
                                     static_cast<int>(accepted),
                                     static_cast<double>(iter_total_s),
                                     static_cast<double>(iter_density_filter_s),
                                     static_cast<double>(iter_draft_closure_s),
                                     static_cast<double>(iter_ann_cache_s),
                                     static_cast<double>(iter_fine_material_sampling_s),
                                     static_cast<double>(iter_ann_shape_predict_s),
                                     static_cast<double>(iter_ems_matrix_assembly_s),
                                     static_cast<double>(iter_pc_setup_s),
                                     static_cast<double>(iter_load_assembly_s),
                                     static_cast<double>(iter_ksp_solve_s),
                                     static_cast<double>(iter_sensitivity_s),
                                     static_cast<double>(iter_sensitivity_filter_s),
                                     static_cast<double>(iter_optimizer_update_s),
                                     static_cast<double>(iter_checkpoint_s),
                                     static_cast<double>(iter_fem_total_s)));
    PetscCall(PetscViewerFlush(hist));
    objective_history.push_back({iter, compliance, volume});
    PetscCall(write_objective_volume_history(output_prefix, objective_history,
                                             optimizer_options.volfrac));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "it=%lld compliance=%.6e volume=%.6f raw=%.6f raw_target=%.6f beta=%.3g change=%.3e ksp_it=%lld move=%.3e accepted=%d time=%.3fs ann_cache=%.3fs ann_shape=%.3fs ems_ke=%.3fs linear_solve=%.3fs fem=%.3fs mma/oc=%.3fs\n",
                          static_cast<long long>(iter),
                          static_cast<double>(compliance),
                          static_cast<double>(volume),
                          static_cast<double>(raw_design_volume),
                          static_cast<double>(step_options.volfrac),
                          static_cast<double>(current_beta),
                          static_cast<double>(change),
                          static_cast<long long>(its),
                          static_cast<double>(step_options.move),
                          static_cast<int>(accepted),
                          static_cast<double>(iter_total_s),
                          static_cast<double>(iter_ann_cache_s),
                          static_cast<double>(iter_ann_shape_predict_s),
                          static_cast<double>(iter_ems_matrix_assembly_s),
                          static_cast<double>(iter_ksp_solve_s),
                          static_cast<double>(iter_fem_total_s),
                          static_cast<double>(iter_optimizer_update_s)));
    last_iteration_accepted = accepted;
    if (!accepted &&
        current_move <= optimizer_options.move_min * (1.0 + 1.0e-10) &&
        reject_streak >= 3) {
      PetscCall(VecCopy(rho_last_good, rho_design));
      change = 0.0;
      PetscCall(PetscPrintf(
          PETSC_COMM_WORLD,
          "Stopping early after %lld rejected minimum-move candidates; final design restored to last accepted state.\n",
          static_cast<long long>(reject_streak)));
      break;
    }
  }
  PetscCall(PetscViewerDestroy(&hist));

  {
    PetscLogDouble final_eval_start = 0.0;
    PetscCall(PetscTime(&final_eval_start));
    PetscInt its = 0;
    PetscReal rnorm = 0.0, rnorm_max = 0.0;
    if (!last_iteration_accepted && have_last_good) {
      PetscCall(VecCopy(rho_last_good, rho_design));
      change = 0.0;
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                            "Final evaluation uses the last accepted design, not the rejected candidate.\n"));
    }
    current_beta =
        heaviside_beta_for_iter(optimizer_options, PetscMax(1, final_iter));
    {
      PetscLogDouble stage_start = 0.0;
      PetscReal elapsed = 0.0;
      PetscCall(PetscTime(&stage_start));
      PetscCall(apply_density_filter(fda, rho_design, mask, filter_denom,
                                     fine_filter_radius,
                                     density_options.void_density, rho_filtered));
      PetscCall(apply_heaviside_projection(fda, rho_filtered, mask,
                                           optimizer_options, current_beta,
                                           density_options.void_density,
                                           rho_phys));
      PetscCall(elapsed_max_since(stage_start, &elapsed));
      timings.density_filter_s += elapsed;
    }
    {
      PetscLogDouble stage_start = 0.0;
      PetscReal elapsed = 0.0;
      PetscCall(PetscTime(&stage_start));
      PetscCall(apply_z_draft_closure(fda, mask, density_options.void_density,
                                      rho_phys));
      PetscCall(elapsed_max_since(stage_start, &elapsed));
      timings.draft_closure_s += elapsed;
    }
    {
      PetscLogDouble stage_start = 0.0;
      PetscReal elapsed = 0.0;
      EmsCacheTiming cache_timing;
      PetscCall(PetscTime(&stage_start));
      PetscCall(rebuild_local_k_cache_from_rho(ctx, fda, rho_phys,
                                               &cache_timing));
      PetscCall(elapsed_max_since(stage_start, &elapsed));
      timings.ann_cache_s += elapsed;
      timings.fine_material_sampling_s += cache_timing.material_sampling_s;
      timings.ann_shape_predict_s += cache_timing.ann_shape_predict_s;
      timings.ems_matrix_assembly_s += cache_timing.matrix_assembly_s;
    }
    {
      PetscLogDouble stage_start = 0.0;
      PetscReal elapsed = 0.0;
      PetscCall(PetscTime(&stage_start));
      if (ems_pc_type_uses_coarse_density_aux_matrix(ems_pc_type)) {
        PetscCall(build_ems_coarse_density(cda, fda, rho_phys, grid,
                                           ems_options.sub_n, coarse_rho));
      }
      PetscCall(setup_ems_preconditioner(ksp, A, P, uda, cda, coarse_rho,
                                         pc_ctx, ctx, ems_pc_type));
      PetscCall(elapsed_max_since(stage_start, &elapsed));
      timings.preconditioner_setup_s += elapsed;
    }
    compliance = 0.0;
    const PetscBool multi_case =
        (ems_options.control_arm_bc &&
         (ems_options.load_case <= 0 || ems_options.load_case > 3))
            ? PETSC_TRUE
            : PETSC_FALSE;
    const PetscInt first_case = multi_case ? 1 : ems_options.load_case;
    const PetscInt last_case = multi_case ? 3 : ems_options.load_case;
    for (PetscInt load_case = first_case; load_case <= last_case; ++load_case) {
      PetscScalar dot = 0.0;
      PetscReal case_volume = 0.0;
      PetscInt case_its = 0;
      PetscReal case_rnorm = 0.0;
      const PetscReal weight = multi_case ? control_arm_case_weight(load_case) : 1.0;
      {
        PetscLogDouble stage_start = 0.0;
        PetscReal elapsed = 0.0;
        PetscCall(PetscTime(&stage_start));
        PetscCall(fill_optimizer_load(uda, grid, ems_options,
                                      optimizer_options.load, load_case, b));
        PetscCall(elapsed_max_since(stage_start, &elapsed));
        timings.load_assembly_s += elapsed;
      }
      PetscCall(VecSet(u, 0.0));
      {
        PetscLogDouble stage_start = 0.0;
        PetscReal elapsed = 0.0;
        PetscCall(PetscTime(&stage_start));
        PetscCall(KSPSolve(ksp, b, u));
        PetscCall(elapsed_max_since(stage_start, &elapsed));
        timings.ksp_solve_s += elapsed;
      }
      PetscCall(KSPGetIterationNumber(ksp, &case_its));
      PetscCall(KSPGetResidualNorm(ksp, &case_rnorm));
      KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
      PetscCall(KSPGetConvergedReason(ksp, &reason));
      PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED,
                 "Final EMsFEM ANN linear solve did not converge for load_case=%lld",
                 static_cast<long long>(load_case));
      {
        PetscLogDouble stage_start = 0.0;
        PetscReal elapsed = 0.0;
        PetscCall(PetscTime(&stage_start));
        PetscCall(write_ems_displacement_checkpoint(optimizer_options,
                                                    multi_case, load_case, u));
        PetscCall(elapsed_max_since(stage_start, &elapsed));
        timings.checkpoint_write_s += elapsed;
      }
      PetscCall(VecDot(b, u, &dot));
      {
        PetscLogDouble stage_start = 0.0;
        PetscReal elapsed = 0.0;
        PetscCall(PetscTime(&stage_start));
        PetscCall(compute_ems_fine_sensitivity(ctx, uda, fda, u, rho_phys, mask,
                                               dc_case, &case_volume));
        PetscCall(elapsed_max_since(stage_start, &elapsed));
        timings.sensitivity_s += elapsed;
      }
      compliance += weight * PetscRealPart(dot);
      volume = case_volume;
      its += case_its;
      rnorm_max = PetscMax(rnorm_max, case_rnorm);
    }
    rnorm = rnorm_max;
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,
                          "final compliance=%.6e volume=%.6f ksp_it=%lld residual=%.3e\n",
                          static_cast<double>(compliance),
                          static_cast<double>(volume),
                          static_cast<long long>(its),
                          static_cast<double>(rnorm)));
    {
      PetscReal elapsed = 0.0;
      PetscCall(elapsed_max_since(final_eval_start, &elapsed));
      timings.final_eval_s += elapsed;
    }
  }

  if (optimizer_options.write_final_vtk) {
    PetscCheck((fine_nelx(grid, ems_options.sub_n) + 1) *
                   (fine_nely(grid, ems_options.sub_n) + 1) *
                   (fine_nelz(grid, ems_options.sub_n) + 1) <=
                   optimizer_options.vtk_max_points,
                PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE,
                "EMsFEM ANN fine-density VTK output exceeds -opt_vtk_max_points; disable -opt_write_final_vtk for production runs");
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    PetscCall(PetscTime(&stage_start));
    PetscCall(write_ems_opt_vtk(uda, fda, final_vtk_file, grid,
                                ems_options.sub_n, u, rho_phys, mask));
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.vtk_write_s += elapsed;
  }
  {
    PetscLogDouble stage_start = 0.0;
    PetscReal elapsed = 0.0;
    PetscCall(PetscTime(&stage_start));
    PetscCall(write_ems_checkpoint(optimizer_options, final_iter, PETSC_TRUE,
                                   rho_design, rho_phys, mask));
    PetscCall(elapsed_max_since(stage_start, &elapsed));
    timings.checkpoint_write_s += elapsed;
  }
  PetscCall(elapsed_max_since(topopt_start, &timings.total_s));
  PetscReal final_raw_design_volume = 0.0;
  PetscCall(compute_design_mean(fda, rho_design, mask, &final_raw_design_volume));
  PetscCall(write_ems_summary(output_prefix, grid, density_options,
                              optimizer_options, ems_options,
                              final_iter, compliance, volume,
                              final_raw_design_volume, change, timings));

  PetscCall(KSPDestroy(&ksp));
  delete pc_ctx;
  pc_ctx = nullptr;
  PetscCall(MatDestroy(&P));
  PetscCall(MatDestroy(&A));
  uda = nullptr;
  PetscCall(VecDestroy(&dc_last_good));
  PetscCall(VecDestroy(&dc_design));
  PetscCall(VecDestroy(&dc_projected));
  PetscCall(VecDestroy(&dc_case));
  PetscCall(VecDestroy(&dc_phys));
  PetscCall(VecDestroy(&dv_design));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&filter_denom));
  PetscCall(VecDestroy(&rho_last_good));
  PetscCall(VecDestroy(&rho_phys));
  PetscCall(VecDestroy(&rho_filtered));
  PetscCall(VecDestroy(&rho_design));
  PetscCall(VecDestroy(&mask));
  PetscCall(VecDestroy(&coarse_rho));
  PetscCall(DMDestroy(&cda));
  PetscCall(DMDestroy(&fda));
  return 0;
}

} // namespace control_arm
