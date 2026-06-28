#pragma once

#include "control_arm/low_order_optimizer.hpp"

#include <petscdmda.h>

#include <vector>

namespace control_arm {

struct DraftDirection {
  PetscInt axis = 2;
  PetscInt sign = 0;
};

const char *draft_axes_error_message();

PetscErrorCode parse_draft_axes(const char *text,
                                std::vector<DraftDirection> *dirs);

PetscErrorCode collect_effective_draft_axes(
    const OptimizerOptions &options, std::vector<DraftDirection> *effective);

PetscErrorCode apply_axis_draft_closure(DM da, Vec mask, PetscReal eta,
                                        PetscInt axis, PetscInt sign, Vec rho);

PetscErrorCode apply_draft_closure(DM da, Vec mask,
                                   const OptimizerOptions &options, Vec rho);

} // namespace control_arm
