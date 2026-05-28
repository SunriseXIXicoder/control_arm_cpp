#pragma once

#include <petscsys.h>

#include <string>
#include <vector>

namespace control_arm {

struct AnnLayer {
  PetscInt in = 0;
  PetscInt out = 0;
  std::string activation;
  std::vector<PetscReal> weight;
  std::vector<PetscReal> bias;
};

struct AnnNetwork {
  std::vector<AnnLayer> layers;

  PetscErrorCode forward(const std::vector<PetscReal> &input,
                         std::vector<PetscReal> *output) const;
};

struct AnnShapeModel {
  PetscInt sub_n = 5;
  PetscInt inside_dofs = 0;
  std::vector<AnnNetwork> networks;

  PetscErrorCode load(const char *directory, PetscInt subcell_count = 5);
  PetscErrorCode predict_inside_shape(const std::vector<PetscReal> &material,
                                      std::vector<PetscReal> *inside_shape) const;
};

} // namespace control_arm
