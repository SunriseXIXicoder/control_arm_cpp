#include "control_arm/ann_model.hpp"

#include <petscmath.h>

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

namespace control_arm {
namespace {

PetscErrorCode read_text_file(const std::string &path, std::string *text) {
  std::ifstream in(path, std::ios::in | std::ios::binary);
  PetscCheck(in.good(), PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
             "Cannot open ANN JSON file: %s", path.c_str());
  std::ostringstream buffer;
  buffer << in.rdbuf();
  *text = buffer.str();
  return 0;
}

std::vector<PetscReal> scan_numbers(const std::string &text) {
  std::vector<PetscReal> values;
  const char *ptr = text.c_str();
  char *end = nullptr;

  while (*ptr != '\0') {
    if (*ptr == '-' || *ptr == '+' || *ptr == '.' || std::isdigit(*ptr)) {
      errno = 0;
      const double value = std::strtod(ptr, &end);
      if (end != ptr && errno == 0) {
        values.push_back(static_cast<PetscReal>(value));
        ptr = end;
        continue;
      }
    }
    ++ptr;
  }
  return values;
}

std::vector<std::string> scan_json_strings(const std::string &text) {
  std::vector<std::string> values;
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] != '"') {
      continue;
    }
    std::string value;
    ++i;
    while (i < text.size() && text[i] != '"') {
      if (text[i] == '\\' && i + 1 < text.size()) {
        ++i;
      }
      value.push_back(text[i]);
      ++i;
    }
    values.push_back(value);
  }
  return values;
}

PetscReal activate(PetscReal x, const std::string &kind) {
  if (kind == "sigmoid") {
    return 1.0 / (PetscExpReal(-x) + 1.0);
  }
  if (kind == "relu") {
    return PetscMax(0.0, x);
  }
  if (kind == "tanh") {
    return PetscTanhReal(x);
  }
  if (kind == "elu") {
    return x >= 0.0 ? x : PetscExpReal(x) - 1.0;
  }
  return x;
}

PetscErrorCode load_network(const std::string &weight_path,
                            const std::vector<std::string> &activations,
                            const std::vector<PetscInt> &dims,
                            AnnNetwork *network) {
  std::string text;
  PetscCall(read_text_file(weight_path, &text));
  const std::vector<PetscReal> numbers = scan_numbers(text);
  PetscInt input_dim = 125;
  std::size_t offset = 0;

  network->layers.clear();
  network->layers.reserve(dims.size());
  std::vector<std::string> layer_activations = activations;
  while (layer_activations.size() < dims.size()) {
    layer_activations.push_back("");
  }

  for (std::size_t layer_id = 0; layer_id < dims.size(); ++layer_id) {
    AnnLayer layer;
    layer.in = input_dim;
    layer.out = dims[layer_id];
    layer.activation = layer_activations[layer_id] == "null" ? "" : layer_activations[layer_id];
    const std::size_t wcount =
        static_cast<std::size_t>(layer.in) * static_cast<std::size_t>(layer.out);
    const std::size_t bcount = static_cast<std::size_t>(layer.out);
    PetscCheck(offset + wcount + bcount <= numbers.size(), PETSC_COMM_SELF,
               PETSC_ERR_FILE_UNEXPECTED,
               "ANN weight file has too few numeric entries: %s", weight_path.c_str());
    layer.weight.assign(numbers.begin() + static_cast<std::ptrdiff_t>(offset),
                        numbers.begin() + static_cast<std::ptrdiff_t>(offset + wcount));
    offset += wcount;
    layer.bias.assign(numbers.begin() + static_cast<std::ptrdiff_t>(offset),
                      numbers.begin() + static_cast<std::ptrdiff_t>(offset + bcount));
    offset += bcount;
    input_dim = layer.out;
    network->layers.push_back(std::move(layer));
  }

  PetscCheck(offset == numbers.size(), PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "ANN weight file has unexpected numeric tail: %s", weight_path.c_str());
  return 0;
}

} // namespace

PetscErrorCode AnnNetwork::forward(const std::vector<PetscReal> &input,
                                   std::vector<PetscReal> *output) const {
  std::vector<PetscReal> current = input;
  std::vector<PetscReal> next;

  for (const AnnLayer &layer : layers) {
    PetscCheck(static_cast<PetscInt>(current.size()) == layer.in, PETSC_COMM_SELF,
               PETSC_ERR_ARG_SIZ,
               "ANN input size mismatch: expected %lld, got %zu",
               static_cast<long long>(layer.in), current.size());
    next.assign(static_cast<std::size_t>(layer.out), 0.0);
    for (PetscInt j = 0; j < layer.out; ++j) {
      PetscReal value = layer.bias[static_cast<std::size_t>(j)];
      for (PetscInt i = 0; i < layer.in; ++i) {
        value += current[static_cast<std::size_t>(i)] *
                 layer.weight[static_cast<std::size_t>(i) *
                                  static_cast<std::size_t>(layer.out) +
                              static_cast<std::size_t>(j)];
      }
      next[static_cast<std::size_t>(j)] = activate(value, layer.activation);
    }
    current.swap(next);
  }

  *output = std::move(current);
  return 0;
}

PetscErrorCode AnnShapeModel::load(const char *directory, PetscInt subcell_count) {
  sub_n = subcell_count;
  inside_dofs = 3 * (sub_n - 1) * (sub_n - 1) * (sub_n - 1);

  std::string func_text;
  const std::string base(directory);
  const std::string func_path =
      base + "/model_function_3DEMs_FEMBLC_20220821_input_5_layer1.json";
  PetscCall(read_text_file(func_path, &func_text));

  std::vector<std::string> activations = scan_json_strings(func_text);
  std::vector<PetscReal> raw_dims = scan_numbers(func_text);
  std::vector<PetscInt> dims;
  dims.reserve(raw_dims.size());
  for (PetscReal value : raw_dims) {
    dims.push_back(static_cast<PetscInt>(value + 0.5));
  }
  PetscCheck(!dims.empty(), PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED,
             "ANN function file has no layer dimensions: %s", func_path.c_str());
  PetscCheck(dims.back() == 1008, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP,
             "This ANN/EMsFEM path expects each shape network to output 1008 values");

  networks.assign(4, AnnNetwork{});
  for (PetscInt i = 0; i < 4; ++i) {
    const std::string weight_path =
        base + "/model_weight_3DEMs_FEMBLC_20220821_input_5_layer" +
        std::to_string(static_cast<long long>(i + 1)) + ".json";
    PetscCall(load_network(weight_path, activations, dims,
                           &networks[static_cast<std::size_t>(i)]));
  }
  return 0;
}

PetscErrorCode AnnShapeModel::predict_inside_shape(
    const std::vector<PetscReal> &material,
    std::vector<PetscReal> *inside_shape) const {
  PetscCheck(static_cast<PetscInt>(material.size()) == sub_n * sub_n * sub_n,
             PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
             "ANN material input must contain sub_n^3 entries");

  std::vector<PetscReal> concatenated;
  concatenated.reserve(4032);
  for (const AnnNetwork &network : networks) {
    std::vector<PetscReal> out;
    PetscCall(network.forward(material, &out));
    concatenated.insert(concatenated.end(), out.begin(), out.end());
  }
  PetscCheck(concatenated.size() ==
                 static_cast<std::size_t>(inside_dofs) * 21u,
             PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ,
             "ANN shape output size mismatch");

  inside_shape->assign(static_cast<std::size_t>(inside_dofs) * 24u, 0.0);
  for (PetscInt row = 0; row < inside_dofs; ++row) {
    PetscReal sx = 0.0, sy = 0.0, sz = 0.0;
    for (PetscInt col = 0; col < 21; ++col) {
      const PetscReal value =
          concatenated[static_cast<std::size_t>(row) * 21u +
                       static_cast<std::size_t>(col)];
      (*inside_shape)[static_cast<std::size_t>(row) * 24u +
                      static_cast<std::size_t>(col)] = value;
      if (col % 3 == 0) sx += value;
      if (col % 3 == 1) sy += value;
      if (col % 3 == 2) sz += value;
    }
    const PetscInt component = row % 3;
    (*inside_shape)[static_cast<std::size_t>(row) * 24u + 21u] =
        (component == 0 ? 1.0 : 0.0) - sx;
    (*inside_shape)[static_cast<std::size_t>(row) * 24u + 22u] =
        (component == 1 ? 1.0 : 0.0) - sy;
    (*inside_shape)[static_cast<std::size_t>(row) * 24u + 23u] =
        (component == 2 ? 1.0 : 0.0) - sz;
  }
  return 0;
}

} // namespace control_arm
