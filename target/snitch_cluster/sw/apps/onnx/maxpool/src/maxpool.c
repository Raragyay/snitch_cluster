#include "maxpool.h"
#include "snrt.h"

#include "tests.h"
#include "printf.h"
#include "data.h"

/*
typedef struct maxpool_attributes_struct {
  int n_dim; // excludes batch size and # channels
  int[5] input_shape; // INCLUDES batch size and # channels, 3d worst case
  int[5] output_shape; // INCLUDES batch size and # channels, 3d worst case

  enum padding_type auto_pad; // NOTSET, deprecated in maxpool v12
  bool ceil_mode; // false
  int[3] dilations; // [1...1]
  int[3] kernel_shape; // required
  int[6] pads; // [0...0], [dim1_begin, dim2_begin, dim1_end, dim2_end]
  bool storage_order; // false (row major)
  int[3] strides; // [1...1]
} maxpool_attributes;

void maxpool_fp64(maxpool_attributes* attr, double* in, double* out, double* idx);

void populate_defaults(maxpool_attributes* attr, int n_dim);
*/

int main() {

  uint32_t cluster_num = snrt_cluster_num();
  uint32_t cluster_id = snrt_cluster_idx();
  uint32_t compute_num = snrt_cluster_compute_core_num();
  uint32_t compute_id = snrt_global_core_idx();

  if (compute_id != 1) return 0;

  maxpool_fp64(&attr2,
                ifmap2,
                output_loc2,
                NULL);

  return 0;

  // return errors;
}