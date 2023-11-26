#include "maxpool.h"
#include "snrt.h"

#include "data_all.h"

#define ENABLE_1D 1
#define ENABLE_2D 1
#define ENABLE_3D 1

#define ENABLE_YES_INDICES 0
#define ENABLE_NO_INDICES 1

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

  // compute_output_shape(&attr1, attr1.output_shape);
  // maxpool_f64_1d_no_index(&attr1,
  //               ifmap1,
  //               output_loc1);

  // compute_output_shape(&attr2, attr2.output_shape);
  // maxpool_f64_2d_no_index(&attr2,
  //               ifmap2,
  //               output_loc2);

  // compute_output_shape(&attr3, attr3.output_shape);
  // maxpool_f64_3d_no_index(&attr3,
  //               ifmap3,
  //               output_loc3);


  #if ENABLE_NO_INDICES

  #if ENABLE_1D
  compute_output_shape(&attr1_1, attr1_1.output_shape);
  maxpool_f64_1d_no_index(&attr1_1,
                ifmap1_1,
                output_loc1_1);
  #endif

  #if ENABLE_2D
  compute_output_shape(&attr2_1, attr2_1.output_shape);
  maxpool_f64_2d_no_index(&attr2_1,
                ifmap2_1,
                output_loc2_1);
  #endif

  #if ENABLE_3D
  compute_output_shape(&attr3_1, attr3_1.output_shape);
  maxpool_f64_3d_no_index(&attr3_1,
                ifmap3_1,
                output_loc3_1);
  #endif


  #if ENABLE_1D
  compute_output_shape(&attr1_2, attr1_2.output_shape);
  maxpool_f64_1d_no_index(&attr1_2,
                ifmap1_2,
                output_loc1_2);
  #endif

  #if ENABLE_2D
  compute_output_shape(&attr2_2, attr2_2.output_shape);
  maxpool_f64_2d_no_index(&attr2_2,
                ifmap2_2,
                output_loc2_2);
  #endif

  #if ENABLE_3D
  compute_output_shape(&attr3_2, attr3_2.output_shape);
  maxpool_f64_3d_no_index(&attr3_2,
                ifmap3_2,
                output_loc3_2);
  #endif


  #if ENABLE_1D
  compute_output_shape(&attr1_3, attr1_3.output_shape);
  maxpool_f64_1d_no_index(&attr1_3,
                ifmap1_3,
                output_loc1_3);
  #endif

  #if ENABLE_2D
  compute_output_shape(&attr2_3, attr2_3.output_shape);
  maxpool_f64_2d_no_index(&attr2_3,
                ifmap2_3,
                output_loc2_3);
  #endif

  #if ENABLE_3D
  compute_output_shape(&attr3_3, attr3_3.output_shape);
  maxpool_f64_3d_no_index(&attr3_3,
                ifmap3_3,
                output_loc3_3);
  #endif

  #endif



  #if ENABLE_YES_INDICES

  #if ENABLE_1D
  compute_output_shape(&attr1_1, attr1_1.output_shape);
  maxpool_f64_1d_with_index(&attr1_1,
                ifmap1_1,
                output_loc1_1,
                idx_loc1_1);
  #endif

  #if ENABLE_2D
  compute_output_shape(&attr2_1, attr2_1.output_shape);
  if (attr2_1.storage_order) {
    maxpool_f64_2d_with_index_col_major(&attr2_1,
                ifmap2_1,
                output_loc2_1,
                idx_loc2_1);
  }
  else {
    maxpool_f64_2d_with_index_row_major(&attr2_1,
                ifmap2_1,
                output_loc2_1,
                idx_loc2_1);
  }
  #endif
  
  #if ENABLE_3D
  compute_output_shape(&attr3_1, attr3_1.output_shape);
  if (attr3_1.storage_order) {
    maxpool_f64_3d_with_index_col_major(&attr3_1,
                ifmap3_1,
                output_loc3_1,
                idx_loc3_1);
  }
  else {
    maxpool_f64_3d_with_index_row_major(&attr3_1,
                ifmap3_1,
                output_loc3_1,
                idx_loc3_1);
  }
  #endif


  #if ENABLE_1D
  compute_output_shape(&attr1_2, attr1_2.output_shape);
  maxpool_f64_1d_with_index(&attr1_2,
                ifmap1_2,
                output_loc1_2,
                idx_loc1_2);
  #endif

  #if ENABLE_2D
  compute_output_shape(&attr2_2, attr2_2.output_shape);
  if (attr2_2.storage_order) {
    maxpool_f64_2d_with_index_col_major(&attr2_2,
                ifmap2_2,
                output_loc2_2,
                idx_loc2_2);
  }
  else {
    maxpool_f64_2d_with_index_row_major(&attr2_2,
                ifmap2_2,
                output_loc2_2,
                idx_loc2_2);
  }
  #endif

  #if ENABLE_3D
  compute_output_shape(&attr3_2, attr3_2.output_shape);
  if (attr3_2.storage_order) {
    maxpool_f64_3d_with_index_col_major(&attr3_2,
                ifmap3_2,
                output_loc3_2,
                idx_loc3_2);
  }
  else {
    maxpool_f64_3d_with_index_row_major(&attr3_2,
                ifmap3_2,
                output_loc3_2,
                idx_loc3_2);
  }
  #endif


  #if ENABLE_1D
  compute_output_shape(&attr1_3, attr1_3.output_shape);
  maxpool_f64_1d_with_index(&attr1_3,
                ifmap1_3,
                output_loc1_3,
                idx_loc1_3);
  #endif

  #if ENABLE_2D
  compute_output_shape(&attr2_3, attr2_3.output_shape);
  if (attr2_3.storage_order) {
    maxpool_f64_2d_with_index_col_major(&attr2_3,
                ifmap2_3,
                output_loc2_3,
                idx_loc2_3);
  }
  else {
    maxpool_f64_2d_with_index_row_major(&attr2_3,
                ifmap2_3,
                output_loc2_3,
                idx_loc2_3);
  }
  #endif

  #if ENABLE_3D
  compute_output_shape(&attr3_3, attr3_3.output_shape);
  if (attr3_3.storage_order) {
    maxpool_f64_3d_with_index_col_major(&attr3_3,
                ifmap3_3,
                output_loc3_3,
                idx_loc3_3);
  }
  else {
    maxpool_f64_3d_with_index_row_major(&attr3_3,
                ifmap3_3,
                output_loc3_3,
                idx_loc3_3);
  }
  #endif

  #endif

  return 0;

}