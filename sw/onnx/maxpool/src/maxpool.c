#include "maxpool.h"
#include "snrt.h"
// #include "math.h"
// #include "printf.h"

// TODO: Replace with better impls? Problems with using math.h...
static inline double floor(double x) {
  return (double) ((int) x);
}

// https://stackoverflow.com/a/8377539
static inline double ceil(double num) {
  int inum = (int)num;
  if (num == (double)inum) {
      return num;
  }
  return inum + 1;
}

static inline int min(int a, int b) {
  return a < b ? a : b;
}

void populate_defaults(maxpool_attributes* attr, int n_dim) {
  attr->n_dim = n_dim;
  attr->auto_pad = NOTSET;
  attr->ceil_mode = 0;
  for (int i = 0; i < n_dim; ++i) {
    attr->dilations[i] = 1;
    attr->pads[i] = 0;
    attr->pads[i + n_dim] = 0;
    attr->strides[i] = 1;
  }
  attr->storage_order = 0;
}

// also recomputes padding if auto_pad is set (deprecated)
void compute_output_shape(maxpool_attributes* attr, int* output_shape) {
  output_shape[0] = attr->input_shape[0];
  output_shape[1] = attr->input_shape[1];
  
  int* input_spatial_shape = attr->input_shape + 2;
  int* output_spatial_shape = output_shape + 2;

  // TODO: the reference impl has different control flow, but it seems redundant?
  if (attr->auto_pad != NOTSET) {

    if (attr->auto_pad == SAME_UPPER || attr->auto_pad == SAME_LOWER) {

      for (int i = 0; i < attr->n_dim; ++i) {
        if (attr->auto_pad == SAME_UPPER) {
          output_spatial_shape[i] = (int) (
            ceil((double) input_spatial_shape[i] / attr->strides[i])
          );
        }
        else {
          output_spatial_shape[i] = (int) (
            floor((double) input_spatial_shape[i] / attr->strides[i])
          );
        }

        int pad_i = (
          (output_spatial_shape[i] - 1) * attr->strides[i]
          + ((attr->kernel_shape[i] - 1) * attr->dilations[i] + 1)
          - input_spatial_shape[i]
        );
        attr->pads[i] = (int) (pad_i / 2);
        attr->pads[i + attr->n_dim] = pad_i - attr->pads[i];
      }

    }
    else { // padding_type::VALID
      // TODO: this computation is inconsistent with the docs? no cases for ceil/floor mode
      for (int i = 0; i < attr->n_dim; ++i) {
        output_spatial_shape[i] = (int) (
          ceil(
            (double) (input_spatial_shape[i] - ((attr->kernel_shape[i] - 1) * attr->dilations[i] + 1) + 1) / attr->strides[i]
          )
        );
      }
    }
    
  }
  else {

    if (attr->ceil_mode) {

      for (int i = 0; i < attr->n_dim; ++i) {
        output_spatial_shape[i] = (int) (
          ceil(
            (double) (input_spatial_shape[i] + attr->pads[i] + attr->pads[i + attr->n_dim] - ((attr->kernel_shape[i] - 1) * attr->dilations[i] + 1))
            / attr->strides[i] + 1
          )
        );
      }

    }
    else {

      for (int i = 0; i < attr->n_dim; ++i) {
        output_spatial_shape[i] = (int) (
          floor(
            (double) (input_spatial_shape[i] + attr->pads[i] + attr->pads[i + attr->n_dim] - ((attr->kernel_shape[i] - 1) * attr->dilations[i] + 1))
            / attr->strides[i] + 1
          )
        );
      }

    }

  }

}

// void maxpool_f64_1d_with_index_row_major(maxpool_attributes* attr, double* in, double* out, int* idx) {
//   #define MAXPOOL_DIM 1
//   #define MAXPOOL_ROW_MAJOR
//   maxpool_fp64_layer(attr, in, out, idx);
//   #undef MAXPOOL_DIM
//   #undef MAXPOOL_ROW_MAJOR
// }

// void maxpool_f64_1d_with_index_col_major(maxpool_attributes* attr, double* in, double* out, int* idx) {
//   #define MAXPOOL_DIM 1
//   #define MAXPOOL_COL_MAJOR
//   maxpool_fp64_layer(attr, in, out, idx);
//   #undef MAXPOOL_DIM
//   #undef MAXPOOL_COL_MAJOR
// }

#define MAXPOOL_FN _maxpool_f64_1d_with_index
#define MAXPOOL_DIM 1
#define MAXPOOL_ROW_MAJOR
#include "_maxpool.c"
void maxpool_f64_1d_with_index(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN(attr, in, out, idx);
}
#undef MAXPOOL_FN
#undef MAXPOOL_DIM
#undef MAXPOOL_ROW_MAJOR

#define MAXPOOL_FN _maxpool_f64_1d_no_index
#define MAXPOOL_DIM 1
#include "_maxpool.c"
void maxpool_f64_1d_no_index(maxpool_attributes* attr, double* in, double* out) {
  MAXPOOL_FN(attr, in, out, NULL);
}
#undef MAXPOOL_FN
#undef MAXPOOL_DIM

#define MAXPOOL_FN _maxpool_f64_2d_with_index_row_major
#define MAXPOOL_DIM 2
#define MAXPOOL_ROW_MAJOR
#include "_maxpool.c"
void maxpool_f64_2d_with_index_row_major(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN(attr, in, out, idx);
}
#undef MAXPOOL_FN
#undef MAXPOOL_DIM
#undef MAXPOOL_ROW_MAJOR

#define MAXPOOL_FN _maxpool_f64_2d_with_index_col_major
#define MAXPOOL_DIM 2
#define MAXPOOL_COL_MAJOR
#include "_maxpool.c"
void maxpool_f64_2d_with_index_col_major(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN(attr, in, out, idx);
}
#undef MAXPOOL_FN
#undef MAXPOOL_DIM
#undef MAXPOOL_COL_MAJOR

#define MAXPOOL_FN _maxpool_f64_2d_no_index
#define MAXPOOL_DIM 2
#include "_maxpool.c"
void maxpool_f64_2d_no_index(maxpool_attributes* attr, double* in, double* out) {
  MAXPOOL_FN(attr, in, out, NULL);
}
#undef MAXPOOL_FN
#undef MAXPOOL_DIM

#define MAXPOOL_FN _maxpool_f64_3d_with_index_row_major
#define MAXPOOL_DIM 3
#define MAXPOOL_ROW_MAJOR
#include "_maxpool.c"
void maxpool_f64_3d_with_index_row_major(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN(attr, in, out, idx);
}
#undef MAXPOOL_FN
#undef MAXPOOL_DIM
#undef MAXPOOL_ROW_MAJOR

#define MAXPOOL_FN _maxpool_f64_3d_with_index_col_major
#define MAXPOOL_DIM 3
#define MAXPOOL_COL_MAJOR
#include "_maxpool.c"
void maxpool_f64_3d_with_index_col_major(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN(attr, in, out, idx);
}
#undef MAXPOOL_FN
#undef MAXPOOL_DIM
#undef MAXPOOL_COL_MAJOR

#define MAXPOOL_FN _maxpool_f64_3d_no_index
#define MAXPOOL_DIM 3
#include "_maxpool.c"
void maxpool_f64_3d_no_index(maxpool_attributes* attr, double* in, double* out) {
  MAXPOOL_FN(attr, in, out, NULL);
}
#undef MAXPOOL_FN
#undef MAXPOOL_DIM



#define MAXPOOL_FN_SINGLE_CORE _maxpool_f64_1d_with_index_single_core
#define MAXPOOL_DIM 1
#define MAXPOOL_ROW_MAJOR
#include "_maxpool_single_core.c"
void maxpool_f64_1d_with_index_single_core(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN_SINGLE_CORE(attr, in, out, idx);
}
#undef MAXPOOL_FN_SINGLE_CORE
#undef MAXPOOL_DIM
#undef MAXPOOL_ROW_MAJOR

#define MAXPOOL_FN_SINGLE_CORE _maxpool_f64_1d_no_index_single_core
#define MAXPOOL_DIM 1
#include "_maxpool_single_core.c"
void maxpool_f64_1d_no_index_single_core(maxpool_attributes* attr, double* in, double* out) {
  MAXPOOL_FN_SINGLE_CORE(attr, in, out, NULL);
}
#undef MAXPOOL_FN_SINGLE_CORE
#undef MAXPOOL_DIM

#define MAXPOOL_FN_SINGLE_CORE _maxpool_f64_2d_with_index_row_major_single_core
#define MAXPOOL_DIM 2
#define MAXPOOL_ROW_MAJOR
#include "_maxpool_single_core.c"
void maxpool_f64_2d_with_index_row_major_single_core(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN_SINGLE_CORE(attr, in, out, idx);
}
#undef MAXPOOL_FN_SINGLE_CORE
#undef MAXPOOL_DIM
#undef MAXPOOL_ROW_MAJOR

#define MAXPOOL_FN_SINGLE_CORE _maxpool_f64_2d_with_index_col_major_single_core
#define MAXPOOL_DIM 2
#define MAXPOOL_COL_MAJOR
#include "_maxpool_single_core.c"
void maxpool_f64_2d_with_index_col_major_single_core(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN_SINGLE_CORE(attr, in, out, idx);
}
#undef MAXPOOL_FN_SINGLE_CORE
#undef MAXPOOL_DIM
#undef MAXPOOL_COL_MAJOR

#define MAXPOOL_FN_SINGLE_CORE _maxpool_f64_2d_no_index_single_core
#define MAXPOOL_DIM 2
#include "_maxpool_single_core.c"
void maxpool_f64_2d_no_index_single_core(maxpool_attributes* attr, double* in, double* out) {
  MAXPOOL_FN_SINGLE_CORE(attr, in, out, NULL);
}
#undef MAXPOOL_FN_SINGLE_CORE
#undef MAXPOOL_DIM

#define MAXPOOL_FN_SINGLE_CORE _maxpool_f64_3d_with_index_row_major_single_core
#define MAXPOOL_DIM 3
#define MAXPOOL_ROW_MAJOR
#include "_maxpool_single_core.c"
void maxpool_f64_3d_with_index_row_major_single_core(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN_SINGLE_CORE(attr, in, out, idx);
}
#undef MAXPOOL_FN_SINGLE_CORE
#undef MAXPOOL_DIM
#undef MAXPOOL_ROW_MAJOR

#define MAXPOOL_FN_SINGLE_CORE _maxpool_f64_3d_with_index_col_major_single_core
#define MAXPOOL_DIM 3
#define MAXPOOL_COL_MAJOR
#include "_maxpool_single_core.c"
void maxpool_f64_3d_with_index_col_major_single_core(maxpool_attributes* attr, double* in, double* out, int* idx) {
  MAXPOOL_FN_SINGLE_CORE(attr, in, out, idx);
}
#undef MAXPOOL_FN_SINGLE_CORE
#undef MAXPOOL_DIM
#undef MAXPOOL_COL_MAJOR

#define MAXPOOL_FN_SINGLE_CORE _maxpool_f64_3d_no_index_single_core
#define MAXPOOL_DIM 3
#include "_maxpool_single_core.c"
void maxpool_f64_3d_no_index_single_core(maxpool_attributes* attr, double* in, double* out) {
  MAXPOOL_FN_SINGLE_CORE(attr, in, out, NULL);
}
#undef MAXPOOL_FN_SINGLE_CORE
#undef MAXPOOL_DIM

void maxpool_backprop_f64_1d(maxpool_attributes* attr, double* vals, int* idx, double* out) {

  int total_ins = attr->input_shape[0] * attr->input_shape[1] * attr->input_shape[2];
  if (snrt_is_dm_core()) {
    snrt_dma_memset(out, 0.0, total_ins * sizeof(double));

    snrt_dma_wait_all();
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
  }
  if (snrt_is_compute_core()) {
    int n_channels = attr->input_shape[0] * attr->input_shape[1];
    int outs_per_channel = attr->output_shape[2];
    snrt_cluster_hw_barrier();

    for (int i = 0; i < n_channels; ++i) {
      for (int j = 0; j < outs_per_channel; ++j) {
        int index = i * outs_per_channel + j;
        out[idx[index]] = vals[index];
      }
    }

    snrt_cluster_hw_barrier();
  }

}

void maxpool_backprop_f64_2d(maxpool_attributes* attr, double* vals, int* idx, double* out) {

  int total_ins = attr->input_shape[0] * attr->input_shape[1] * attr->input_shape[2] * attr->input_shape[3];
  if (snrt_is_dm_core()) {
    snrt_dma_memset(out, 0.0, total_ins * sizeof(double));

    snrt_dma_wait_all();
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
  }
  if (snrt_is_compute_core()) {
    int n_channels = attr->input_shape[0] * attr->input_shape[1];
    int outs_per_channel = attr->output_shape[2] * attr->output_shape[3];
    snrt_cluster_hw_barrier();

    for (int i = 0; i < n_channels; ++i) {
      for (int j = 0; j < outs_per_channel; ++j) {
        int index = i * outs_per_channel + j;
        out[idx[index]] = vals[index];
      }
    }

    snrt_cluster_hw_barrier();
  }

}

void maxpool_backprop_f64_3d(maxpool_attributes* attr, double* vals, int* idx, double* out) {

  int total_ins = attr->input_shape[0] * attr->input_shape[1] * attr->input_shape[2] * attr->input_shape[3] * attr->input_shape[4];
  if (snrt_is_dm_core()) {
    snrt_dma_memset(out, 0.0, total_ins * sizeof(double));

    snrt_dma_wait_all();
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
  }
  if (snrt_is_compute_core()) {
    int n_channels = attr->input_shape[0] * attr->input_shape[1];
    int outs_per_channel = attr->output_shape[2] * attr->output_shape[3] * attr->output_shape[4];
    snrt_cluster_hw_barrier();

    for (int i = 0; i < n_channels; ++i) {
      for (int j = 0; j < outs_per_channel; ++j) {
        int index = i * outs_per_channel + j;
        out[idx[index]] = vals[index];
      }
    }

    snrt_cluster_hw_barrier();
  }

}
