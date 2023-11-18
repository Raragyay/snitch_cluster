#include "maxpool.h"
#include "snrt.h"
// #include "math.h"
#include "printf.h"

#define DMA_ATTRIBS 1
#define DMA_INDICES 1
#define USE_SSR_FREP_1D 1
#define USE_SSR_FREP_2D 1
#define USE_SSR_FREP_3D 0
#define USE_SSR_FREP_ALL 0

// #define ssr_asm_with_index(out_idx, total_iter) \
// asm volatile( \
//   "li t0, 0\n" /* counter, start at one because initial val has idx 0 */ \
//   "li t1, 0\n" /* cur max idx */ \
//   "fadd.d ft3, %[zero], ft0\n" /* ft3 = cur max val */ \
//   /* begin loop */ \
//   "addi t0, t0, 1\n" \
//   "fmax.d ft4, ft3, ft0\n" \
//   "feq.d t3, ft4, ft3\n" \
//   "bne t3, zero, 12\n" /* branch if no update needed */ \
//   "fadd.d ft3, %[zero], ft4\n" /* update cur max val */ \
//   "add t1, zero, t0\n" /* update cur max idx */ \
//   "bne t0, %[n_iter], -24\n" \
//   "fadd.d ft1, %[zero], ft3\n" \
//   "addi %[idx_out], t1, 0\n" /* write out max idx to memory */ \
//   : [idx_out] "=r"(out_idx) \
//   : [zero] "f"(0.0), [n_iter] "r"(total_iter) \
//   : "t0", "t1", "t3", "ft0", "ft1", "ft2", "ft3", "ft4", "memory", "zero" \
// )

// #define ssr_asm_no_index(total_iter) \
// asm volatile( \
//   "fadd.d ft3, %[zero], ft0\n" /* load the initial value */ \
//   "frep.o %[n_frep], 1, 0, 0\n" \
//   "fmax.d ft3, ft3, ft0\n" \
//   "fadd.d ft1, %[zero], ft3\n" /* store the final value */ \
//   : \
//   : [zero] "f"(0.0), [n_frep] "r"(total_iter) /* loading initial val takes 1 read */ \
//   : "ft0", "ft1", "ft2", "ft3", "memory" \
// )

// TODO: Replace with better impls? Problems with using math.h...
double floor(double x) {
  return (double) ((int) x);
}

// https://stackoverflow.com/a/8377539
double ceil(double num)
{
  int inum = (int)num;
  if (num == (double)inum) {
      return num;
  }
  return inum + 1;
}

int min(int a, int b) {
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
