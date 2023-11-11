#include "maxpool.h"
#include "snrt.h"
#include "math.h"

/*
enum padding_type {
  NOTSET,
  SAME_UPPER,
  SAME_LOWER,
  VALID
};
*/

/*
typedef struct maxpool_attributes_struct {
  size_t n_dim; // excludes batch size and # channels
  size_t[5] input_shape; // INCLUDES batch size and # channels, 3d worst case
  size_t[5] output_shape; // INCLUDES batch size and # channels, 3d worst case

  enum padding_type auto_pad; // NOTSET, deprecated in maxpool v12
  int ceil_mode; // 0
  size_t[3] dilations; // [1...1]
  size_t[3] kernel_shape; // required
  size_t[6] pads; // [0...0]
  int storage_order; // 0 (row major)
  size_t[3] strides; // [1...1]
} maxpool_attributes;
*/

// TODO: malloc...

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
            ceil(input_spatial_shape[i] / attr->strides[i])
          );
        }
        else {
          output_spatial_shape[i] = (int) (
            floor(input_spatial_shape[i] / attr->strides[i])
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
            (input_spatial_shape[i] - ((attr->kernel_shape[i] - 1) * attr->dilations[i] + 1) + 1) / attr->strides[i]
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
            (input_spatial_shape[i] + attr->pads[i] + attr->pads[i + attr->n_dim] - ((attr->kernel_shape[i] - 1) * attr->dilations[i] + 1))
            / attr->strides[i] + 1
          )
        );
      }

    }
    else {

      for (int i = 0; i < attr->n_dim; ++i) {
        output_spatial_shape[i] = (int) (
          floor(
            (input_spatial_shape[i] + attr->pads[i] + attr->pads[i + attr->n_dim] - ((attr->kernel_shape[i] - 1) * attr->dilations[i] + 1))
            / attr->strides[i] + 1
          )
        );
      }

    }

  }

}

/*
typedef struct maxpool_attributes_struct {
  size_t n_dim; // excludes batch size and # channels
  size_t[5] input_shape; // INCLUDES batch size and # channels, 3d worst case
  size_t[5] output_shape; // INCLUDES batch size and # channels, 3d worst case

  enum padding_type auto_pad; // NOTSET, deprecated in maxpool v12
  int ceil_mode; // 0
  size_t[3] dilations; // [1...1]
  size_t[3] kernel_shape; // required
  size_t[6] pads; // [0...0], [dim1_begin, dim2_begin, dim1_end, dim2_end]
  int storage_order; // 0 (row major)
  size_t[3] strides; // [1...1]
} maxpool_attributes;
*/

void maxpool_fp64_1d(maxpool_attributes* attr, double* in, double* out, int* idx) {

  if (attr->n_dim != 1) return; // error

  int total_channels = attr->input_shape[0] * attr->input_shape[1]; // batch size * num channels

  int height = attr->input_shape[2];
  int x_step = height;

  int pooled_height = attr->output_shape[2];
  int y_step = pooled_height;

  for (int i = 0; i < total_channels; ++i) {

    int x_d = i * x_step;
    int y_d = i * y_step;

    for (int ph = 0; ph < pooled_height; ++ph) {
      int hstart = ph * attr->strides[0] - attr->pads[0];
      int hend = hstart + attr->kernel_shape[0] * attr->dilations[0];

      double Yh;
      int Yh_init = 0;
      int h_index;

      for (int h = hstart; h < hend; h += attr->dilations[0]) {
        if (h < 0 || h >= height) continue;

        if (!Yh_init || in[x_d + h] > Yh) {
          Yh = in[x_d + h];
          Yh_init = 1;
          h_index = h;
        }
      }

      out[y_d + ph] = Yh;
      idx[y_d + ph] = i * x_step + h_index;
    }

  }

}

/*
typedef struct maxpool_attributes_struct {
  size_t n_dim; // excludes batch size and # channels
  size_t[5] input_shape; // INCLUDES batch size and # channels, 3d worst case
  size_t[5] output_shape; // INCLUDES batch size and # channels, 3d worst case

  enum padding_type auto_pad; // NOTSET, deprecated in maxpool v12
  int ceil_mode; // 0
  size_t[3] dilations; // [1...1]
  size_t[3] kernel_shape; // required
  size_t[6] pads; // [0...0], [dim1_begin, dim2_begin, dim1_end, dim2_end]
  int storage_order; // 0 (row major)
  size_t[3] strides; // [1...1]
} maxpool_attributes;
*/

void maxpool_fp64_2d(maxpool_attributes* attr, double* in, double* out, int* idx) {

  if (attr->n_dim != 2) return; // error

  int total_channels = attr->input_shape[0] * attr->input_shape[1]; // batch size * num channels

  int height = attr->input_shape[2];
  int width = attr->input_shape[3];
  int x_step = height * width;

  int total_els = total_channels * height * width;

  int pooled_height = attr->output_shape[2];
  int pooled_width = attr->output_shape[3];
  int y_step = pooled_height * pooled_width;

  for (int i = 0; i < total_channels; ++i) {

    int x_d = i * x_step;
    int y_d = i * y_step;

    for (int ph = 0; ph < pooled_height; ++ph) {
      int hstart = ph * attr->strides[0] - attr->pads[0];
      int hend = hstart + attr->kernel_shape[0] * attr->dilations[0];
      for (int pw = 0; pw < pooled_width; ++pw) {
        int wstart = pw * attr->strides[1] - attr->pads[1];
        int wend = wstart + attr->kernel_shape[1] * attr->dilations[1];
        int pool_index = ph * pooled_width + pw;

        int h_index, w_index;
        double Yh;
        int Yh_init = 0;

        for (int h = hstart; h < hend; h += attr->dilations[0]) {
          if (h < 0 || h >= height) continue;

          for (int w = wstart; w < wend; w += attr->dilations[1]) {
            if (w < 0 || w >= width) continue;

            int input_index = h * width + w;
            if (input_index < 0 || input_index > total_els) continue;

            if (!Yh_init || in[x_d + input_index] > Yh) {
              Yh = in[x_d + input_index];
              Yh_init = 1;
              h_index = h;
              w_index = w;
            }
          }
        }

        if (!Yh_init) continue;

        out[y_d + pool_index] = Yh;
        if (!attr->storage_order) idx[y_d + pool_index] = i * x_step + h_index + width + w_index;
        else idx[y_d + pool_index] = i * x_step + h_index + w_index * height;

      }
    }

  }
  
}

/*
typedef struct maxpool_attributes_struct {
  size_t n_dim; // excludes batch size and # channels
  size_t[5] input_shape; // INCLUDES batch size and # channels, 3d worst case
  size_t[5] output_shape; // INCLUDES batch size and # channels, 3d worst case

  enum padding_type auto_pad; // NOTSET, deprecated in maxpool v12
  int ceil_mode; // 0
  size_t[3] dilations; // [1...1]
  size_t[3] kernel_shape; // required
  size_t[6] pads; // [0...0], [dim1_begin, dim2_begin, dim1_end, dim2_end]
  int storage_order; // 0 (row major)
  size_t[3] strides; // [1...1]
} maxpool_attributes;
*/

void maxpool_fp64_3d(maxpool_attributes* attr, double* in, double* out, int* idx) {

  if (attr->n_dim != 3) return; // error

  int total_channels = attr->input_shape[0] * attr->input_shape[1]; // batch size * num channels

  int height = attr->input_shape[2];
  int width = attr->input_shape[3];
  int depth = attr->input_shape[4];
  int x_step = height * width * depth;

  int pooled_height = attr->output_shape[2];
  int pooled_width = attr->output_shape[3];
  int pooled_depth = attr->output_shape[4];
  int y_step = pooled_height * pooled_width * pooled_depth;

  for (int i = 0; i < total_channels; ++i) {

    int x_d = i * x_step;
    int y_d = i * y_step;

    for (int ph = 0; ph < pooled_height; ++ph) {
      int hstart = ph * attr->strides[0] - attr->pads[0];
      int hend = hstart + attr->kernel_shape[0] * attr->dilations[0];
      for (int pw = 0; pw < pooled_width; ++pw) {
        int wstart = pw * attr->strides[1] - attr->pads[1];
        int wend = wstart + attr->kernel_shape[1] * attr->dilations[1];
        for (int pd = 0; pd < pooled_depth; ++pd) {
          int dstart = pd * attr->strides[2] - attr->pads[2];
          int dend = dstart + attr->kernel_shape[2] + attr->dilations[2];

          int pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
          int h_index, w_index, d_index;
          double Yh;
          int Yh_init = 0;

          for (int h = hstart; h < hend; h += attr->dilations[0]) {
            if (h < 0 || h >= height) continue;

            for (int w = wstart; w < wend; w += attr->dilations[1]) {
              if (w < 0 || w >= width) continue;

              for (int d = dstart; d < dend; d += attr->dilations[2]) {
                if (d < 0 || d >= depth) continue;

                int input_index = h * width * depth + w * depth + d;
                if (!Yh_init || in[x_d + input_index] > Yh) {
                  Yh = in[x_d + input_index];
                  Yh_init = 1;
                  h_index = h;
                  w_index = w;
                  d_index = d;
                }
              }
            }
          }

          out[y_d + pool_index] = Yh;
          if (!attr->storage_order) idx[y_d + pool_index] = i * x_step + h_index * width * depth + w_index * depth + d_index;
          else idx[y_d + pool_index] = i * x_step + h_index + w_index * height + d_index * height * width;
          
        }
      }
    }

  }
  
}

/*
typedef struct maxpool_attributes_struct {
  size_t n_dim; // excludes batch size and # channels
  size_t[5] input_shape; // INCLUDES batch size and # channels, 3d worst case
  size_t[5] output_shape; // INCLUDES batch size and # channels, 3d worst case

  enum padding_type auto_pad; // NOTSET, deprecated in maxpool v12
  int ceil_mode; // 0
  size_t[3] dilations; // [1...1]
  size_t[3] kernel_shape; // required
  size_t[6] pads; // [0...0], [dim1_begin, dim2_begin, dim1_end, dim2_end]
  int storage_order; // 0 (row major)
  size_t[3] strides; // [1...1]
} maxpool_attributes;
*/

void maxpool_fp64(maxpool_attributes* attr, double* in, double* out, int* idx) {

  compute_output_shape(attr, attr->output_shape);

  switch (attr->n_dim) {
  case 1:
    maxpool_fp64_1d(attr, in, out, idx);
  case 2:
    maxpool_fp64_2d(attr, in, out, idx);
  case 3:
    maxpool_fp64_3d(attr, in, out, idx);
  default:
    // error not implemented
    break;
  }
  
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
