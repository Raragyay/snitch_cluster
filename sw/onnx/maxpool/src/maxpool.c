#include "maxpool.h"
#include "snrt.h"
// #include "math.h"
#include "printf.h"

#define DMA_USE_CACHED_ATTRIBS 1

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

void maxpool_fp64_1d(maxpool_attributes*, double*, double*, int*, int, int);
void maxpool_fp64_2d(maxpool_attributes*, double*, double*, int*, int, int);

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

  // printf("output shape: %d %d %d %d %d\n", output_shape[0], output_shape[1], output_shape[2], output_shape[3], output_shape[4]);

}

void maxpool_fp64_layer(maxpool_attributes* attribs_raw, double* in, double* out, int* idx) {

  uint32_t cluster_num = snrt_cluster_num();
  uint32_t cluster_id = snrt_cluster_idx();
  uint32_t compute_num = snrt_cluster_compute_core_num();
  uint32_t compute_id = snrt_global_core_idx();

  // It seems that we need 8 byte alignment?
  const size_t ATTRIBS_SIZE = sizeof(maxpool_attributes) + (sizeof(maxpool_attributes) % 8);

  char* ptr = (char*) snrt_l1_next();
  
  #ifdef DMA_USE_CACHED_ATTRIBS
  maxpool_attributes* attribs = (maxpool_attributes*) ptr;
  #else
  maxpool_attributes* attribs = attribs_raw;
  #endif

  if (snrt_is_dm_core()) {
    // load the attribs into l1
    snrt_dma_start_1d(ptr, attribs_raw, sizeof(maxpool_attributes));

    ptr += ATTRIBS_SIZE;
    #ifdef DMA_USE_CACHED_ATTRIBS
    snrt_dma_wait_all();
    #endif

    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2];
    if (attribs->n_dim > 1) {
      total_ins *= attribs->input_shape[3];
      if (attribs->n_dim > 2) {
        total_ins *= attribs->input_shape[4];
        //
      }
    }
    // load input data
    snrt_dma_start_1d(ptr, in, sizeof(double) * total_ins);

    ptr += sizeof(double) * total_ins;
    // for 8 byte alignment, theoretically shouldn't be needed for doubles.
    ptr += ((size_t) ptr) % 8; // cursed

    snrt_dma_wait_all();
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();

    int total_outs = attribs->output_shape[0] * attribs->output_shape[1] * attribs->output_shape[2];
    if (attribs->n_dim > 1) {
      total_outs *= attribs->output_shape[3];
      if (attribs->n_dim > 2) {
        total_outs *= attribs->output_shape[4];
        //
      }
    }

    snrt_dma_start_1d(out, ptr, sizeof(double) * total_outs);

    snrt_dma_wait_all();
    snrt_cluster_hw_barrier();
  }
  
  if (snrt_is_compute_core()) {
    snrt_cluster_hw_barrier();

    char* inputs_start = ptr + ATTRIBS_SIZE;
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2];
    if (attribs->n_dim > 1) {
      total_ins *= attribs->input_shape[3];
      if (attribs->n_dim > 2) {
        total_ins *= attribs->input_shape[4];
        //
      }
    }
    char* outputs_start = inputs_start + sizeof(double) * total_ins;
    outputs_start += ((size_t) outputs_start) % 8; // cursed again

    switch (attribs->n_dim) {
    case 1:
      maxpool_fp64_1d(attribs, (double*) inputs_start, (double*) outputs_start, NULL, compute_id, compute_num);
      break;
    case 2:
      maxpool_fp64_2d(attribs, (double*) inputs_start, (double*) outputs_start, NULL, compute_id, compute_num);
      break;
    case 3:
      if (compute_id == 1) maxpool_fp64_3d(attribs, (double*) inputs_start, (double*) outputs_start, NULL, compute_id, compute_num);
      break;
    default:
      break; // error not implemented
    }

    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
  }

}

void maxpool_fp64_1d(maxpool_attributes* attr, double* in, double* out, int* idx, int core_idx, int n_cores) {

  if (attr->n_dim != 1) return; // error

  int total_channels = attr->input_shape[0] * attr->input_shape[1]; // batch size * num channels

  int height = attr->input_shape[2];
  int x_step = height;

  int pooled_height = attr->output_shape[2];
  int y_step = pooled_height;

  int n_steps = total_channels * pooled_height;
  for (int step = core_idx; step < n_steps; step += n_cores) {

    int i = step / pooled_height;
    int ph = step % pooled_height;

    int x_d = i * x_step;
    int y_d = i * y_step;

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
    if (idx != NULL) idx[y_d + ph] = i * x_step + h_index;

  }

}

void maxpool_fp64_2d(maxpool_attributes* attr, double* in, double* out, int* idx, int core_idx, int n_cores) {

  if (attr->n_dim != 2) return; // error

  int total_channels = attr->input_shape[0] * attr->input_shape[1]; // batch size * num channels

  int height = attr->input_shape[2];
  int width = attr->input_shape[3];
  int x_step = height * width;

  int total_els = total_channels * height * width;

  int pooled_height = attr->output_shape[2];
  int pooled_width = attr->output_shape[3];
  int y_step = pooled_height * pooled_width;

  int n_steps = total_channels * y_step;
  for (int step = core_idx; step < n_steps; step += n_cores) {

    int i = step / y_step;
    int inst_idx = step % y_step;
    int ph = inst_idx / pooled_width;
    int pw = inst_idx % pooled_width;

    int x_d = i * x_step;
    int y_d = i * y_step;

    int hstart = ph * attr->strides[0] - attr->pads[0];
    int hend = hstart + attr->kernel_shape[0] * attr->dilations[0];

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
    if (idx != NULL) {
      if (!attr->storage_order) idx[y_d + pool_index] = i * x_step + h_index + width + w_index;
      else idx[y_d + pool_index] = i * x_step + h_index + w_index * height;
    }

  }
  
}

void maxpool_fp64_3d(maxpool_attributes* attr, double* in, double* out, int* idx, int core_idx, int n_cores) {

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

  // int n_steps = total_channels * y_step;
  // for (int step = core_idx; step < n_steps; step += n_cores) {

  //   int i = step / y_step;
  //   int inst_idx = step % y_step;

  //   // int pd = inst_idx / (pooled_height * pooled_width);
  //   // inst_idx -= pd * pooled_height * pooled_width;
  //   // int pw = inst_idx / pooled_height;
  //   // int ph = inst_idx % pooled_height;

  //   // x = depth, y = width, z = height
  //   int ph = inst_idx / (pooled_width * pooled_depth);
  //   inst_idx -= ph * pooled_width * pooled_depth;
  //   int pw = inst_idx / pooled_depth;
  //   int pd = inst_idx % pooled_depth;

  //   int x_d = i * x_step;
  //   int y_d = i * y_step;

  //   int hstart = ph * attr->strides[0] - attr->pads[0];
  //   int hend = hstart + attr->kernel_shape[0] * attr->dilations[0];

  //   int wstart = pw * attr->strides[1] - attr->pads[1];
  //   int wend = wstart + attr->kernel_shape[1] * attr->dilations[1];

  //   int dstart = pd * attr->strides[2] - attr->pads[2];
  //   int dend = dstart + attr->kernel_shape[2] + attr->dilations[2];

  //   int pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
  //   int h_index, w_index, d_index;
  //   double Yh;
  //   int Yh_init = 0;

  //   for (int h = hstart; h < hend; h += attr->dilations[0]) {
  //     if (h < 0 || h >= height) continue;

  //     for (int w = wstart; w < wend; w += attr->dilations[1]) {
  //       if (w < 0 || w >= width) continue;

  //       for (int d = dstart; d < dend; d += attr->dilations[2]) {
  //         if (d < 0 || d >= depth) continue;

  //         int input_index = h * width * depth + w * depth + d;
  //         if (!Yh_init || in[x_d + input_index] > Yh) {
  //           Yh = in[x_d + input_index];
  //           Yh_init = 1;
  //           h_index = h;
  //           w_index = w;
  //           d_index = d;
  //         }
  //       }
  //     }
  //   }

  //   out[y_d + pool_index] = Yh;
  //   if (idx != NULL) {
  //     if (!attr->storage_order) idx[y_d + pool_index] = i * x_step + h_index * width * depth + w_index * depth + d_index;
  //     else idx[y_d + pool_index] = i * x_step + h_index + w_index * height + d_index * height * width;
  //   }

  // }




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
          if (idx != NULL) {
            if (!attr->storage_order) idx[y_d + pool_index] = i * x_step + h_index * width * depth + w_index * depth + d_index;
            else idx[y_d + pool_index] = i * x_step + h_index + w_index * height + d_index * height * width;
          }
          
        }
      }
    }

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
