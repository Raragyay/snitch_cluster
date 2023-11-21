#ifndef MAXPOOL_FN
#error "Maxpool function name not defined"
#endif

#define _CONCAT(A, B) A ## B
#define CONCAT(A, B) _CONCAT(A, B)

#define _1D _1d
#define _2D _2d
#define _3D _3d

#define MAXPOOL_FN_1D CONCAT(MAXPOOL_FN, _1D)
#define MAXPOOL_FN_2D CONCAT(MAXPOOL_FN, _2D)
#define MAXPOOL_FN_3D CONCAT(MAXPOOL_FN, _3D)

#ifndef ___MAXPOOL_C
#define ___MAXPOOL_C
static inline void ssr_asm_with_index(int*, int);

void ssr_asm_with_index(int* out_idx, int total_iter) {
  asm volatile(
    "li t0, 0\n" /* counter, start at one because initial val has idx 0 */
    "li t1, 0\n" /* cur max idx */
    "fadd.d ft3, %[zero], ft0\n" /* ft3 = cur max val */
    /* begin loop */
    "addi t0, t0, 1\n"
    "fmax.d ft4, ft3, ft0\n"
    "feq.d t3, ft4, ft3\n"
    "bne t3, zero, 12\n" /* branch if no update needed */
    "fadd.d ft3, %[zero], ft4\n" /* update cur max val */
    "add t1, zero, t0\n" /* update cur max idx */
    "bne t0, %[n_iter], -24\n"
    "fadd.d ft1, %[zero], ft3\n"
    "addi %[idx_out], t1, 0\n" /* write out max idx to memory */
    : [idx_out] "=r"(*out_idx)
    : [zero] "f"(0.0), [n_iter] "r"(total_iter)
    : "t0", "t1", "t3", "ft0", "ft1", "ft2", "ft3", "ft4", "memory", "zero"
  );
}

static inline void ssr_asm_no_index(int);

void ssr_asm_no_index(int total_iter) {
  asm volatile(
    "fadd.d ft3, %[zero], ft0\n" /* load the initial value */
    "frep.o %[n_frep], 1, 0, 0\n"
    "fmax.d ft3, ft3, ft0\n"
    "fadd.d ft1, %[zero], ft3\n" /* store the final value */
    :
    : [zero] "f"(0.0), [n_frep] "r"(total_iter) /* loading initial val takes 1 read */
    : "ft0", "ft1", "ft2", "ft3", "memory"
  );
}

#endif

static inline void MAXPOOL_FN_1D(maxpool_attributes*, double*, double*, int*, int, int);
static inline void MAXPOOL_FN_2D(maxpool_attributes*, double*, double*, int*, int, int);
static inline void MAXPOOL_FN_3D(maxpool_attributes*, double*, double*, int*, int, int);

inline void MAXPOOL_FN(maxpool_attributes*, double*, double*, int*);

void MAXPOOL_FN(maxpool_attributes* attribs_raw, double* in, double* out, int* idx) {

  uint32_t cluster_num = snrt_cluster_num();
  uint32_t cluster_id = snrt_cluster_idx();
  uint32_t compute_num = snrt_cluster_compute_core_num();
  uint32_t compute_id = snrt_global_core_idx();

  // It seems that we need 8 byte alignment?
  const size_t ATTRIBS_SIZE = sizeof(maxpool_attributes) + (sizeof(maxpool_attributes) % 8);

  char* ptr = (char*) snrt_l1_next();
  
  #if DMA_ATTRIBS
  maxpool_attributes* attribs = (maxpool_attributes*) ptr;
  #else
  maxpool_attributes* attribs = attribs_raw;
  #endif

  if (snrt_is_dm_core()) {
    // load the attribs into l1
    snrt_dma_start_1d(ptr, attribs_raw, sizeof(maxpool_attributes));

    ptr += ATTRIBS_SIZE;
    #if DMA_ATTRIBS
    snrt_dma_wait_all();
    #endif

    #if MAXPOOL_DIM == 1
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2];
    #elif MAXPOOL_DIM == 2
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2] * attribs->input_shape[3];
    #elif MAXPOOL_DIM == 3
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2] * attribs->input_shape[3] * attribs->input_shape[4];
    #endif

    // int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2];
    // if (attribs->n_dim > 1) {
    //   total_ins *= attribs->input_shape[3];
    //   if (attribs->n_dim > 2) {
    //     total_ins *= attribs->input_shape[4];
    //     //
    //   }
    // }
    // load input data
    snrt_dma_start_1d(ptr, in, sizeof(double) * total_ins);

    ptr += sizeof(double) * total_ins;
    // for 8 byte alignment, theoretically shouldn't be needed for doubles.
    ptr += ((size_t) ptr) % 8; // cursed

    snrt_dma_wait_all();
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();

    #if MAXPOOL_DIM == 1
    int total_outs = attribs->output_shape[0] * attribs->output_shape[1] * attribs->output_shape[2];
    #elif MAXPOOL_DIM == 2
    int total_outs = attribs->output_shape[0] * attribs->output_shape[1] * attribs->output_shape[2] * attribs->output_shape[3];
    #elif MAXPOOL_DIM == 3
    int total_outs = attribs->output_shape[0] * attribs->output_shape[1] * attribs->output_shape[2] * attribs->output_shape[3] * attribs->output_shape[4];
    #endif
    // int total_outs = attribs->output_shape[0] * attribs->output_shape[1] * attribs->output_shape[2];
    // if (attribs->n_dim > 1) {
    //   total_outs *= attribs->output_shape[3];
    //   if (attribs->n_dim > 2) {
    //     total_outs *= attribs->output_shape[4];
    //     //
    //   }
    // }

    snrt_dma_start_1d(out, ptr, sizeof(double) * total_outs);

    snrt_dma_wait_all();
    snrt_cluster_hw_barrier();
  }
  
  if (snrt_is_compute_core()) {
    snrt_cluster_hw_barrier();

    char* inputs_start = ptr + ATTRIBS_SIZE;
    #if MAXPOOL_DIM == 1
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2];
    #elif MAXPOOL_DIM == 2
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2] * attribs->input_shape[3];
    #elif MAXPOOL_DIM == 3
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2] * attribs->input_shape[3] * attribs->input_shape[4];
    #endif
    // int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2];
    // if (attribs->n_dim > 1) {
    //   total_ins *= attribs->input_shape[3];
    //   if (attribs->n_dim > 2) {
    //     total_ins *= attribs->input_shape[4];
    //     //
    //   }
    // }
    char* outputs_start = inputs_start + sizeof(double) * total_ins;
    outputs_start += ((size_t) outputs_start) % 8; // cursed again

    #if MAXPOOL_DIM == 1
    MAXPOOL_FN_1D(attribs, (double*) inputs_start, (double*) outputs_start, idx, compute_id, compute_num);
    #elif MAXPOOL_DIM == 2
    snrt_mcycle();
    MAXPOOL_FN_2D(attribs, (double*) inputs_start, (double*) outputs_start, idx, compute_id, compute_num);
    snrt_mcycle();
    #elif MAXPOOL_DIM == 3
    MAXPOOL_FN_3D(attribs, (double*) inputs_start, (double*) outputs_start, idx, compute_id, compute_num);
    #endif

    // switch (attribs->n_dim) {
    // case 1:
    //   maxpool_fp64_1d(attribs, (double*) inputs_start, (double*) outputs_start, idx, compute_id, compute_num);
    //   break;
    // case 2:
    //   maxpool_fp64_2d(attribs, (double*) inputs_start, (double*) outputs_start, idx, compute_id, compute_num);
    //   break;
    // case 3:
    //   maxpool_fp64_3d(attribs, (double*) inputs_start, (double*) outputs_start, idx, compute_id, compute_num);
    //   break;
    // default:
    //   break; // error not implemented
    // }

    snrt_fpu_fence();
    snrt_cluster_hw_barrier();
    snrt_cluster_hw_barrier();
  }

}

void MAXPOOL_FN_1D(maxpool_attributes* attr, double* in, double* out, int* idx, int core_idx, int n_cores) {

  // if (attr->n_dim != 1) return; // error

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
    int hend = min(hstart + attr->kernel_shape[0] * attr->dilations[0], height);

    if (hstart < 0) {
      hstart = (hstart % attr->dilations[0]);
      if (hstart < 0) hstart += attr->dilations[0];
    }

    int h_index;
    #if USE_SSR_FREP_1D || USE_SSR_FREP_ALL

    int n_iter = (hend - hstart + attr->dilations[0] - 1) / attr->dilations[0];
    if (n_iter == 1) {
      out[y_d + ph] = in[x_d + hstart];
      
      #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
      idx[y_d + ph] = hstart;
      #endif
      continue;
    }
    //printf("iter: %d %d %d\n", n_iter, hstart, hend);
    snrt_ssr_loop_1d(SNRT_SSR_DM0, n_iter, sizeof(double) * attr->dilations[0]);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 1, 0); // value output

    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, in + x_d + hstart);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out + y_d + ph);

    snrt_ssr_enable();

    #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
    ssr_asm_with_index(&h_index, n_iter - 1);
    snrt_ssr_disable();
    idx[y_d + ph] = hstart + h_index * attr->dilations[0];
    #else
    ssr_asm_no_index(n_iter - 2);
    snrt_ssr_disable();
    #endif

    #else

    double Yh;
    int Yh_init = 0;
    for (int h = hstart; h < hend; h += attr->dilations[0]) {
      // if (h < 0) continue;
      // if (h >= height) break;
      if (!Yh_init || in[x_d + h] > Yh) {
        Yh = in[x_d + h];
        Yh_init = 1;
        #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
        h_index = h;
        #endif
      }
    }

    out[y_d + ph] = Yh;
    #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
    idx[y_d + ph] = h_index;
    #endif

    #endif

  }

}

void MAXPOOL_FN_2D(maxpool_attributes* attr, double* in, double* out, int* idx, int core_idx, int n_cores) {

  // if (attr->n_dim != 2) return; // error

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
    int hend = min(hstart + attr->kernel_shape[0] * attr->dilations[0], height);

    if (hstart < 0) {
      hstart = (hstart % attr->dilations[0]);
      if (hstart < 0) hstart += attr->dilations[0];
    }

    int wstart = pw * attr->strides[1] - attr->pads[1];;
    int wend = min(wstart + attr->kernel_shape[1] * attr->dilations[1], width);

    if (wstart < 0) {
      wstart = (wstart % attr->dilations[1]);
      if (wstart < 0) wstart += attr->dilations[1];
    }

    int pool_index = ph * pooled_width + pw;

    #if USE_SSR_FREP_2D || USE_SSR_FREP_ALL

      int max_index;

      int n_iter_h = (hend - hstart + attr->dilations[0] - 1) / attr->dilations[0];
      int n_iter_w = (wend - wstart + attr->dilations[1] - 1) / attr->dilations[1];
      if (n_iter_h * n_iter_w == 1) {
        out[y_d + pool_index] = in[x_d + hstart * width + wstart];
        #if defined(MAXPOOL_COL_MAJOR)
          idx[y_d + pool_index] = hstart + wstart * height;
        #elif defined(MAXPOOL_ROW_MAJOR)
          idx[y_d + pool_index] = hstart * width + wstart;
        #endif
        continue;
      }

      snrt_ssr_loop_2d(SNRT_SSR_DM0, n_iter_w, n_iter_h, sizeof(double) * attr->dilations[1], sizeof(double) * attr->dilations[0] * width);
      snrt_ssr_loop_1d(SNRT_SSR_DM1, 1, 0); // value output

      snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, in + x_d + hstart * width + wstart);
      snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out + y_d + pool_index);

      snrt_ssr_enable();

      #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)

      ssr_asm_with_index(&max_index, n_iter_h * n_iter_w - 1);
      snrt_ssr_disable();

      int h_index = max_index / n_iter_w * attr->dilations[0] + hstart;
      int w_index = (max_index % n_iter_w) * attr->dilations[1] + wstart;

      #ifdef MAXPOOL_COL_MAJOR
      idx[y_d + pool_index] = h_index + w_index * height;
      #else
      idx[y_d + pool_index] = h_index * width + w_index;
      #endif

      #else

        ssr_asm_no_index(n_iter_h * n_iter_w - 2);
        snrt_ssr_disable();

      #endif

    #else

      #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
      int h_index, w_index;
      #endif
      double Yh;
      int Yh_init = 0;
      for (int h = hstart; h < hend; h += attr->dilations[0]) {
        // if (h < 0 || h >= height) continue;

        for (int w = wstart; w < wend; w += attr->dilations[1]) {
          // if (w < 0 || w >= width) continue;

          int input_index = h * width + w;
          // if (input_index < 0 || input_index > total_els) continue;

          if (!Yh_init || in[x_d + input_index] > Yh) {
            Yh = in[x_d + input_index];
            Yh_init = 1;
            #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
            h_index = h;
            w_index = w;
            #endif
          }
        }
      }

      // if (!Yh_init) continue;

      out[y_d + pool_index] = Yh;
      
      #if defined(MAXPOOL_COL_MAJOR)
        idx[y_d + pool_index] = h_index + w_index * height;
      #elif defined(MAXPOOL_ROW_MAJOR)
        idx[y_d + pool_index] = h_index * width + w_index;
      #endif

    #endif

  }
  
}

void MAXPOOL_FN_3D(maxpool_attributes* attr, double* in, double* out, int* idx, int core_idx, int n_cores) {

  // if (attr->n_dim != 3) return; // error

  int total_channels = attr->input_shape[0] * attr->input_shape[1]; // batch size * num channels

  int height = attr->input_shape[2];
  int width = attr->input_shape[3];
  int depth = attr->input_shape[4];
  int x_step = height * width * depth;

  int pooled_height = attr->output_shape[2];
  int pooled_width = attr->output_shape[3];
  int pooled_depth = attr->output_shape[4];
  int y_step = pooled_height * pooled_width * pooled_depth;

  int n_steps = total_channels * y_step;
  for (int step = core_idx; step < n_steps; step += n_cores) {

    int i = step / y_step;
    int inst_idx = step % y_step;

    // int pd = inst_idx / (pooled_height * pooled_width);
    // inst_idx -= pd * pooled_height * pooled_width;
    // int pw = inst_idx / pooled_height;
    // int ph = inst_idx % pooled_height;

    // x = depth, y = width, z = height
    int ph = inst_idx / (pooled_width * pooled_depth);
    inst_idx -= ph * pooled_width * pooled_depth;
    int pw = inst_idx / pooled_depth;
    int pd = inst_idx % pooled_depth;

    int x_d = i * x_step;
    int y_d = i * y_step;

    int hstart = ph * attr->strides[0] - attr->pads[0];
    int hend = hstart + attr->kernel_shape[0] * attr->dilations[0];

    int wstart = pw * attr->strides[1] - attr->pads[1];
    int wend = wstart + attr->kernel_shape[1] * attr->dilations[1];

    int dstart = pd * attr->strides[2] - attr->pads[2];
    int dend = dstart + attr->kernel_shape[2] * attr->dilations[2];

    int pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;

    #if USE_SSR_FREP_2D || USE_SSR_FREP_ALL
    #else

      #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
      int h_index, w_index, d_index;
      #endif
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
              #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
              h_index = h;
              w_index = w;
              d_index = d;
              #endif
            }
          }
        }
      }

      out[y_d + pool_index] = Yh;
      #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
        #if defined(MAXPOOL_ROW_MAJOR)
        idx[y_d + pool_index] = h_index * width * depth + w_index * depth + d_index;
        #else
        idx[y_d + pool_index] = h_index + w_index * height + d_index * height * width;
        #endif
      #endif

    #endif

  }
  
}
