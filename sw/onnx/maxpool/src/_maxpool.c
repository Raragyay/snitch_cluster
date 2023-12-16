#ifdef MAXPOOL_FN // serves as an include guard

#define _CONCAT(A, B) A ## B
#define CONCAT(A, B) _CONCAT(A, B)

#define _1D _1d
#define _2D _2d
#define _3D _3d
#define UNTILED _UNTILED

#define MAXPOOL_FN_1D CONCAT(MAXPOOL_FN, _1D)
#define MAXPOOL_FN_2D CONCAT(MAXPOOL_FN, _2D)
#define MAXPOOL_FN_3D CONCAT(MAXPOOL_FN, _3D)
#define MAXPOOL_FN_UNTILED CONCAT(MAXPOOL_FN, UNTILED)

#ifndef ___MAXPOOL_C
#define ___MAXPOOL_C

#define DMA_ATTRIBS 1
#define DMA_INDICES 1
#define USE_SSR_FREP_1D 1
#define USE_SSR_FREP_2D 1
#define USE_SSR_FREP_3D 1
#define USE_SSR_FREP_ALL 0
#define USE_DOUBLE_BUFFERING 1
#define ENABLE_SPECIALIZED 1

#define ENABLE_BENCHMARKING 1

// Assume we have ~100kb of free cache
#define BASE_USABLE_CACHE 100000

// number of usable bytes in l1
#define USABLE_CACHE BASE_USABLE_CACHE

// for 8 byte alignment
#define ATTRIBS_SIZE ((sizeof(maxpool_attributes) + 4) - ((sizeof(maxpool_attributes) + 4) % 8))

#ifdef DMA_ATTRIBS
  #undef USABLE_CACHE
  #define USABLE_CACHE (BASE_USABLE_CACHE - ATTRIBS_SIZE)
#endif

// number of usable bytes per half
#define HALF_CACHE ((USABLE_CACHE / 2) - 16)

static inline int ceil_div(int, int);

int ceil_div(int a, int b) {
  return (((a)-1) / (b)) + 1;
}

static inline int align(int);

int align(int a) {
  return (a + 4) - ((a + 4) % 8);
}

static inline int neg_mod(int, int);

int neg_mod(int x, int y) {
  return ((-x % y) + y) % y;
}

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

void ssr_asm_no_index(int n_iter_minus_two) {

  if (n_iter_minus_two == 0) {
    asm volatile( 
      "fadd.d ft3, %[zero], ft0\n" /* load the initial value */
      "fmax.d ft1, ft3, ft0\n"
      :
      : [zero] "f"(0.0)
      : "ft0", "ft1", "ft2", "ft3", "memory"
    );
  }
  else if (n_iter_minus_two % 2 == 0) {
    asm volatile(
      "fadd.d ft3, %[zero], ft0\n" /* load the initial value */
      "fmax.d ft3, ft3, ft0\n"
      "frep.o %[n_frep], 2, 0, 0\n"
      "fmax.d ft4, ft3, ft0\n"
      "fmax.d ft3, ft4, ft0\n"
      "fadd.d ft1, %[zero], ft3\n" /* store the final value */
      :
      : [zero] "f"(0.0), [n_frep] "r"(n_iter_minus_two / 2 - 1) /* loading initial val takes 1 read */
      : "ft0", "ft1", "ft2", "ft3", "ft4", "memory"
    );
  }
  else {
    asm volatile(
      "fadd.d ft3, %[zero], ft0\n" /* load the initial value */
      "frep.o %[n_frep], 2, 0, 0\n"
      "fmax.d ft4, ft3, ft0\n"
      "fmax.d ft3, ft4, ft0\n"
      "fadd.d ft1, %[zero], ft3\n" /* store the final value */
      :
      : [zero] "f"(0.0), [n_frep] "r"(n_iter_minus_two / 2) /* loading initial val takes 1 read */
      : "ft0", "ft1", "ft2", "ft3", "ft4", "memory"
    );
  }
  // asm volatile(
  //   "fadd.d ft3, %[zero], ft0\n" /* load the initial value */
  //   "frep.o %[n_frep], 1, 0, 0\n"
  //   "fmax.d ft3, ft3, ft0\n"
  //   "fadd.d ft1, %[zero], ft3\n" /* store the final value */
  //   :
  //   : [zero] "f"(0.0), [n_frep] "r"(n_iter_minus_two) /* loading initial val takes 1 read */
  //   : "ft0", "ft1", "ft2", "ft3", "memory"
  // );
}

static inline void ssr_asm_no_index(int);

void ssr_asm_no_index_optimized(int n_iter_minus_two, int total_iters) {

  // DUMP(n_iter_minus_two);
  // DUMP(total_iters);

  // asm volatile(
  //   "li t0, 0\n"

  //   "fadd.d ft3, %[zero], ft0\n"
  //   "frep.o %[n_frep], 1, 0, 0\n"
  //   "fmax.d ft3, ft3, ft0\n"
  //   "fadd.d ft1, %[zero], ft3\n"

  //   "addi t0, t0, 1\n"

  //   // "fmv.x.w %[tmp], fa0\n"
  //   // "mv      %[tmp], %[tmp]\n"

  //   "bne t0, %[total_iters], -20\n"

  //   :/* [tmp] "+r"(tmp)*/
  //   : [zero] "f"(0.0),
  //     // [work_this_core] "r"(work_this_core),
  //     [n_frep] "r"(n_frep),
  //     // [n_channels] "r"(n_channels),
  //     [total_iters] "r"(total_iters)
  //   : "t0", "t1", "ft0", "ft1", "ft2", "ft3", "memory", "zero"
  // );

  if (n_iter_minus_two == 0) {
    asm volatile(
      "li t0, 0\n"

      "fadd.d ft3, %[zero], ft0\n" /* load the initial value */
      "fmax.d ft1, ft3, ft0\n"

      "addi t0, t0, 1\n"

      "bne t0, %[total_iters], -12\n"

      :
      : [zero] "f"(0.0),
        [total_iters] "r"(total_iters)
      : "t0", "ft0", "ft1", "ft2", "ft3", "memory", "zero"
    );
  }
  else if (n_iter_minus_two == -1) {
    asm volatile(
      "li t0, 0\n"

      "fadd.d ft1, %[zero], ft0\n"

      "addi t0, t0, 1\n"

      "bne t0, %[total_iters], -8\n"

      :
      : [zero] "f"(0.0),
        [total_iters] "r"(total_iters)
      : "t0", "ft0", "ft1", "ft2", "memory", "zero"
    );
  }
  else if (n_iter_minus_two % 2 == 0) {
    asm volatile(
      "li t0, 0\n"

      "fadd.d ft3, %[zero], ft0\n" /* load the initial value */
      "fmax.d ft3, ft3, ft0\n"
      "frep.o %[n_frep], 2, 0, 0\n"
      "fmax.d ft4, ft3, ft0\n"
      "fmax.d ft3, ft4, ft0\n"
      "fadd.d ft1, %[zero], ft3\n" /* store the final value */

      "addi t0, t0, 1\n"

      "bne t0, %[total_iters], -28\n"

      :
      : [zero] "f"(0.0),
        [n_frep] "r"(n_iter_minus_two / 2 - 1),
        [total_iters] "r"(total_iters)
      : "t0", "t1", "ft0", "ft1", "ft2", "ft3", "ft4", "memory", "zero"
    );
  }
  else {
    asm volatile(
      "li t0, 0\n"

      "fadd.d ft3, %[zero], ft0\n" /* load the initial value */
      "frep.o %[n_frep], 2, 0, 0\n"
      "fmax.d ft4, ft3, ft0\n"
      "fmax.d ft3, ft4, ft0\n"
      "fadd.d ft1, %[zero], ft3\n" /* store the final value */

      "addi t0, t0, 1\n"

      "bne t0, %[total_iters], -24\n"

      :
      : [zero] "f"(0.0),
        [n_frep] "r"(n_iter_minus_two / 2),
        [total_iters] "r"(total_iters)
      : "t0", "t1", "ft0", "ft1", "ft2", "ft3", "ft4", "memory", "zero"
    );
  }
}

#endif

static inline void MAXPOOL_FN_1D(maxpool_attributes*, double*, double*, int*, int, int, int);
static inline void MAXPOOL_FN_2D(maxpool_attributes*, double*, double*, int*, int, int, int);
static inline void MAXPOOL_FN_3D(maxpool_attributes*, double*, double*, int*, int, int, int);

static inline void MAXPOOL_FN_UNTILED(maxpool_attributes*, double*, double*, int*, int, int, int, int, char*);

void MAXPOOL_FN_UNTILED(maxpool_attributes* attribs,
                        double* in,
                        double* out,
                        int* idx,
                        int compute_id,
                        int compute_num,
                        int total_ins,
                        int total_outs,
                        char* ptr) {
  // if (compute_id == 1) DUMP(20052961);
  if (snrt_is_dm_core()) {

    // load input data
    snrt_dma_start_1d(ptr, in, sizeof(double) * total_ins);

    ptr += sizeof(double) * total_ins;
    // for 8 byte alignment, theoretically shouldn't be needed for doubles.
    ptr += ((size_t) ptr) % 8; // cursed

    snrt_dma_wait_all();
    snrt_cluster_hw_barrier(); // 1: input loaded

    snrt_cluster_hw_barrier(); // 2: computation finished

    snrt_dma_start_1d(out, ptr, sizeof(double) * total_outs);
    ptr += sizeof(double) * total_outs;
    ptr += ((size_t) ptr) % 8;

    #if DMA_INDICES
    snrt_dma_wait_all();
    snrt_dma_start_1d(idx, ptr, sizeof(int) * total_outs);
    #endif

    snrt_dma_wait_all();
    snrt_cluster_hw_barrier(); // 3: output written to main mem
  }
  
  if (snrt_is_compute_core()) {
    snrt_cluster_hw_barrier(); // 1

    char* inputs_start = ptr;
    char* outputs_start = inputs_start + sizeof(double) * total_ins;
    outputs_start += ((size_t) outputs_start) % 8; // cursed again

    #if DMA_INDICES
    int* idx_out = (int*) (outputs_start + sizeof(double) * total_outs);
    #else
    int* idx_out = idx;
    #endif

    #if ENABLE_BENCHMARKING
    uint32_t a = snrt_mcycle();
    #endif
    #if MAXPOOL_DIM == 1
    MAXPOOL_FN_1D(attribs, (double*) inputs_start, (double*) outputs_start, idx_out, compute_id, compute_num, total_outs);
    #elif MAXPOOL_DIM == 2
    MAXPOOL_FN_2D(attribs, (double*) inputs_start, (double*) outputs_start, idx_out, compute_id, compute_num, total_outs);
    #elif MAXPOOL_DIM == 3
    MAXPOOL_FN_3D(attribs, (double*) inputs_start, (double*) outputs_start, idx_out, compute_id, compute_num, total_outs);
    #endif
    #if ENABLE_BENCHMARKING
    uint32_t b = snrt_mcycle();
    // if (snrt_global_core_idx() == 0) DUMP(b - a);
    #endif

    snrt_fpu_fence();
    snrt_cluster_hw_barrier(); // 2
    snrt_cluster_hw_barrier(); // 3
  }
}

inline void MAXPOOL_FN(maxpool_attributes*, double*, double*, int*);

void MAXPOOL_FN(maxpool_attributes* attribs_raw, double* in, double* out, int* idx) {

  uint32_t cluster_num = snrt_cluster_num();
  uint32_t cluster_id = snrt_cluster_idx();
  uint32_t compute_num = snrt_cluster_compute_core_num();
  uint32_t compute_id = snrt_global_core_idx();

  char* ptr = (char*) snrt_l1_next();
  
  #if DMA_ATTRIBS
    maxpool_attributes* attribs = (maxpool_attributes*) ptr;
    if (snrt_is_dm_core()) {
      // load the attribs into l1
      snrt_dma_start_1d(ptr, attribs_raw, sizeof(maxpool_attributes));
      
      snrt_dma_wait_all();
      snrt_cluster_hw_barrier();
    }
    if (snrt_is_compute_core()) snrt_cluster_hw_barrier(); // wait for attribs to load
  #else
    maxpool_attributes* attribs = attribs_raw;
  #endif

  ptr += ATTRIBS_SIZE;

  #if MAXPOOL_DIM == 1
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2];
    int total_outs = attribs->output_shape[0] * attribs->output_shape[1] * attribs->output_shape[2];
    int elems_per_matrix = attribs->input_shape[2];
    int outs_per_matrix = attribs->output_shape[2];
  #elif MAXPOOL_DIM == 2
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2] * attribs->input_shape[3];
    int total_outs = attribs->output_shape[0] * attribs->output_shape[1] * attribs->output_shape[2] * attribs->output_shape[3];
    int elems_per_matrix = attribs->input_shape[2] * attribs->input_shape[3];
    int outs_per_matrix = attribs->output_shape[2] * attribs->output_shape[3];
  #elif MAXPOOL_DIM == 3
    int total_ins = attribs->input_shape[0] * attribs->input_shape[1] * attribs->input_shape[2] * attribs->input_shape[3] * attribs->input_shape[4];
    int total_outs = attribs->output_shape[0] * attribs->output_shape[1] * attribs->output_shape[2] * attribs->output_shape[3] * attribs->output_shape[4];
    int elems_per_matrix = attribs->input_shape[2] * attribs->input_shape[3] * attribs->input_shape[4];
    int outs_per_matrix = attribs->output_shape[2] * attribs->output_shape[3] * attribs->output_shape[4];
  #endif
  // if (compute_id == 1) DUMP(elems_per_matrix);

  #if !USE_DOUBLE_BUFFERING
  MAXPOOL_FN_UNTILED(attribs, in, out, idx, compute_id, compute_num, total_ins, total_outs, ptr);
  #else

  #if (defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)) && DMA_INDICES
  int bytes_per_batch = elems_per_matrix * sizeof(double) + outs_per_matrix * (sizeof(double) + sizeof(int));
  #else
  int bytes_per_batch = elems_per_matrix * sizeof(double) + outs_per_matrix * sizeof(double);
  #endif
  
  int total_channels = attribs->input_shape[0] * attribs->input_shape[1];
  if (bytes_per_batch * total_channels <= USABLE_CACHE) {
    MAXPOOL_FN_UNTILED(attribs, in, out, idx, compute_id, compute_num, total_ins, total_outs, ptr);

    return;
  }

  // if (compute_id == 1) DUMP(10052961);

  char* first_ptr = ptr;
  char* second_ptr = ptr + align(USABLE_CACHE / 2);
  // whether we should use the second half of available cache
  int use_second = 0;

  int batches_per_cache = HALF_CACHE / bytes_per_batch;
  // We tile one matrix at a time.
  // Tiling partial matrices is quite complicated due to stride/dilation/padding parameters.
  if (batches_per_cache == 0) {
    printf("Error: Cache must handle at least one matrix.\n");
    return;
  }
  int elems_per_cache = batches_per_cache * elems_per_matrix;
  int outs_per_cache = batches_per_cache * outs_per_matrix;

  maxpool_attributes copy = *attribs;
  copy.input_shape[0] = batches_per_cache;
  copy.input_shape[1] = 1;
  copy.output_shape[0] = batches_per_cache;
  copy.output_shape[1] = 1;

  int num_caches = ceil_div(total_channels, batches_per_cache);

  if (snrt_is_dm_core()) {
    snrt_dma_start_1d(first_ptr, in, elems_per_cache * sizeof(double));
  }

  char* this_ptr;
  char* other_ptr;
  char* idx_ptr = NULL;
  int i;
  for (i = 1; i < num_caches - 1; ++i) {
    use_second = !use_second;
    this_ptr = use_second ? second_ptr : first_ptr;
    other_ptr = use_second ? first_ptr : second_ptr;
    #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
      #if DMA_INDICES
      idx_ptr = other_ptr + (elems_per_cache + outs_per_cache) * sizeof(double);
      #else
      idx_ptr = idx + outs_per_cache * (i - 1);
      #endif
    #endif

    if (snrt_is_dm_core()) {

      // load input data into this_ptr
      snrt_dma_start_1d(this_ptr, ((double*) in) + elems_per_cache * i, elems_per_cache * sizeof(double));

      snrt_dma_wait_all();

      snrt_cluster_hw_barrier(); // 1: wait for computation to finish in other_ptr

      // write output from other_ptr
      snrt_dma_start_1d(((double*) out) + outs_per_cache * (i - 1),
        ((double*) other_ptr) + elems_per_cache,
        outs_per_cache * sizeof(double));
      #if (defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)) && DMA_INDICES
      snrt_dma_start_1d(((int*) idx) + outs_per_cache * (i - 1),
        idx_ptr,
        outs_per_cache * sizeof(int));
      #endif
    }
    if (snrt_is_compute_core()) {
      // do computation on data in other_ptr
      #if ENABLE_BENCHMARKING
      snrt_mcycle();
      #endif
      #if MAXPOOL_DIM == 1
      MAXPOOL_FN_1D(&copy,
        (double*) other_ptr,
        ((double*) other_ptr) + elems_per_cache,
        (int*) idx_ptr,
        compute_id,
        compute_num,
        outs_per_cache);
      #elif MAXPOOL_DIM == 2
      MAXPOOL_FN_2D(&copy,
        (double*) other_ptr,
        ((double*) other_ptr) + elems_per_cache,
        (int*) idx_ptr,
        compute_id,
        compute_num,
        outs_per_cache);
      #else
      MAXPOOL_FN_3D(&copy,
        (double*) other_ptr,
        ((double*) other_ptr) + elems_per_cache,
        (int*) idx_ptr,
        compute_id,
        compute_num,
        outs_per_cache);
      #endif
      #if ENABLE_BENCHMARKING
      snrt_mcycle();
      #endif

      snrt_cluster_hw_barrier(); // 1

    }
  }

  // unroll the last iter since it might not be a full batch
  // we are guaranteed at least 3 batches since <3 batch inputs fallback to the untiled version
  use_second = !use_second;
  this_ptr = use_second ? second_ptr : first_ptr;
  other_ptr = use_second ? first_ptr : second_ptr;
  #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
    #if DMA_INDICES
    idx_ptr = other_ptr + (elems_per_cache + outs_per_cache) * sizeof(double);
    #else
    idx_ptr = idx + outs_per_cache * (i - 1);
    #endif
  #endif

  int batches_left = (total_channels - (batches_per_cache * (num_caches - 1)));
  int ins_left = batches_left * elems_per_matrix;
  int outs_left = batches_left * outs_per_matrix;
  if (snrt_is_dm_core()) {
    // load input data into this_ptr
    snrt_dma_start_1d(this_ptr,
      ((double*) in) + elems_per_cache * i,
      ins_left * sizeof(double));

    snrt_dma_wait_all();

    snrt_cluster_hw_barrier(); // 1: wait for computation to finish in other_ptr

    // write output from other_ptr
    snrt_dma_start_1d(((double*) out) + outs_per_cache * (i - 1),
      ((double*) other_ptr) + elems_per_cache,
      outs_per_cache * sizeof(double));
    #if (defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)) && DMA_INDICES
    snrt_dma_start_1d(((int*) idx) + outs_per_cache * (i - 1),
      idx_ptr,
      outs_per_cache * sizeof(int));
    #endif
    snrt_cluster_hw_barrier(); // 2: wait for computation to finish in this_ptr

    snrt_dma_start_1d(((double*) out) + outs_per_cache * i,
      ((double*) this_ptr) + elems_per_cache,
      outs_left * sizeof(double));
    #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
      #if DMA_INDICES
      idx_ptr = this_ptr + (elems_per_cache + outs_per_cache) * sizeof(double);
      #else
      idx_ptr = idx + outs_per_cache * i;
      #endif
    #endif

    #if (defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)) && DMA_INDICES
    snrt_dma_start_1d(((int*) idx) + outs_per_cache * i,
      idx_ptr,
      outs_left * sizeof(int));
    #endif
    snrt_cluster_hw_barrier(); // 3: all done

  }
  if (snrt_is_compute_core()) {
    #if ENABLE_BENCHMARKING
    snrt_mcycle();
    #endif
    // do computation on data in other_ptr
    #if MAXPOOL_DIM == 1
    MAXPOOL_FN_1D(&copy,
      (double*) other_ptr,
      ((double*) other_ptr) + elems_per_cache,
      (int*) idx_ptr,
      compute_id,
      compute_num,
      outs_per_cache);
    #elif MAXPOOL_DIM == 2
    MAXPOOL_FN_2D(&copy,
      (double*) other_ptr,
      ((double*) other_ptr) + elems_per_cache,
      (int*) idx_ptr,
      compute_id,
      compute_num,
      outs_per_cache);
    #else
    MAXPOOL_FN_3D(&copy,
      (double*) other_ptr,
      ((double*) other_ptr) + elems_per_cache,
      (int*) idx_ptr,
      compute_id,
      compute_num,
      outs_per_cache);
    #endif
    #if ENABLE_BENCHMARKING
    snrt_mcycle();
    #endif

    snrt_cluster_hw_barrier(); // 1

    // do remaining computation in this_ptr

    #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
      #if DMA_INDICES
      idx_ptr = this_ptr + (elems_per_cache + outs_per_cache) * sizeof(double);
      #else
      idx_ptr = idx + outs_per_cache * i;
      #endif
    #endif

    copy.input_shape[0] = batches_left;
    copy.input_shape[1] = 1;
    copy.output_shape[0] = batches_left;
    copy.output_shape[1] = 1;

    #if ENABLE_BENCHMARKING
    snrt_mcycle();
    #endif
    #if MAXPOOL_DIM == 1
    MAXPOOL_FN_1D(&copy,
      (double*) this_ptr,
      ((double*) this_ptr) + elems_per_cache,
      (int*) idx_ptr,
      compute_id,
      compute_num,
      outs_left);
    #elif MAXPOOL_DIM == 2
    MAXPOOL_FN_2D(&copy,
      (double*) this_ptr,
      ((double*) this_ptr) + elems_per_cache,
      (int*) idx_ptr,
      compute_id,
      compute_num,
      outs_left);
    #else
    MAXPOOL_FN_3D(&copy,
      (double*) this_ptr,
      ((double*) this_ptr) + elems_per_cache,
      (int*) idx_ptr,
      compute_id,
      compute_num,
      outs_left);
    #endif
    #if ENABLE_BENCHMARKING
    snrt_mcycle();
    #endif

    snrt_fpu_fence();
    snrt_cluster_hw_barrier(); // 2

    snrt_cluster_hw_barrier(); // 3: all done

  }

  #endif

}

void MAXPOOL_FN_1D(maxpool_attributes* attr,
                   double* in,
                   double* out,
                   int* idx,
                   int start_step,
                   int n_cores,
                   int end_step) {

  #if ENABLE_SPECIALIZED && !defined(MAXPOOL_ROW_MAJOR) && !defined(MAXPOOL_COL_MAJOR)
  // The optimized algorithm doesn't work if there are kernels that touch padding on both sides.
  if ((attr->kernel_shape[0] - 1) * attr->dilations[0] + 1 <= attr->input_shape[2]) {
    // Due to us double buffering only an integer number of matrices,
    // we are therefore called to process an integer number of matrices.
    int input_size = attr->input_shape[2];
    int pooled_size = attr->output_shape[2];
    int n_channels = attr->input_shape[0] * attr->input_shape[1];

    int new_pooled_size = pooled_size;
    double* in_copy = in;
    double* out_copy = out;

    if (attr->pads[0] != 0 || attr->pads[1] != 0) {

      int work_n_channels = n_channels / n_cores;
      if (start_step < n_channels % n_cores) ++work_n_channels;

      if (work_n_channels > 0) {

        // If we have padding we special case the first and last kernel(s) of each channel.
        // If stride is less than padding it's possible multiple kernels at the start will include padding.
        // Same can happen at the end regardless of stride.
        if (attr->pads[0] != 0) {
          int padding_left = attr->pads[0];
          int out_offset = 0;
          while (padding_left > 0) {
            int n_skip = ceil_div(padding_left, attr->dilations[0]);

            int kernel_left = attr->kernel_shape[0] - n_skip;

            snrt_ssr_loop_2d(SNRT_SSR_DM0,
              kernel_left,
              work_n_channels,
              attr->dilations[0] * sizeof(double),
              n_cores * input_size * sizeof(double));

            snrt_ssr_loop_1d(SNRT_SSR_DM1,
              work_n_channels,
              n_cores * pooled_size * sizeof(double));

            int in_offset = neg_mod(padding_left, attr->dilations[0]);
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, in + in_offset + start_step * input_size);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out + out_offset + start_step * pooled_size);

            snrt_ssr_enable();

            const register int n_frep = kernel_left - 2;
            const register int total_iters = work_n_channels;

            ssr_asm_no_index_optimized(n_frep, total_iters);

            snrt_ssr_disable();
            snrt_fpu_fence();

            if (new_pooled_size == 1) return;
            --new_pooled_size;
            ++out_offset;
            padding_left -= attr->strides[0];

          }

        }

        if (attr->pads[1] != 0) {

          // Get an upper bound on the number of kernels that might be affected by rightside padding.
          // There might be a more exact calculation but it seems complicated.
          int padded_kernels = ceil_div(attr->kernel_shape[0] * attr->dilations[0] - 1, attr->strides[0]);

          for (int i = 0; i < padded_kernels; ++i) {

            // Could be negative if kernels use padding on both sides.
            int extra = neg_mod(attr->pads[0], attr->strides[0]);
            int offset = (pooled_size - padded_kernels + i - ceil_div(attr->pads[0], attr->strides[0])) * attr->strides[0] + extra;

            // How many vals of actual data we have before reaching padding.
            int vals_left = min(attr->kernel_shape[0], ceil_div(input_size - offset, attr->dilations[0]));

            snrt_ssr_loop_2d(SNRT_SSR_DM0,
              vals_left,
              work_n_channels,
              attr->dilations[0] * sizeof(double),
              n_cores * input_size * sizeof(double));

            snrt_ssr_loop_1d(SNRT_SSR_DM1,
              work_n_channels,
              n_cores * pooled_size * sizeof(double));
            
            snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, in_copy + offset + start_step * input_size);
            snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out_copy + pooled_size - padded_kernels + i + start_step * pooled_size);

            snrt_ssr_enable();

            const register int n_frep = vals_left - 2;
            const register int total_iters = work_n_channels;

            ssr_asm_no_index_optimized(n_frep, total_iters);

            snrt_ssr_disable();
            snrt_fpu_fence();

            --new_pooled_size;
            if (new_pooled_size < 1) return;

          }

        }
      }

      if (attr->pads[0] != 0) {
        int n_front_pad = ceil_div(attr->pads[0], attr->strides[0]);
        in += -attr->pads[0] + n_front_pad * attr->strides[0];
        out += n_front_pad;
      }
    }

    // The default will try to distribute kernels of a single matrix evenly between cores.
    // If the pool size is less than number of cores work will be distributed unevenly.
    // If n_channels >= n_cores then we can make it almost optimal by distributing entire
    // matrices between the cores. This strategy works at scale,
    // we could additionally implement special cases for very small inputs if desired.
    if (/*pooled_size < n_cores && n_channels >= n_cores*/1) {

      int work_n_channels = n_channels / n_cores;
      if (start_step < n_channels % n_cores) ++work_n_channels;
      if (work_n_channels < 1) return;

      // new_pooled_size represents the number of outputs still needed for each channel after
      // precomputing kernels that involve padding.
      snrt_ssr_loop_3d(SNRT_SSR_DM0,
        attr->kernel_shape[0],
        new_pooled_size,
        work_n_channels,
        attr->dilations[0] * sizeof(double),
        attr->strides[0] * sizeof(double),
        n_cores * input_size * sizeof(double));

      snrt_ssr_loop_2d(SNRT_SSR_DM1,
        new_pooled_size,
        work_n_channels,
        sizeof(double),
        n_cores * pooled_size * sizeof(double));

      snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, in + start_step * input_size);
      snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, out + start_step * pooled_size);

      snrt_ssr_enable();

      const register int n_frep = attr->kernel_shape[0] - 2;
      const register int total_iters = new_pooled_size * work_n_channels;

      // frep performs kernel shape fmax ops, total of work_per_channel frep's
      ssr_asm_no_index_optimized(n_frep, total_iters);

      snrt_ssr_disable();
      snrt_fpu_fence();
      return;

    }

    int work_per_channel = new_pooled_size / n_cores;
    if (start_step < new_pooled_size % n_cores) ++work_per_channel;
    // Can happen when pooled_size < n_cores and we don't catch it with a special case
    if (work_per_channel < 1) return;

    // innermost iters, inner iters, outer iters, innermost stride, inner stride, outer stride
    snrt_ssr_loop_3d(SNRT_SSR_DM0,
      attr->kernel_shape[0],
      work_per_channel,
      n_channels,
      attr->dilations[0] * sizeof(double),
      attr->strides[0] * sizeof(double) * n_cores,
      input_size * sizeof(double));

    snrt_ssr_loop_2d(SNRT_SSR_DM1,
      work_per_channel,
      n_channels,
      n_cores * sizeof(double),
      pooled_size * sizeof(double));

    // Start won't be after the first channel since work_per_channel would be 0 and we would early return.
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, in + start_step * attr->strides[0]);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, out + start_step);

    snrt_ssr_enable();

    const register int n_frep = attr->kernel_shape[0] - 2;
    const register int total_iters = work_per_channel * n_channels;

    // frep performs kernel shape fmax ops, total of work_per_channel frep's
    ssr_asm_no_index_optimized(n_frep, total_iters);

    snrt_ssr_disable();
    snrt_fpu_fence();
    
    return;
  }
  #endif

  int height = attr->input_shape[2];
  int x_step = height;

  int pooled_height = attr->output_shape[2];
  int y_step = pooled_height;

  // int n_steps = total_channels * pooled_height;
  for (int step = start_step; step < end_step; step += n_cores) {

    int i = step / pooled_height;
    int ph = step % pooled_height;

    int x_d = i * x_step;
    int y_d = i * y_step;

    int hstart = ph * attr->strides[0] - attr->pads[0];
    int hend = min(hstart + attr->kernel_shape[0] * attr->dilations[0], height);

    if (hstart < 0) {
      hstart = neg_mod(hstart, attr->dilations[0]);
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

    snrt_ssr_loop_1d(SNRT_SSR_DM0, n_iter, sizeof(double) * attr->dilations[0]);
    snrt_ssr_loop_1d(SNRT_SSR_DM1, 1, 0); // value output

    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, in + x_d + hstart);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out + y_d + ph);

    snrt_ssr_enable();

    #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)
    ssr_asm_with_index(&h_index, n_iter - 1);
    snrt_ssr_disable();
    snrt_fpu_fence();
    idx[y_d + ph] = hstart + h_index * attr->dilations[0];
    #else
    ssr_asm_no_index(n_iter - 2);
    snrt_ssr_disable();
    snrt_fpu_fence();
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

void MAXPOOL_FN_2D(maxpool_attributes* attr,
                   double* in,
                   double* out,
                   int* idx,
                   int start_step,
                   int n_cores,
                   int end_step) {

  #if ENABLE_SPECIALIZED && !defined(MAXPOOL_ROW_MAJOR) && !defined(MAXPOOL_COL_MAJOR)

  // Check the special case of a very nice input: 1 dilation, stride = kernel, no padding, kernel tiles perfectly.
  // If the input is like this we can reduce the dimensions of the loop by 1 and do it all in SSR.
  if (attr->pads[0] == 0 && attr->pads[1] == 0 && attr->pads[2] == 0 && attr->pads[3] == 0 &&
      attr->dilations[0] == 1 && attr->dilations[1] == 1 &&
      attr->strides[0] == attr->kernel_shape[0] && attr->strides[1] == attr->kernel_shape[1] &&
      attr->input_shape[2] % attr->kernel_shape[0] == 0 && attr->input_shape[3] % attr->kernel_shape[1] == 0 &&
      // We can probably use a different algorithm to make this efficient with fewer channels,
      // but this is enough to demonstrate the optimal case.
      attr->input_shape[0] * attr->input_shape[1] >= n_cores) {

    int input_size = attr->input_shape[2] * attr->input_shape[3];
    int pooled_size = attr->output_shape[2] * attr->output_shape[3];
    int n_channels = attr->input_shape[0] * attr->input_shape[1];
    int total_rows = n_channels * attr->output_shape[2];

    int in_h = attr->input_shape[2];
    int in_w = attr->input_shape[3];
    int out_h = attr->output_shape[2];
    int out_w = attr->output_shape[3];

    int work_n_rows = total_rows / n_cores;
    if (start_step < total_rows % n_cores) ++work_n_rows;
    if (work_n_rows < 1) return;

    snrt_ssr_loop_4d(SNRT_SSR_DM0,
      attr->kernel_shape[1],
      attr->kernel_shape[0],
      out_w,
      work_n_rows,
      attr->dilations[1] * sizeof(double),
      in_w * attr->dilations[0] * sizeof(double),
      attr->strides[1] * sizeof(double),
      n_cores * in_w * attr->strides[0] * sizeof(double));
    
    snrt_ssr_loop_2d(SNRT_SSR_DM1,
      out_w,
      work_n_rows,
      sizeof(double),
      n_cores * out_w * sizeof(double));
    
    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, in + start_step * in_w * attr->strides[0]);
    snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, out + start_step * out_w);

    snrt_ssr_enable();

    const register int n_frep = attr->kernel_shape[0] * attr->kernel_shape[1] - 2;
    const register int total_iters = work_n_rows * out_w;

    ssr_asm_no_index_optimized(n_frep, total_iters);

    snrt_ssr_disable();
    snrt_fpu_fence();

    return;

  }

  // Inputs that have a kernel affected by multiple paddings is not supported.
  // A more complicated special casing algorithm may resolve this.
  if ((attr->kernel_shape[0] - 1) * attr->dilations[0] + 1 <= attr->input_shape[2] && (attr->kernel_shape[1] - 1) * attr->dilations[1] + 1 <= attr->input_shape[3]) {
    
    int input_size = attr->input_shape[2] * attr->input_shape[3];
    int pooled_size = attr->output_shape[2] * attr->output_shape[3];
    int n_channels = attr->input_shape[0] * attr->input_shape[1];

    int in_h = attr->input_shape[2];
    int in_w = attr->input_shape[3];
    int out_h = attr->output_shape[2];
    int out_w = attr->output_shape[3];

    if (attr->pads[0] != 0 || attr->pads[1] != 0 || attr->pads[2] != 0 || attr->pads[3] != 0) {

      int height = attr->input_shape[2];
      int width = attr->input_shape[3];
      int x_step = height * width;

      int pooled_height = attr->output_shape[2];
      int pooled_width = attr->output_shape[3];
      int y_step = pooled_size;

      // The bottom and right bounds are a worst case. They can probably be tightened with a better formula.
      int pad_top = ceil_div(attr->pads[0], attr->strides[0]);
      int pad_bot_remove = ceil_div(attr->kernel_shape[0] * attr->dilations[0] - 1, attr->strides[0]);
      int pad_bot_bound = attr->output_shape[2] - pad_bot_remove;
      int pad_left = ceil_div(attr->pads[1], attr->strides[1]);
      int pad_right_remove = ceil_div(attr->kernel_shape[1] * attr->dilations[1] - 1, attr->strides[1]);
      int pad_right_bound = attr->output_shape[3] - pad_right_remove;
      // if (start_step == 0) DUMP(pad_top);
      // if (start_step == 0) DUMP(ceil_div(attr->kernel_shape[1] - 1, attr->strides[1]));
      // if (start_step == 0) DUMP(pad_bot_bound);

      // if (start_step == 0) DUMP(pad_left);
      // if (start_step == 0) DUMP(ceil_div(attr->kernel_shape[0] - 1, attr->strides[0]));
      // if (start_step == 0) DUMP(pad_right_bound);

      // This is just adapted from the old version of maxpool with SSR/FREP.
      // It's pretty inefficient for computing the kernels with padding and wastes lots of iterations.
      // We could alternatively split the matrix into 8 sections along the edges and vertices.
      // That lets us batch the four edges, but it is pretty tedious so it is not done here.
      for (int step = start_step; step < end_step; step += n_cores) {

        int i = step / y_step;
        int inst_idx = step % y_step;
        int ph = inst_idx / pooled_width;
        int pw = inst_idx % pooled_width;
        
        if (ph >= pad_top && ph < pad_bot_bound && pw >= pad_left && pw < pad_right_bound) continue;

        int x_d = i * x_step;
        int y_d = i * y_step;

        int hstart = ph * attr->strides[0] - attr->pads[0];
        int hend = min(hstart + attr->kernel_shape[0] * attr->dilations[0], height);

        if (hstart < 0) {
          hstart = neg_mod(hstart, attr->dilations[0]);
        }

        int wstart = pw * attr->strides[1] - attr->pads[1];
        int wend = min(wstart + attr->kernel_shape[1] * attr->dilations[1], width);

        if (wstart < 0) {
          wstart = neg_mod(wstart, attr->dilations[1]);
        }

        int pool_index = ph * pooled_width + pw;

        int n_iter_h = (hend - hstart + attr->dilations[0] - 1) / attr->dilations[0];
        int n_iter_w = (wend - wstart + attr->dilations[1] - 1) / attr->dilations[1];
        if (n_iter_h * n_iter_w == 1) {
          out[y_d + pool_index] = in[x_d + hstart * width + wstart];
          continue;
        }

        snrt_ssr_loop_2d(SNRT_SSR_DM0, n_iter_w, n_iter_h, sizeof(double) * attr->dilations[1], sizeof(double) * attr->dilations[0] * width);
        snrt_ssr_loop_1d(SNRT_SSR_DM1, 1, 0); // value output

        snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_2D, in + x_d + hstart * width + wstart);
        snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out + y_d + pool_index);

        snrt_ssr_enable();
        ssr_asm_no_index(n_iter_h * n_iter_w - 2);
        snrt_ssr_disable();
        snrt_fpu_fence();

      }

      int top_offset = -attr->pads[0] + pad_top * attr->strides[0];
      int left_offset = -attr->pads[1] + pad_left * attr->strides[1];
      // if (start_step == 0) DUMP(top_offset);
      // if (start_step == 0) DUMP(left_offset);

      in += in_w * top_offset + left_offset;
      out += out_w * pad_top + pad_left;
      // if (start_step == 0) DUMP(out_w * pad_top + pad_left);
      // if (start_step == 0) DUMP(out_h);
      // if (start_step == 0) DUMP(out_w);

      out_h -= pad_top;
      out_h -= pad_bot_remove;

      out_w -= pad_left;
      out_w -= pad_right_remove;

      if (out_h <= 0 || out_w <= 0) return;

      // if (start_step == 0) DUMP(out_h);
      // if (start_step == 0) DUMP(out_w);

      // return;
    }

    // The first and simplest strategy is to distribute channels evenly. This is close to optimal at scale
    // as work is unbalanced by at most one channel between any two cores.
    // We are limited to 4D SSR so the channel iteration must be a traditional loop.
    for (int i = start_step; i < n_channels; i += n_cores) {

      snrt_ssr_loop_4d(SNRT_SSR_DM0,
        attr->kernel_shape[1],
        attr->kernel_shape[0],
        out_w,
        out_h,
        attr->dilations[1] * sizeof(double),
        in_w * attr->dilations[0] * sizeof(double),
        attr->strides[1] * sizeof(double),
        in_w * attr->strides[0] * sizeof(double));
      
      // 2D is necessary in case there is padding
      snrt_ssr_loop_2d(SNRT_SSR_DM1,
        out_w,
        out_h,
        sizeof(double),
        attr->output_shape[3] * sizeof(double));
      
      snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, in + input_size * i);
      snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_2D, out + pooled_size * i);

      snrt_ssr_enable();

      const register int n_frep = attr->kernel_shape[0] * attr->kernel_shape[1] - 2;
      const register int total_iters = out_w * out_h;

      ssr_asm_no_index_optimized(n_frep, total_iters);

      snrt_ssr_disable();
      snrt_fpu_fence();

    }

    return;

    // Another distribution strategy can be to share work evenly in a single channel.
    // We can't use a naive method of work splitting due to limitations of SSR stride.
    // We could distribute each row evenly, but that results in uneven distribution when
    // number of columns is not a multiple of n_cores.
    // An even distribution strategy could be to distribute each row evenly up to the
    // closest multiple of n_cores <= the number of columns, then distribute the remaining columns
    // up to the multiple of n_cores <= number of rows. Then each kernel of the remainder is distributed
    // to one core each. This strategy requires 3 SSR rotations per channel, however.

  }
  #endif

  int height = attr->input_shape[2];
  int width = attr->input_shape[3];
  int x_step = height * width;

  // int total_els = total_channels * height * width;

  int pooled_height = attr->output_shape[2];
  int pooled_width = attr->output_shape[3];
  int y_step = pooled_height * pooled_width;

  // int n_steps = total_channels * y_step;
  for (int step = start_step; step < end_step; step += n_cores) {

    int i = step / y_step;
    int inst_idx = step % y_step;
    int ph = inst_idx / pooled_width;
    int pw = inst_idx % pooled_width;

    int x_d = i * x_step;
    int y_d = i * y_step;

    int hstart = ph * attr->strides[0] - attr->pads[0];
    int hend = min(hstart + attr->kernel_shape[0] * attr->dilations[0], height);

    if (hstart < 0) {
      hstart = neg_mod(hstart, attr->dilations[0]);
    }

    int wstart = pw * attr->strides[1] - attr->pads[1];
    int wend = min(wstart + attr->kernel_shape[1] * attr->dilations[1], width);

    if (wstart < 0) {
      wstart = neg_mod(wstart, attr->dilations[1]);
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
      snrt_fpu_fence();

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
        snrt_fpu_fence();
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

void MAXPOOL_FN_3D(maxpool_attributes* attr,
                   double* in,
                   double* out,
                   int* idx,
                   int start_step,
                   int n_cores,
                   int end_step) {

  // if (attr->n_dim != 3) return; // error

  // int total_channels = attr->input_shape[0] * attr->input_shape[1]; // batch size * num channels

  int height = attr->input_shape[2];
  int width = attr->input_shape[3];
  int depth = attr->input_shape[4];
  int x_step = height * width * depth;

  int pooled_height = attr->output_shape[2];
  int pooled_width = attr->output_shape[3];
  int pooled_depth = attr->output_shape[4];
  int y_step = pooled_height * pooled_width * pooled_depth;

  // int n_steps = total_channels * y_step;
  for (int step = start_step; step < end_step; step += n_cores) {

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
    int hend = min(hstart + attr->kernel_shape[0] * attr->dilations[0], height);

    if (hstart < 0) {
      hstart = neg_mod(hstart, attr->dilations[0]);
    }

    int wstart = pw * attr->strides[1] - attr->pads[1];;
    int wend = min(wstart + attr->kernel_shape[1] * attr->dilations[1], width);

    if (wstart < 0) {
      wstart = neg_mod(wstart, attr->dilations[1]);
    }

    int dstart = pd * attr->strides[2] - attr->pads[2];
    int dend = min(dstart + attr->kernel_shape[2] * attr->dilations[2], depth);

    if (dstart < 0) {
      dstart = neg_mod(dstart, attr->dilations[2]);
    }

    int pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;

    #if USE_SSR_FREP_3D || USE_SSR_FREP_ALL

      int max_index;

      int n_iter_h = (hend - hstart + attr->dilations[0] - 1) / attr->dilations[0];
      int n_iter_w = (wend - wstart + attr->dilations[1] - 1) / attr->dilations[1];
      int n_iter_d = (dend - dstart + attr->dilations[2] - 1) / attr->dilations[2];
      if (n_iter_h * n_iter_w * n_iter_d == 1) {
        out[y_d + pool_index] = in[x_d + hstart * width * depth + wstart * depth + dstart];
        #if defined(MAXPOOL_COL_MAJOR)
          idx[y_d + pool_index] = hstart + wstart * height + dstart * height * width;
        #elif defined(MAXPOOL_ROW_MAJOR)
          idx[y_d + pool_index] = hstart * width * depth + wstart * depth + dstart;
        #endif
        continue;
      }

      snrt_ssr_loop_3d(
        SNRT_SSR_DM0,
        n_iter_d,
        n_iter_w,
        n_iter_h,
        sizeof(double) * attr->dilations[2],
        sizeof(double) * attr->dilations[1] * depth,
        sizeof(double) * attr->dilations[0] * depth * width);
      snrt_ssr_loop_1d(SNRT_SSR_DM1, 1, 0); // value output

      snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_3D, in + x_d + hstart * width * depth + wstart * depth + dstart);
      snrt_ssr_write(SNRT_SSR_DM1, SNRT_SSR_1D, out + y_d + pool_index);

      snrt_ssr_enable();

      #if defined(MAXPOOL_ROW_MAJOR) || defined(MAXPOOL_COL_MAJOR)

        ssr_asm_with_index(&max_index, n_iter_d * n_iter_h * n_iter_w - 1);
        snrt_ssr_disable();
        snrt_fpu_fence();

        // int h_index = max_index / (n_iter_w * n_iter_d) * attr->dilations[0] + hstart;
        // max_index -= ph * pooled_width * pooled_depth;
        // int w_index = (max_index / n_iter_d) * attr->dilations[1] + wstart;
        // int d_index = (max_index % n_iter_d) * attr->dilations[2] + dstart;
        int h_index = (max_index / (n_iter_w * n_iter_d)) * attr->dilations[0] + hstart;
        int w_index = ((max_index / n_iter_d) % n_iter_w) * attr->dilations[1] + wstart;
        int d_index = (max_index % n_iter_d) * attr->dilations[2] + dstart;

        #ifdef MAXPOOL_COL_MAJOR
        idx[y_d + pool_index] = h_index + w_index * height + d_index * height * width;
        #else
        idx[y_d + pool_index] = h_index * width * depth + w_index * depth + d_index;
        #endif

      #else
        ssr_asm_no_index(n_iter_d * n_iter_h * n_iter_w - 2);
        snrt_ssr_disable();
        snrt_fpu_fence();

      #endif

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

#endif
