enum padding_type {
  NOTSET,
  SAME_UPPER,
  SAME_LOWER,
  VALID
};

typedef struct maxpool_attributes_struct {
  int n_dim; // excludes batch size and # channels
  int input_shape[5]; // INCLUDES batch size and # channels, 3d worst case
  int output_shape[5]; // INCLUDES batch size and # channels, 3d worst case

  enum padding_type auto_pad; // NOTSET, deprecated in maxpool v12
  int ceil_mode; // false
  int dilations[3]; // [1...1]
  int kernel_shape[3]; // required
  int pads[6]; // [0...0], [dim1_begin, dim2_begin, dim1_end, dim2_end]
  int storage_order; // false (row major)
  int strides[3]; // [1...1]
} maxpool_attributes;

// void maxpool_fp64(maxpool_attributes* attr, double* in, double* out, int* idx);

void populate_defaults(maxpool_attributes*, int);

void compute_output_shape(maxpool_attributes*, int*);

void maxpool_fp64_1d(maxpool_attributes*, double*, double*, int*, int, int);

void maxpool_fp64_1d_layer(maxpool_attributes*, double*, double*, int*);
