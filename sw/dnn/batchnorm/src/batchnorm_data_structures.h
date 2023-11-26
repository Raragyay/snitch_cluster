#pragma once
typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    uint32_t TILE_CI;
    double *ifmap;
    double *ofmap;
    double *gamma;
    double *beta;

    float eps;
    precision_t dtype;
} batchnorm_layer_t;

typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    // uint32_t TILE_CI;
    double const *ifmap;

    double *ofmap;

    double *running_mean;
    double *running_var;

    double *weight;
    double *bias;

    float eps;
    float momentum;
    precision_t dtype;
} batchnorm_training_layer_t;

typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;

    double const *ifmap;
    double const *grad_ofmap;
    double const *running_mean;
    double const *running_var;
    double const *weight;

    double *grad_ifmap;
    double *grad_weight;
    double *grad_bias;

    float eps;
    precision_t dtype;
} batchnorm_backward_layer_t;

typedef struct {
    uint32_t CI;
    uint32_t IH;
    uint32_t IW;
    // uint32_t TILE_CI;

    double const *ifmap;
    double const *grad_ofmap;
    double const *current_mean;
    double const *current_var;
    double const *weight;

    double *grad_ifmap;
    double *grad_weight;
    double *grad_bias;

    float eps;
    precision_t dtype;
} batchnorm_backward_training_layer_t;
