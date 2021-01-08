#ifndef DENSE_LAYER_H_
#define DENSE_LAYER_H_

#include "../utils/utils.h"
#include "../utils/Activations/activations.h"

typedef void (*__init_params)();
typedef void (*__compute)();

typedef struct dense{
  dARRAY * weights;
  dARRAY * bias;
  char * activation;
  int num_of_computation_nodes;
  dARRAY * cache;
  dARRAY * dZ;
  dARRAY * dW;
  dARRAY * db;
  dARRAY * dA;
  dARRAY * A;
  __compute forward_prop;
  __compute back_prop;
  __init_params initalize_params;
  char * initializer;
  double dropout;
  double lambda;
  dARRAY * dropout_mask;
  int isTraining;
  char * layer_type;
}Dense_layer;

typedef struct dense_arg{
  int layer_size;
  char * activation;
  char * initializer;
  double dropout;
  double lambda;
  char * layer_type;
}dense_args; 

#define Dense(...) Dense((dense_args){.layer_size=20,.activation="relu",.initializer="he",.dropout=1.0,.lambda=0.0,.layer_type="hidden",__VA_ARGS__});

#ifdef __cplusplus
extern "C" {
#endif
  void (Dense)(dense_args dense_layer_args);
  void init_params();
  dARRAY * init_weights(int * weights_dims,const char * init_type);
  dARRAY * init_bias(int * bias_dims);
  void forward_pass();
  void backward_pass();
#ifdef __cplusplus
}
#endif
#endif //DENSE_H_