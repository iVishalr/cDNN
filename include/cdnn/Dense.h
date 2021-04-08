 /** 
  * File:    MSELoss.h 
  * 
  * Author:  Vishal R (vishalr@pesu.pes.edu or vishalramesh01@gmail.com) 
  * 
  * Summary of File: 
  *   This file contains all the function headers required for implementation of Dense Layers. 
  */ 

#ifndef DENSE_LAYER_H_
#define DENSE_LAYER_H_

#include "utils.h"
#include "activations.h"

typedef struct dense_arg{
  int layer_size;
  char * activation;
  char * initializer;
  float dropout;
  char * layer_type;
}dense_args; 

#ifdef __cplusplus
extern "C" {
#endif
  typedef void (*__init_params)();
  typedef void (*__compute)();
  void (Dense)(dense_args dense_layer_args);
  void init_params();
  dARRAY * init_weights(int * weights_dims,const char * init_type);
  dARRAY * init_bias(int * bias_dims);
  void forward_pass_DENSE();
  void backward_pass_DENSE();
#ifdef __cplusplus
}
#endif

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
  __compute forward;
  __compute backward;
  __init_params initalize_params;
  char * initializer;
  float dropout;
  dARRAY * dropout_mask;
  int isTraining;
  char * layer_type;
}Dense_layer;

/**!
* @brief Function : Dense - Constructor that constructs the Dense Layer.
* @param layer_size Specifies the number of nodes in the layer.
* @param activation Specifies the acitvation function to be used.
* @param initializer Specifies the type of weight initialization to be used.
* @param dropout Specifies the probability of dropping out neurons in the layer.
* @param layer_type Specifies the type of layer.
* @return void
*/ 
#define Dense(...) Dense((dense_args){.layer_size=20,.activation="relu",.initializer="he",.dropout=1.0,.layer_type="hidden",__VA_ARGS__});

#endif //DENSE_H_