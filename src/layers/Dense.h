#include "../utils/Activations/activations.h"

typedef void * (*__init_params)(dARRAY * layer);
typedef void * (*__compute)(dARRAY * layer, dARRAY * prev_layer);

typedef struct dense{
  dARRAY * weights;
  dARRAY * bias;
  int layer_num;
  char * activation;
  int num_of_computation_nodes;
  char * layer_type;
  dARRAY * cache;
  dARRAY * grad_in;
  dARRAY * grad_out;
  __compute forward_prop;
  __compute back_prop;
  __init_params initalize_params;
  char * initializer;
}Dense_layer;




typedef struct{
  int layer_size;
  char * activation;
  char * initializer;
}dense_args; 

#define Dense(...) Dense((dense_args){.layer_size=20,.activation="relu",.initializer="he",__VA_ARGS__});
void (Dense)(dense_args dense_layer_args);
void init_params(Dense_layer * layer);
dARRAY * init_weights(int * weights_dims,const char * init_type);
dARRAY * init_bias(int * bias_dims);
void forward_pass(Dense_layer * layer, Dense_layer * prev_layer);
void backward_pass(Dense_layer * layer, Dense_layer * prev_layer);
