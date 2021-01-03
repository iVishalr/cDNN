#include "./Dense.h"

void init_params(Dense_layer * layer){
  int weight_dims[] = {layer->num_of_computation_nodes,10};
  int bias_dims[] = {layer->num_of_computation_nodes,1};
  layer->weights = init_weights(weight_dims,layer->initializer);
  layer->bias = init_bias(bias_dims);
}

dARRAY * init_weights(int * weights_dims,const char * init_type){
  dARRAY * weights = randn(weights_dims);
  if(!strcmp(init_type,"he")){
    weights = mulScalar(weights,sqrt(2/5));
  }
  else if(!strcmp(init_type,"xavier")){
    weights = mulScalar(weights,sqrt(2/(5+weights_dims[0])));
  }
  else{
    weights = mulScalar(weights,0.01);
  }
  return weights;
}

dARRAY * init_bias(int * bias_dims){
  dARRAY * bias = zeros(bias_dims);
  return bias;
}

void forward_pass(Dense_layer * layer, Dense_layer * prev_layer){
  //TODO
}
void backward_pass(Dense_layer * layer, Dense_layer * prev_layer){
  //TODO
}

void (Dense)(dense_args dense_layer_args){
  Dense_layer * layer = (Dense_layer*)malloc(sizeof(Dense_layer));
  layer->activation = dense_layer_args.activation;
  layer->layer_num = 0;
  layer->num_of_computation_nodes = dense_layer_args.layer_size;
  layer->layer_type = "normal";
  layer->initalize_params = init_params;
  layer->initializer = dense_layer_args.initializer;
  layer->cache = NULL;
  layer->grad_in = NULL;
  layer->grad_out = NULL;
  layer->forward_prop = forward_pass;
  layer->back_prop = backward_pass;
  //finally we need to append to computation graph;
}

int main(){
  Dense(.layer_size=100,.activation="tanh");
}