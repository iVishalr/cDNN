#include "./Input.h"
#include "../neural_net/neural_net.h"

void (Input)(Input_args input_layer_args){
  Input_layer * layer = (Input_layer*)malloc(sizeof(Input_layer));
  layer->input_features_size = input_layer_args.layer_size;
  layer->A = input_layer_args.input_features;
  layer->isTraining = 1;
  //finally we need to append to computation graph
  append_graph(layer,"Input");
}