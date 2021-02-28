#include "./Input.h"
#include "../model/model.h"

extern __Model__ * m;

//This will just be an identity mapping of input features
void forward_pass_input(){
  m->graph->INPUT->A = m->x_train_mini_batch[m->current_mini_batch];
}

//Leave this empty as Input layer need not calculate any gradients
//This is just to satisfy the forward and backward API
void backward_pass_input(){ } 

void (Input)(Input_args input_layer_args){
  Input_layer * layer = (Input_layer*)malloc(sizeof(Input_layer));
  layer->input_features_size = input_layer_args.layer_size;
  layer->forward = forward_pass_input;
  layer->backward = backward_pass_input;
  //Append to computation graph
  append_graph(layer,"Input");
}