 /** 
  * File:    neural_net.h 
  * 
  * Author:  Vishal R (vishalr@pesu.pes.edu or vishalramesh01@gmail.com) 
  * 
  * Summary of File: 
  *   This file contains all the function headers required for creating static Computation Graphs.
  *   Also contains header files for memory management of the graph and connects all the layers in the neural network. 
  */ 

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "Dense.h"
#include "Input.h"

#include "loss.h"

#include "sgd.h"
#include "momentum.h"
#include "adam.h"
#include "adagrad.h"
#include "rmsprop.h"


enum layer_type {NONE, INPUT, DENSE, LOSS};
typedef struct computational_graph{
  struct computational_graph * next_layer;
  struct computational_graph * prev_layer;
  enum layer_type type;
  union
  {
    Dense_layer * DENSE;
    Input_layer * INPUT;
    loss_layer * LOSS;
  };
}Computation_Graph;

#ifdef __cplusplus
extern "C"{
#endif
  Computation_Graph * new_node(void * layer, char * type);
  void append_graph(void * layer, char * type);
  void printComputation_Graph(Computation_Graph * G);
  Computation_Graph * destroy_Graph(Computation_Graph * G);
#ifdef __cplusplus
}
#endif

#endif //NEURAL_NET_H