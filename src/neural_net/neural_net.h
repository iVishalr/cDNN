#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "../layers/Dense.h"
#include "../layers/Input.h"
#include "../utils/utils.h"
#include "../loss_functions/cross_entropy_loss.h"
#include "../optimizers/gradient_descent.h"
#include "../plot/plot.h"

enum layer_type {NONE, INPUT, DENSE, LOSS, OPTIMIZER};
typedef struct computational_graph{
  struct computational_graph * next_layer;
  struct computational_graph * prev_layer;
  enum layer_type type;
  union
  {
    Dense_layer * DENSE;
    Input_layer * INPUT;
  };
}Computation_Graph;

#ifdef __cplusplus
extern "C"{
#endif
  Computation_Graph * new_node(void * layer, char * type);
  void append_graph(void * layer, char * type);
  void printComputation_Graph(Computation_Graph * G);
  Computation_Graph * destroy_G(Computation_Graph * G);
#ifdef __cplusplus
}
#endif

#endif //NEURAL_NET_H