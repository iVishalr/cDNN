#include "../utils/Activations/activations.h"

typedef dARRAY * (*Compute)(dARRAY * layer_weights, dARRAY * a_prev);
typedef dARRAY * (*init_params)(dARRAY * weight, dARRAY * bias, char * activation);

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
  Compute linear_output;
  init_params initalize_params;
}Dense;

Dense * Dense__init__();
