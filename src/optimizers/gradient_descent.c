#include "gradient_descent.h"
#include "../model/model.h"

extern __Model__ * m;

void GD(double lr){
  Computation_Graph * temp = m->graph;
  dARRAY * layer_weights, *layer_biases, *grad_W, *grad_b;
  while(temp!=NULL){
    m->current_layer = temp;
    if(temp->type!=INPUT){
      layer_weights = temp->DENSE->weights;
      layer_biases = temp->DENSE->bias;
      grad_W = temp->DENSE->dW;
      grad_b = temp->DENSE->db;
      dARRAY * mul_lr_W = mulScalar(grad_W,lr);
      temp->DENSE->weights = subtract(layer_weights,mul_lr_W);
      free2d(layer_weights);
      free2d(mul_lr_W);
      dARRAY * mul_lr_b = mulScalar(grad_W,lr);
      temp->DENSE->bias = subtract(layer_biases,mul_lr_b);
      free2d(layer_biases);
      free2d(mul_lr_b);
      free2d(grad_W);
      free2d(grad_b);
      grad_W = grad_b = layer_weights = layer_biases = mul_lr_W = mul_lr_b = NULL;
    }
    temp = temp->next_layer;
  }
}