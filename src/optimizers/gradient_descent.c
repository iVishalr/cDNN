#include "gradient_descent.h"
#include "../model/model.h"

extern __Model__ * m;

void SGD(){
  Computation_Graph * temp = m->graph;
  while(temp!=NULL){
    if(temp->type==INPUT || temp->type==LOSS){
      temp = temp->next_layer;
      continue;
    }

    dARRAY * update_term1_dW = mulScalar(temp->DENSE->dW,m->learning_rate);
    dARRAY * update_term1_db = mulScalar(temp->DENSE->db,m->learning_rate);

    free2d(temp->DENSE->dW);
    free2d(temp->DENSE->db);
    temp->DENSE->dW = NULL;
    temp->DENSE->db = NULL;

    if(m->regularization!=NULL){
      float mul_grad = m->lambda * m->learning_rate/(float)m->y_train_mini_batch[m->current_mini_batch]->shape[1];
      dARRAY * update_term2_dW = mulScalar(temp->DENSE->weights,mul_grad);
      dARRAY * update_term2_db = mulScalar(temp->DENSE->bias,mul_grad);

      dARRAY * ptr_update_term1_dW = update_term1_dW;
      dARRAY * ptr_update_term1_db = update_term1_db;

      update_term1_dW = add(ptr_update_term1_dW,update_term2_dW);
      update_term1_db = add(ptr_update_term1_db,update_term2_db);

      free2d(update_term2_dW);
      free2d(update_term2_db);
      free2d(ptr_update_term1_dW);
      free2d(ptr_update_term1_db);
      update_term2_dW = NULL;
      update_term2_db = NULL;
      ptr_update_term1_dW = NULL;
      ptr_update_term1_db = NULL;
    }

    dARRAY * ptr_layer_W = temp->DENSE->weights;
    dARRAY * ptr_layer_b = temp->DENSE->bias;

    temp->DENSE->weights = subtract(ptr_layer_W,update_term1_dW);
    temp->DENSE->bias = subtract(ptr_layer_b,update_term1_db);

    free2d(update_term1_dW);
    free2d(update_term1_db);
    update_term1_dW = NULL;
    update_term1_db = NULL;
    
    free2d(ptr_layer_W);
    free2d(ptr_layer_b);
    ptr_layer_W = NULL;
    ptr_layer_b = NULL;
    
    if(temp->DENSE->dropout_mask!=NULL && temp->DENSE->dropout<(float)1.0)
      free2d(temp->DENSE->dropout_mask);
    if(temp->DENSE->A!=NULL)
      free2d(temp->DENSE->A);
    if(temp->DENSE->cache!=NULL)
      free2d(temp->DENSE->cache);
    if(temp->DENSE->dA!=NULL)
      free2d(temp->DENSE->dA);
    if(temp->DENSE->dZ!=NULL)
      free2d(temp->DENSE->dZ);
    temp->DENSE->dA = NULL;
    temp->DENSE->cache = NULL;
    temp->DENSE->dropout_mask = NULL;
    temp->DENSE->A = NULL;
    m->output = NULL;
    temp->DENSE->dZ = NULL;
    temp = temp->next_layer;
  }
}