#include "./adagrad.h"
#include "../model/model.h"

extern __Model__ * m;

void adagrad(){
  Computation_Graph * temp = m->graph->next_layer;
  int layer=0;
  while(temp!=NULL){
    dARRAY * scaled_grads_dW = power(temp->DENSE->dW,2);
    dARRAY * scaled_grads_db = power(temp->DENSE->db,2);
    dARRAY * ptr_cache_dW = m->cache_dW[layer];
    dARRAY * ptr_cache_db = m->cache_db[layer];

    m->cache_dW[layer] = add(ptr_cache_dW,scaled_grads_dW);
    m->cache_db[layer] = add(ptr_cache_db,scaled_grads_db);

    free2d(scaled_grads_dW);
    free2d(scaled_grads_db);
    free2d(ptr_cache_dW);
    free2d(ptr_cache_db);
    scaled_grads_dW = scaled_grads_db = ptr_cache_dW = ptr_cache_db = NULL;

    dARRAY * div_factor_temp_dW = power(m->cache_dW[layer],0.5);
    dARRAY * div_factor_temp_db = power(m->cache_db[layer],0.5);

    dARRAY * div_factor_dW = addScalar(div_factor_temp_dW,m->epsilon);
    dARRAY * div_factor_db = addScalar(div_factor_db,m->epsilon);

    free2d(div_factor_temp_dW);
    free2d(div_factor_temp_db);
    div_factor_temp_dW = div_factor_temp_db = NULL;

    dARRAY * mul_lr_w = mulScalar(temp->DENSE->dW,m->learning_rate);
    dARRAY * mul_lr_b = mulScalar(temp->DENSE->bias,m->learning_rate);

    dARRAY * update_term_w = divison(mul_lr_w,div_factor_dW);
    dARRAY * update_term_b = divison(mul_lr_b,div_factor_db);

    free2d(mul_lr_w);
    free2d(mul_lr_b);
    free2d(div_factor_dW);
    free2d(div_factor_db);
    mul_lr_w = mul_lr_b = div_factor_dW = div_factor_db = NULL;

    dARRAY * grad_W = temp->DENSE->weights;
    dARRAY * grad_b = temp->DENSE->bias;

    temp->DENSE->weights = subtract(grad_W,update_term_w);
    temp->DENSE->bias = subtract(grad_b,update_term_b);

    free2d(grad_W);
    free2d(grad_b);
    free2d(update_term_w);
    free2d(update_term_b);
    grad_W = grad_b = update_term_w = update_term_b = NULL;

    if(temp->DENSE->dropout_mask!=NULL)
      free2d(temp->DENSE->dropout_mask);
    if(temp->DENSE->A!=NULL)
      free2d(temp->DENSE->A);
    if(temp->DENSE->cache!=NULL)
      free2d(temp->DENSE->cache);
    if(temp->DENSE->dA!=NULL)
      free2d(temp->DENSE->dA);
    if(temp->DENSE->dZ!=NULL)
      free2d(temp->DENSE->dZ);
    temp->DENSE->dA=temp->DENSE->cache = temp->DENSE->A = temp->DENSE->dropout_mask = temp->DENSE->dZ = NULL;
    
    layer++;
    temp = temp->next_layer;
  }
}