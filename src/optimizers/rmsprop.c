#include "./rmsprop.h"
#include "../model/model.h"

extern __Model__ * m;
//cache = cache * decay_rate + (1-decay_rate) * dx^2
void RMSProp(){
  Computation_Graph * temp = m->graph;
  int layer = 0;
  while(temp!=NULL){
    if(temp->type==LOSS || temp->type==INPUT){
      temp = temp->next_layer;
      continue;
    }
    dARRAY * scaled_dW = power(temp->DENSE->dW,2);
    dARRAY * scaled_db = power(temp->DENSE->db,2);

    float mul_factor = 1-m->beta;
    dARRAY * wavg_term2_dW = mulScalar(scaled_dW,mul_factor);
    dARRAY * wavg_term2_db = mulScalar(scaled_db,mul_factor);

    free2d(scaled_dW);
    free2d(scaled_db);
    scaled_dW = NULL;
    scaled_db = NULL;

    mul_factor = m->beta;
    dARRAY * ptr_vt_dW = m->v_t_dW[layer];
    dARRAY * ptr_vt_db = m->v_t_db[layer];

    dARRAY * wavg_term1_dW = mulScalar(ptr_vt_dW,mul_factor);
    dARRAY * wavg_term1_db = mulScalar(ptr_vt_db,mul_factor);
    
    free2d(ptr_vt_dW);
    free2d(ptr_vt_db);
    ptr_vt_dW = NULL;
    ptr_vt_db = NULL;
    m->v_t_dW[layer] = NULL;
    m->v_t_dW[layer] = NULL;

    m->v_t_dW[layer] = add(wavg_term1_dW,wavg_term2_dW);
    m->v_t_db[layer] = add(wavg_term1_db,wavg_term2_db);

    free2d(wavg_term1_dW);
    free2d(wavg_term1_db);
    free2d(wavg_term2_dW);
    free2d(wavg_term2_db);
    wavg_term1_dW = NULL;
    wavg_term1_db = NULL;
    wavg_term2_dW = NULL;
    wavg_term2_db = NULL;

    ptr_vt_dW = m->v_t_dW[layer];
    ptr_vt_db = m->v_t_db[layer];

    dARRAY * div_factor_temp_dW = squareroot(ptr_vt_dW);
    dARRAY * div_factor_temp_db = squareroot(ptr_vt_db);

    ptr_vt_dW = NULL;
    ptr_vt_db = NULL;

    dARRAY * div_factor_dW = addScalar(div_factor_temp_dW,m->epsilon);
    dARRAY * div_factor_db = addScalar(div_factor_temp_db,m->epsilon);

    free2d(div_factor_temp_dW);
    free2d(div_factor_temp_db);
    div_factor_temp_dW = NULL;
    div_factor_temp_db = NULL;

    dARRAY * mul_lr_w = mulScalar(temp->DENSE->dW,m->learning_rate);
    dARRAY * mul_lr_b = mulScalar(temp->DENSE->db,m->learning_rate);

    dARRAY * update_term_dW = divison(mul_lr_w,div_factor_dW);
    dARRAY * update_term_db = divison(mul_lr_b,div_factor_db);

    free2d(div_factor_dW);
    free2d(div_factor_db);
    free2d(mul_lr_w);
    free2d(mul_lr_b);
    free2d(temp->DENSE->dW);
    free2d(temp->DENSE->db);
    div_factor_dW = NULL;
    div_factor_db = NULL;
    mul_lr_w = NULL;
    mul_lr_b = NULL;
    temp->DENSE->dW = NULL;
    temp->DENSE->db = NULL;

    if(m->regularization!=NULL){
      dARRAY * ptr_update_term_dW = update_term_dW;
      dARRAY * ptr_update_term_db = update_term_db;

      float mul_decay = m->lambda * m->learning_rate / (float)m->y_train_mini_batch[m->current_mini_batch]->shape[1];
      dARRAY * update_term2_dW = mulScalar(temp->DENSE->weights,mul_decay);
      dARRAY * update_term2_db = mulScalar(temp->DENSE->bias,mul_decay);

      update_term_dW = add(ptr_update_term_dW,update_term2_dW);
      update_term_db = add(ptr_update_term_db,update_term2_db);

      free2d(update_term2_dW);
      free2d(update_term2_db);
      free2d(ptr_update_term_dW);
      free2d(ptr_update_term_db);
      update_term2_dW = NULL;
      update_term2_db = NULL;
      ptr_update_term_dW = NULL;
      ptr_update_term_db = NULL;
    }

    dARRAY * grad_W = temp->DENSE->weights;
    dARRAY * grad_b = temp->DENSE->bias;

    temp->DENSE->weights = subtract(grad_W,update_term_dW);
    temp->DENSE->bias = subtract(grad_b,update_term_db);

    free2d(grad_W);
    free2d(grad_b);
    free2d(update_term_dW);
    free2d(update_term_db);
    grad_W = NULL;
    grad_b = NULL;
    update_term_dW = NULL;
    update_term_db = NULL;

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
    temp->DENSE->dA = NULL;
    temp->DENSE->cache = NULL;
    temp->DENSE->A = NULL;
    temp->DENSE->dropout_mask = NULL;
    temp->DENSE->dZ = NULL;
    m->output = NULL;
    
    layer++;
    temp = temp->next_layer;
  }
}