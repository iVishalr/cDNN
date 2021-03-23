#include "./adam.h"
#include "../model/model.h"

extern __Model__ * m;

void adam(){
  int layer = 0;
  Computation_Graph * temp = m->graph;
  
  while(temp!=NULL){
    if(temp->type==LOSS || temp->type==INPUT){
      temp = temp->next_layer;
      continue;
    }
    //calculate first momentum
    float mul_factor = 1.0f-m->beta1;
    dARRAY * term1_W = mulScalar(temp->DENSE->dW,mul_factor);
    dARRAY * term1_b = mulScalar(temp->DENSE->db,mul_factor);

    mul_factor = m->beta1;
    dARRAY * ptr_m_t_dW = m->m_t_dW[layer];
    dARRAY * ptr_m_t_db = m->m_t_db[layer];

    dARRAY * term2_W = mulScalar(ptr_m_t_dW,mul_factor);
    dARRAY * term2_b = mulScalar(ptr_m_t_db,mul_factor);
    
    free2d(ptr_m_t_dW);
    free2d(ptr_m_t_db);
    ptr_m_t_dW = NULL;
    ptr_m_t_db = NULL;
    
    m->m_t_dW[layer] = add(term1_W,term2_W);
    m->m_t_db[layer] = add(term1_b,term2_b);

    free2d(term1_W);
    free2d(term2_W);
    free2d(term1_b);
    free2d(term2_b);
    term1_W = NULL;
    term2_W = NULL;    
    term1_b = NULL;
    term2_b = NULL;

    //calculate second momentum
    mul_factor = 1-m->beta2;
    dARRAY * scaled_grads_w = power(temp->DENSE->dW,2.0f);
    dARRAY * scaled_grads_b = power(temp->DENSE->db,2.0f);
    term1_W = mulScalar(scaled_grads_w,mul_factor);
    term1_b = mulScalar(scaled_grads_b,mul_factor);

    free2d(scaled_grads_w);
    free2d(scaled_grads_b);
    scaled_grads_w = NULL;
    scaled_grads_b = NULL;

    mul_factor = m->beta2;
    dARRAY * ptr_v_t_dW = m->v_t_dW[layer];
    dARRAY * ptr_v_t_db = m->v_t_db[layer];
    term2_W = mulScalar(ptr_v_t_dW,mul_factor);
    term2_b = mulScalar(ptr_v_t_db,mul_factor);
    
    free2d(ptr_v_t_dW);
    free2d(ptr_v_t_db);
    ptr_v_t_dW = NULL;
    ptr_v_t_db = NULL;

    m->v_t_dW[layer] = add(term1_W,term2_W);
    m->v_t_db[layer] = add(term1_b,term2_b);

    free2d(term1_W);
    free2d(term2_W);
    free2d(term1_b);
    free2d(term2_b);
    term1_W = NULL;
    term2_W = NULL;
    term1_b = NULL;
    term2_b = NULL;

    free2d(temp->DENSE->dW);
    free2d(temp->DENSE->db);
    temp->DENSE->dW = NULL;
    temp->DENSE->db = NULL;

    float first_momentum_scaling_factor = 1-powf(m->beta1,m->time_step);
    float second_momentum_scaling_factor = 1-powf(m->beta2,m->time_step);
    
    ptr_m_t_dW = m->m_t_dW[layer];
    ptr_m_t_db = m->m_t_db[layer];
    
    dARRAY * m_t_dW_corrected = divScalar(ptr_m_t_dW,first_momentum_scaling_factor);
    dARRAY * m_t_db_corrected = divScalar(ptr_m_t_db,first_momentum_scaling_factor);

    ptr_v_t_dW = m->v_t_dW[layer];
    ptr_v_t_db = m->v_t_db[layer];
    
    dARRAY * v_t_dW_corrected = divScalar(ptr_v_t_dW,second_momentum_scaling_factor);
    dARRAY * v_t_db_corrected = divScalar(ptr_v_t_db,second_momentum_scaling_factor);
    
    ptr_m_t_dW = NULL;
    ptr_m_t_db = NULL;
    ptr_v_t_dW = NULL;
    ptr_v_t_db = NULL;

    dARRAY * layer_weights, * layer_biases;
    layer_weights = NULL;
    layer_biases = NULL;

    layer_weights = temp->DENSE->weights;
    layer_biases = temp->DENSE->bias;

    dARRAY * sqrt_v_t_dW = squareroot(v_t_dW_corrected);
    dARRAY * sqrt_v_t_db = squareroot(v_t_db_corrected);

    free2d(v_t_dW_corrected);
    free2d(v_t_db_corrected);
    v_t_dW_corrected = NULL;
    v_t_db_corrected = NULL;

    dARRAY * decay_factor_dW = addScalar(sqrt_v_t_dW,m->epsilon);
    dARRAY * decay_factor_db = addScalar(sqrt_v_t_db,m->epsilon);

    free2d(sqrt_v_t_dW);
    free2d(sqrt_v_t_db);
    sqrt_v_t_dW = NULL;
    sqrt_v_t_db = NULL;

    dARRAY * mul_lr_w = mulScalar(m_t_dW_corrected,m->learning_rate);
    dARRAY * mul_lr_b = mulScalar(m_t_db_corrected,m->learning_rate);

    free2d(m_t_dW_corrected);
    free2d(m_t_db_corrected);
    m_t_dW_corrected = NULL;
    m_t_db_corrected = NULL;

    dARRAY * update_term2_w = divison(mul_lr_w,decay_factor_dW);
    dARRAY * update_term2_b = divison(mul_lr_b,decay_factor_db);

    free2d(mul_lr_w);
    free2d(mul_lr_b);
    free2d(decay_factor_dW);
    free2d(decay_factor_db);
    mul_lr_w = NULL;
    mul_lr_b = NULL;
    decay_factor_dW = NULL;
    decay_factor_db = NULL;

    if(m->regularization!=NULL){
      float mul_grad = m->lambda * m->learning_rate/(float)m->y_train_mini_batch[m->current_mini_batch]->shape[1];
      dARRAY * update_term2_dW = mulScalar(temp->DENSE->weights,mul_grad);
      dARRAY * update_term2_db = mulScalar(temp->DENSE->bias,mul_grad);

      dARRAY * ptr_update_term1_dW = update_term2_w;
      dARRAY * ptr_update_term1_db = update_term2_b;

      update_term2_w = add(ptr_update_term1_dW,update_term2_dW);
      update_term2_b = add(ptr_update_term1_db,update_term2_db);

      free2d(update_term2_dW);
      free2d(update_term2_db);
      free2d(ptr_update_term1_dW);
      free2d(ptr_update_term1_db);
      update_term2_dW = NULL;
      update_term2_db = NULL;
      ptr_update_term1_dW = NULL;
      ptr_update_term1_db = NULL;
    }

    temp->DENSE->weights = subtract(layer_weights,update_term2_w);
    temp->DENSE->bias = subtract(layer_biases,update_term2_b);

    free2d(layer_weights);
    free2d(layer_biases);
    free2d(update_term2_w);
    free2d(update_term2_b);
    layer_weights = NULL;
    layer_biases = NULL;
    update_term2_w = NULL;
    update_term2_b = NULL;

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
