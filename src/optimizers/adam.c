#include "./adam.h"
#include "../model/model.h"

extern __Model__ * m;

void adam(){
  Computation_Graph * temp = m->graph->next_layer;
  int layer = 0;
  while(temp->next_layer->type!=LOSS){
    //calculate first momentum
    //m_dW
    double mul_factor = 1-m->beta1;
    dARRAY * term1 = mulScalar(temp->DENSE->dW,mul_factor);
    mul_factor = m->beta1;
    dARRAY * term2 = mulScalar(m->m_t_dW[layer],mul_factor);
    free2d(m->m_t_dW[layer]);
    m->m_t_dW[layer] = NULL;
    m->m_t_dW[layer] = add(term1,term2);

    free2d(term1);
    free2d(term2);
    term1 = term2 = NULL;

    //m_db
    term1 = mulScalar(temp->DENSE->db,mul_factor);
    mul_factor = m->beta1;
    term2 = mulScalar(m->m_t_db[layer],mul_factor);
    free2d(m->m_t_db[layer]);
    m->m_t_db[layer] = NULL;
    m->m_t_db[layer] = add(term1,term2);

    free2d(term1);
    free2d(term2);
    term1 = term2 = NULL;

    //calculate second momentum
    mul_factor = 1-m->beta2;
    dARRAY * scaled_grads_w = power(temp->DENSE->dW,2);
    term1 = mulScalar(scaled_grads_w,mul_factor);

    free2d(scaled_grads_w);
    scaled_grads_w = NULL;

    mul_factor = m->beta2;
    term2 = mulScalar(m->v_t_dW[layer],mul_factor);
    free2d(m->v_t_dW[layer]);
    m->v_t_dW[layer] = NULL;
    m->v_t_dW[layer] = add(term1,term2);

    free2d(term1);
    free2d(term2);
    term1 = term2 = NULL;

    //calculate v_t_db
    mul_factor = 1-m->beta2;
    dARRAY * scaled_grads_b = power(temp->DENSE->db,2);
    term1 = mulScalar(scaled_grads_b,mul_factor);

    free2d(scaled_grads_b);
    scaled_grads_b = NULL;

    mul_factor = m->beta2;
    term2 = mulScalar(m->v_t_db[layer],mul_factor);
    free2d(m->v_t_db[layer]);
    m->v_t_db[layer] = NULL;
    m->v_t_db[layer] = add(term1,term2);

    free2d(term1);
    free2d(term2);
    term1 = term2 = NULL;
    
    free2d(temp->DENSE->dW);
    free2d(temp->DENSE->db);
    temp->DENSE->dW = temp->DENSE->db = NULL;

    double first_momentum_scaling_factor = 1-pow(m->beta1,m->time_step);
    double second_momentum_scaling_factor = 1-pow(m->beta2,m->time_step);
    
    dARRAY * m_t_dW_corrected = divScalar(m->m_t_dW[layer],first_momentum_scaling_factor);
    dARRAY * m_t_db_corrected = divScalar(m->m_t_db[layer],first_momentum_scaling_factor);

    dARRAY * v_t_dW_corrected = divScalar(m->v_t_dW[layer],second_momentum_scaling_factor);
    dARRAY * v_t_db_corrected = divScalar(m->v_t_db[layer],second_momentum_scaling_factor);

    dARRAY * layer_weights, * layer_biases;
    layer_weights = NULL;
    layer_biases = NULL;

    layer_weights = temp->DENSE->weights;
    layer_biases = temp->DENSE->bias;

    dARRAY * sqrt_v_t_dW = power(v_t_dW_corrected,0.5);
    dARRAY * sqrt_v_t_db = power(v_t_db_corrected,0.5);

    free2d(v_t_dW_corrected);
    free2d(v_t_db_corrected);
    v_t_dW_corrected = v_t_db_corrected = NULL;

    dARRAY * decay_factor_dW = addScalar(sqrt_v_t_dW,m->epsilon);
    dARRAY * decay_factor_db = addScalar(sqrt_v_t_db,m->epsilon);

    free2d(sqrt_v_t_dW);
    free2d(sqrt_v_t_db);
    sqrt_v_t_dW = sqrt_v_t_db = NULL;

    dARRAY * mul_lr_w = mulScalar(m_t_dW_corrected,m->learning_rate);
    dARRAY * mul_lr_b = mulScalar(m_t_db_corrected,m->learning_rate);

    free2d(m_t_dW_corrected);
    free2d(m_t_db_corrected);
    m_t_dW_corrected = m_t_db_corrected = NULL;

    dARRAY * update_term2_w = divison(mul_lr_w,decay_factor_dW);
    dARRAY * update_term2_b = divison(mul_lr_b,decay_factor_db);

    free2d(mul_lr_w);
    free2d(mul_lr_b);
    free2d(decay_factor_dW);
    free2d(decay_factor_db);
    mul_lr_w = mul_lr_b = decay_factor_dW = decay_factor_db = NULL;

    temp->DENSE->weights = subtract(layer_weights,update_term2_w);
    temp->DENSE->bias = subtract(layer_biases,update_term2_b);

    free2d(layer_weights);
    free2d(layer_biases);
    free2d(update_term2_w);
    free2d(update_term2_b);
    layer_weights = layer_biases = update_term2_w = update_term2_b = NULL;

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
