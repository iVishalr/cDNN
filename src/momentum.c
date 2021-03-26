#include <model.h>
#include <momentum.h>

extern __Model__ * m;

void Momentum(){
  Computation_Graph * temp = m->graph;
  int layer = 0;
  while(temp!=NULL){
    if(temp->type==LOSS || temp->type==INPUT){
      temp = temp->next_layer;
      continue;
    }
    //calculate first momentum
    //m_dW
    float mul_factor = 1-m->beta;
    
    dARRAY * term1 = mulScalar(temp->DENSE->dW,mul_factor);
    
    mul_factor = m->beta;
    
    dARRAY * ptr_dW = m->m_t_dW[layer];
    dARRAY * term2 = mulScalar(ptr_dW,mul_factor);

    free2d(ptr_dW);
    free2d(temp->DENSE->dW);
    temp->DENSE->dW = NULL;
    ptr_dW = NULL;

    m->m_t_dW[layer] = add(term1,term2);
    
    free2d(term1);
    free2d(term2);
    term1 = NULL;
    term2 = NULL;
    
    //m_db
    mul_factor = 1-m->beta;
    term1 = mulScalar(temp->DENSE->db,mul_factor);
    
    mul_factor = m->beta;
    dARRAY * ptr_db = m->m_t_db[layer];
    
    term2 = mulScalar(ptr_db,mul_factor);

    free2d(ptr_db);
    free2d(temp->DENSE->db);
    temp->DENSE->db = NULL;
    ptr_db = NULL;

    m->m_t_db[layer] = add(term1,term2);

    free2d(term1);
    free2d(term2);
    term1 = NULL;
    term2 = NULL;

    dARRAY * layer_weights, * layer_biases;
    layer_weights = NULL;
    layer_biases = NULL;

    layer_weights = temp->DENSE->weights;
    layer_biases = temp->DENSE->bias;

    ptr_dW = m->m_t_dW[layer];
    ptr_db = m->m_t_db[layer];

    dARRAY * mul_lr_w = mulScalar(ptr_dW,m->learning_rate);
    dARRAY * mul_lr_b = mulScalar(ptr_db,m->learning_rate);

    if(m->regularization!=NULL){
      dARRAY * ptr_update_term_dW = mul_lr_w;
      dARRAY * ptr_update_term_db = mul_lr_b;

      float mul_decay = m->lambda * m->learning_rate / (float)m->y_train_mini_batch[m->current_mini_batch]->shape[1];
      dARRAY * update_term2_dW = mulScalar(temp->DENSE->weights,mul_decay);
      dARRAY * update_term2_db = mulScalar(temp->DENSE->bias,mul_decay);

      mul_lr_w = add(ptr_update_term_dW,update_term2_dW);
      mul_lr_b = add(ptr_update_term_db,update_term2_db);

      free2d(update_term2_dW);
      free2d(update_term2_db);
      free2d(ptr_update_term_dW);
      free2d(ptr_update_term_db);
      update_term2_dW = NULL;
      update_term2_db = NULL;
      ptr_update_term_dW = NULL;
      ptr_update_term_db = NULL;
    }

    temp->DENSE->weights = subtract(layer_weights,mul_lr_w);
    temp->DENSE->bias = subtract(layer_biases,mul_lr_b);

    free2d(layer_weights);
    free2d(layer_biases);
    free2d(mul_lr_w);
    free2d(mul_lr_b);
    layer_weights = NULL;
    layer_biases = NULL;
    mul_lr_w = NULL;
    mul_lr_b = NULL;

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

    layer++;
    temp = temp->next_layer;
  }
}
