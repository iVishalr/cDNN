#include "./momentum.h"
#include "../model/model.h"

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
    temp->DENSE->dW = ptr_dW = NULL;

    m->m_t_dW[layer] = add(term1,term2);

    free2d(term1);
    free2d(term2);
    term1 = term2 = NULL;

    //m_db
    mul_factor = 1-m->beta;
    term1 = mulScalar(temp->DENSE->db,mul_factor);
    
    mul_factor = m->beta;
    dARRAY * ptr_db = m->m_t_db[layer];
    
    term2 = mulScalar(ptr_db,mul_factor);

    free2d(ptr_db);
    free2d(temp->DENSE->db);
    temp->DENSE->db = ptr_db = NULL;

    m->m_t_db[layer] = add(term1,term2);

    free2d(term1);
    free2d(term2);
    term1 = term2 = NULL;

    dARRAY * layer_weights, * layer_biases;
    layer_weights = NULL;
    layer_biases = NULL;

    layer_weights = temp->DENSE->weights;
    layer_biases = temp->DENSE->bias;

    dARRAY * mul_lr_w = mulScalar(m->m_t_dW[layer],m->learning_rate);
    dARRAY * mul_lr_b = mulScalar(m->m_t_db[layer],m->learning_rate);

    temp->DENSE->weights = subtract(layer_weights,mul_lr_w);
    temp->DENSE->bias = subtract(layer_biases,mul_lr_b);

    free2d(layer_weights);
    free2d(layer_biases);
    free2d(mul_lr_w);
    free2d(mul_lr_b);
    layer_weights = layer_biases = mul_lr_w = mul_lr_b = NULL;

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
    temp->DENSE->dA=temp->DENSE->cache = temp->DENSE->A = temp->DENSE->dropout_mask = temp->DENSE->dZ = m->output = NULL;

    layer++;
    temp = temp->next_layer;
  }
}
