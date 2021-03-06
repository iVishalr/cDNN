#include <cdnn/activations.h>
#include <cdnn/model.h>

extern __Model__ * m;

dARRAY * forward_pass_tanh(){
  dARRAY * tanh_out = (dARRAY*)malloc(sizeof(dARRAY));
  tanh_out->matrix = (float*)calloc(m->current_layer->DENSE->cache->shape[0]*m->current_layer->DENSE->cache->shape[1],sizeof(float));
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(m,tanh_out) schedule(static)
  for(int i=0;i<m->current_layer->DENSE->cache->shape[0]*m->current_layer->DENSE->cache->shape[1];i++){
    //Computing the tanh function
    float exp_res1 = exp(m->current_layer->DENSE->cache->matrix[i]);
    float exp_res2 = exp(-1*m->current_layer->DENSE->cache->matrix[i]);
    tanh_out->matrix[i] = (exp_res1 - exp_res2)/(exp_res1 + exp_res2);
  }
  tanh_out->shape[0] = m->current_layer->DENSE->cache->shape[0];
  tanh_out->shape[1] = m->current_layer->DENSE->cache->shape[1];
  return tanh_out;
}

dARRAY * backward_pass_tanh(){
  dARRAY * tanh_out = (dARRAY*)malloc(sizeof(dARRAY));
  tanh_out->matrix = (float*)calloc(m->current_layer->DENSE->A->shape[0]*m->current_layer->DENSE->A->shape[1],sizeof(float));
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(m,tanh_out) schedule(static)
  for(int i=0;i<m->current_layer->DENSE->A->shape[0]*m->current_layer->DENSE->A->shape[1];i++){
    //Computing the tanh function
    float exp_res1 = exp(m->current_layer->DENSE->A->matrix[i]);
    float exp_res2 = exp(-1*m->current_layer->DENSE->A->matrix[i]);
    float temp = (exp_res1 - exp_res2)/(exp_res1 + exp_res2);
    //gradient of tanh is g'(z) = 1 - (tanh(z))^2
    tanh_out->matrix[i] = 1-pow(temp,2.0f);
  }
  tanh_out->shape[0] = m->current_layer->DENSE->A->shape[0];
  tanh_out->shape[1] = m->current_layer->DENSE->A->shape[1];
  return tanh_out;
}

Tanh * Tanh__init__(dARRAY * linear_matrix){
  Tanh * tanh = (Tanh*)malloc(sizeof(Tanh));
  tanh->forward = forward_pass_tanh;
  tanh->backward = backward_pass_tanh;
  tanh->in_dims[0] = tanh->out_dims[0] = linear_matrix->shape[0];
  tanh->in_dims[1] = tanh->out_dims[1] = linear_matrix->shape[1];
  return tanh;
}

dARRAY * (TanH)(Tanh_args args){
  Tanh * t = Tanh__init__(args.input);
  if(!args.status)
    return t->forward();
  else 
    return t->backward();
  free(t);
  t=NULL;
}