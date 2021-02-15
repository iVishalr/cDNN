#include "./activations.h"
#include "../model/model.h"

extern __Model__ * m;

dARRAY * forward_pass_sigmoid(){
  dARRAY * sigmoid_outf = NULL;
  sigmoid_outf = (dARRAY*)malloc(sizeof(dARRAY));
  sigmoid_outf->matrix = (float*)calloc(m->current_layer->DENSE->cache->shape[0]*m->current_layer->DENSE->cache->shape[1],sizeof(float));
  omp_set_num_threads(8);
  #pragma omp parallel for num_threads(8) shared(m,sigmoid_outf) schedule(static)
  for(int i=0;i<m->current_layer->DENSE->cache->shape[0]*m->current_layer->DENSE->cache->shape[1];i++)
    sigmoid_outf->matrix[i] = (float)(1.0/(float)(1+exp((float)(-1.0*m->current_layer->DENSE->cache->matrix[i]))));
  sigmoid_outf->shape[0] = m->current_layer->DENSE->cache->shape[0];
  sigmoid_outf->shape[1] = m->current_layer->DENSE->cache->shape[1];
  return sigmoid_outf;
}

dARRAY * backward_pass_sigmoid(){
  dARRAY * sigmoid_outb = NULL;
  dARRAY * temp = NULL;
  dARRAY * one = NULL;
  int dims[] = {m->current_layer->DENSE->A->shape[0],m->current_layer->DENSE->A->shape[1]};
  one = ones(dims);
  temp = subtract(one,m->current_layer->DENSE->A);

  free2d(one);
  one=NULL;

  sigmoid_outb = multiply(m->current_layer->DENSE->A,temp);
  free2d(temp);
  temp = NULL;
  return sigmoid_outb;
}

Sigmoid * Sigmoid__init__(dARRAY * layer_matrix){
  Sigmoid * sigmoid = (Sigmoid*)malloc(sizeof(Sigmoid));
  sigmoid->forward = forward_pass_sigmoid;
  sigmoid->backward = backward_pass_sigmoid;
  sigmoid->in_dims[0] = sigmoid->out_dims[0] = layer_matrix->shape[0];
  sigmoid->in_dims[1] = sigmoid->out_dims[1] = layer_matrix->shape[1];
  return sigmoid;
}


dARRAY * (sigmoid)(Sigmoid_args args){
  Sigmoid * s = Sigmoid__init__(args.input);
  if(!args.status)
    return s->forward(args.input);
  else
    return s->backward(args.input);
  free(s);
  s=NULL;
}