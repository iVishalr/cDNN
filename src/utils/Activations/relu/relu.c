#include "../activations.h"
#include "../../../model/model.h"

extern __Model__ * m;

dARRAY * forward_pass_relu(){
  dARRAY * relu_outf = (dARRAY*)malloc(sizeof(dARRAY));
  relu_outf->matrix = (double*)calloc(m->current_layer->DENSE->cache->shape[0]*m->current_layer->DENSE->cache->shape[1],sizeof(double));
  omp_set_num_threads(4);
  #pragma omp parallel for
  for(int i=0;i<m->current_layer->DENSE->cache->shape[0]*m->current_layer->DENSE->cache->shape[1];i++)
    relu_outf->matrix[i] = m->current_layer->DENSE->cache->matrix[i]>(double)0 ?(double)m->current_layer->DENSE->cache->matrix[i] : (double)0.0;
  relu_outf->shape[0] = m->current_layer->DENSE->cache->shape[0];
  relu_outf->shape[1] = m->current_layer->DENSE->cache->shape[1];
  return relu_outf;
}

dARRAY * backward_pass_relu(){
  dARRAY * relu_outb = (dARRAY*)malloc(sizeof(dARRAY));
  relu_outb->matrix = (double*)calloc(m->current_layer->DENSE->A->shape[0]*m->current_layer->DENSE->A->shape[1],sizeof(double));
  omp_set_num_threads(4);
  #pragma omp parallel for
  for(int i=0;i<m->current_layer->DENSE->A->shape[0]*m->current_layer->DENSE->A->shape[1];i++)
    relu_outb->matrix[i] = m->current_layer->DENSE->A->matrix[i]>(double)0 ? (double)1.0 : (double)0.0;
  relu_outb->shape[0] = m->current_layer->DENSE->A->shape[0];
  relu_outb->shape[1] = m->current_layer->DENSE->A->shape[1];
  return relu_outb;
}

ReLu * ReLu__init__(dARRAY * linear_matrix){
  ReLu * relu = (ReLu*)malloc(sizeof(ReLu));
  relu->forward = forward_pass_relu;
  relu->backward = backward_pass_relu;
  relu->in_dims[0] = relu->out_dims[0] = linear_matrix->shape[0];
  relu->in_dims[1] = relu->out_dims[1] = linear_matrix->shape[1];
  return relu;
}

dARRAY * (relu)(ReLu_args args){
  ReLu * r = ReLu__init__(args.input);
  if(!args.status)
    return r->forward();
  else
    return r->backward();
  free(r);
  r = NULL;
}

