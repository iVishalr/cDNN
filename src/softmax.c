#include <activations.h>
#include <model.h>

extern __Model__ * m;

dARRAY * forward_pass_softmax(){
  dARRAY * softmax_outf = NULL;
  dARRAY * exp_sub_max = exponentional(m->current_layer->DENSE->cache);

  dARRAY * div_factor = sum(exp_sub_max,0);
  softmax_outf = divison(exp_sub_max,div_factor);

  free2d(exp_sub_max);
  free2d(div_factor);
  exp_sub_max = div_factor = NULL; 
  
  return softmax_outf;
}

dARRAY * backward_pass_softmax(){
  //not used as it is wrong. backward pass of sigmoid is difficult to implement
  dARRAY * softmax_outb = NULL;
  dARRAY * temp = NULL;
  dARRAY * one = NULL;
  int dims[] = {m->current_layer->DENSE->A->shape[0],m->current_layer->DENSE->A->shape[1]};
  one = ones(dims);
  temp = subtract(one,m->current_layer->DENSE->A);

  free2d(one);
  one=NULL;

  softmax_outb = multiply(m->current_layer->DENSE->A,temp);
  free2d(temp);
  temp = NULL;
  return softmax_outb;
}

Softmax * Softmax__init__(dARRAY * layer_matrix){
  Softmax * sfmax = (Softmax*)malloc(sizeof(Softmax));
  sfmax->forward = forward_pass_softmax;
  sfmax->backward = backward_pass_softmax;
  sfmax->in_dims[0] = sfmax->out_dims[0] = layer_matrix->shape[0];
  sfmax->in_dims[1] = sfmax->out_dims[1] = layer_matrix->shape[1];
  return sfmax;
}

dARRAY * (softmax)(softmax_args args){
  Softmax * sfmax = Softmax__init__(args.input);
  if(!args.status)
    return sfmax->forward();
  else
    return sfmax->backward();
  free(sfmax);
  sfmax = NULL;
}
