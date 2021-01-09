#include "./loss_functions.h"

double cross_entropy_loss(Dense_layer * layer, dARRAY * Y){
  int m = Y->shape[1];
  dARRAY * temp_1 = NULL;
  temp_1 = (dARRAY *)malloc(sizeof(dARRAY));
  temp_1->matrix = (double*)malloc(sizeof(double)*layer->A->shape[0]*layer->A->shape[1]);
  #pragma omp parallel for
  for(int i=0;i<layer->A->shape[0]*layer->A->shape[1];i++){
    temp_1->matrix[i] = log(layer->A->matrix[i]);
  }
  temp_1->shape[0] = layer->A->shape[0];
  temp_1->shape[1] = layer->A->shape[1];

  int dims[] = {layer->A->shape[0],layer->A->shape[1]};
  dARRAY * temp_ones = ones(dims);
  dARRAY * temp_sub = subtract(temp_ones,layer->A);
  
  dARRAY * temp_2 = NULL;
  temp_2 = (dARRAY *)malloc(sizeof(dARRAY));
  temp_2->matrix = (double*)malloc(sizeof(double)*layer->A->shape[0]*layer->A->shape[1]);

  #pragma omp parallel for
  for(int i=0;i<layer->A->shape[0]*layer->A->shape[1];i++){
    temp_2->matrix[i] = log(temp_sub->matrix[i]);
  }
  temp_2->shape[0] = layer->A->shape[0];
  temp_2->shape[1] = layer->A->shape[1];

  free2d(temp_sub);
  temp_sub = NULL;

  dARRAY * loss_term_1 = multiply(Y,temp_1);
  free2d(temp_1);
  temp_1 = NULL;

  temp_sub = subtract(temp_ones,Y);
  free2d(temp_ones);
  temp_ones = NULL;

  dARRAY * loss_term_2 = multiply(temp_sub,temp_2);
  free2d(temp_sub);
  temp_sub = NULL;

  dARRAY * temp_loss = add(loss_term_1,loss_term_2);
  free2d(loss_term_1);
  free2d(loss_term_2);
  loss_term_1 = loss_term_2 = NULL;

  dARRAY * temp_loss_res = sum(temp_loss,1);
  dARRAY * loss = mulScalar(temp_loss_res,-1);
  free2d(temp_loss_res);
  temp_loss_res = NULL;

  dARRAY * cost = divScalar(loss,m);
  free2d(loss);
  loss = NULL;

  double total_cost = cost->matrix[0];
  free2d(cost);
  cost = NULL;

  return total_cost;
}