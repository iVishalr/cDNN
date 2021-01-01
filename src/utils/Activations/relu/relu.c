#include "./relu.h"

dARRAY * forward_pass(dARRAY * layer_matrix){
  dARRAY * relu_out = (dARRAY*)malloc(sizeof(dARRAY));
  relu_out->matrix = (double*)malloc(sizeof(double)*layer_matrix->shape[0]*layer_matrix->shape[1]);
  for(int i=0;i<layer_matrix->shape[0]*layer_matrix->shape[1];i++)
    relu_out->matrix[i] = layer_matrix->matrix[i]>(double)0 ? layer_matrix->matrix[i] : 0;
  relu_out->shape[0] = layer_matrix->shape[0];
  relu_out->shape[1] = layer_matrix->shape[1];
  return relu_out;
}

dARRAY * backward_pass(dARRAY * layer_matrix){
  dARRAY * relu_out = (dARRAY*)malloc(sizeof(dARRAY));
  relu_out->matrix = (double*)malloc(sizeof(double)*layer_matrix->shape[0]*layer_matrix->shape[1]);
  for(int i=0;i<layer_matrix->shape[0]*layer_matrix->shape[1];i++)
    relu_out->matrix[i] = layer_matrix->matrix[i]>(double)0 ? 1 : 0;
  relu_out->shape[0] = layer_matrix->shape[0];
  relu_out->shape[1] = layer_matrix->shape[1];
  return relu_out;
}

ReLu * ReLu__init__(dARRAY * linear_matrix,int layer){
  ReLu * relu = (ReLu*)malloc(sizeof(ReLu));
  relu->forward_prop = forward_pass;
  relu->back_prop = backward_pass;
  relu->in_dims[0] = relu->out_dims[0] = linear_matrix->shape[0];
  relu->in_dims[1] = relu->out_dims[1] = linear_matrix->shape[1];
  relu->layer_num = layer;
  return relu;
}

int main(){
  int dims[] = {5,10};
  dARRAY * linear_matrix = randn(dims);
  ReLu * relu = ReLu__init__(linear_matrix,1);
  dARRAY * relu_op1 = NULL;
  dARRAY * relu_op2 = NULL;
  relu_op1 = relu->forward_prop(linear_matrix);
  relu_op2 = relu->back_prop(relu_op1);
  printf("Input : \n");
  for(int i=0;i<linear_matrix->shape[0];i++){
    for(int j=0;j<linear_matrix->shape[1];j++){
      printf("%lf ",linear_matrix->matrix[i*linear_matrix->shape[1]+j]);
    }
    printf("\n");
  }

  printf("Forward Prop : \n");
  for(int i=0;i<relu_op1->shape[0];i++){
    for(int j=0;j<relu_op1->shape[1];j++){
      printf("%lf ",relu_op1->matrix[i*relu_op1->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("Back Prop : \n");
  for(int i=0;i<relu_op2->shape[0];i++){
    for(int j=0;j<relu_op2->shape[1];j++){
      printf("%lf ",relu_op2->matrix[i*relu_op2->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  free2d(relu_op1);
  free2d(relu_op2);
  free2d(linear_matrix);
  free(relu);
  return 0;
}