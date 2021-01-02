#include "./sigmoid.h"

dARRAY * forward_pass(dARRAY * linear_matrix){
  dARRAY * sigmoid_out = (dARRAY*)malloc(sizeof(dARRAY));
  sigmoid_out->matrix = (double*)malloc(sizeof(double)*linear_matrix->shape[0]*linear_matrix->shape[1]);
  omp_set_num_threads(4);
  #pragma omp parallel for shared(linear_matrix,sigmoid_out)
  for(int i=0;i<linear_matrix->shape[0]*linear_matrix->shape[1];i++)
    sigmoid_out->matrix[i] = 1/(1+exp(-1*linear_matrix->matrix[i]));
  sigmoid_out->shape[0] = linear_matrix->shape[0];
  sigmoid_out->shape[1] = linear_matrix->shape[1];
  return sigmoid_out;
}

dARRAY * backward_pass(dARRAY * linear_matrix){
  dARRAY * sigmoid_out;
  omp_set_num_threads(4);
  int dims[] = {linear_matrix->shape[0],linear_matrix->shape[1]};
  sigmoid_out = multiply(linear_matrix,subtract(ones(dims),linear_matrix));
  sigmoid_out->shape[0] = linear_matrix->shape[0];
  sigmoid_out->shape[1] = linear_matrix->shape[1];
  return sigmoid_out;
}

Sigmoid * Sigmoid__init__(dARRAY * layer_matrix, int layer){
  Sigmoid * sigmoid = (Sigmoid*)malloc(sizeof(Sigmoid));
  sigmoid->forward_prop = forward_pass;
  sigmoid->back_prop = backward_pass;
  sigmoid->in_dims[0] = sigmoid->out_dims[0] = layer_matrix->shape[0];
  sigmoid->in_dims[1] = sigmoid->out_dims[1] = layer_matrix->shape[1];
  sigmoid->layer_num = layer;
  return sigmoid;
}

int main(){
  int dims[] = {2,4};
  dARRAY * linear_matrix = randn(dims);
  Sigmoid * relu = Sigmoid__init__(linear_matrix,1);
  dARRAY * relu_op1 = NULL;
  dARRAY * relu_op2 = NULL;
  relu_op1 = relu->forward_prop(linear_matrix);
  
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
  relu_op2 = relu->back_prop(linear_matrix);
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