#include "./test_activations.h"

void test_ReLu(){
  printf("Testing ReLu ...\n");

  int dims[] = {5,10};
  dARRAY * linear_matrix = randn(dims);
  ReLu * relu = ReLu__init__(linear_matrix,1);

  dARRAY * relu_op1 = NULL;
  dARRAY * relu_op2 = NULL;

  relu_op1 = relu->forward_prop(linear_matrix);
  relu_op2 = relu->back_prop(linear_matrix);

  printf("Input : \n");
  display(linear_matrix);
  printf("\nForward Prop : \n");
  display(relu_op1);
  printf("\nBack Prop : \n");
  display(relu_op2);
  free2d(relu_op1);
  free2d(relu_op2);
  free2d(linear_matrix);
  free(relu);
}

void test_Sigmoid(){
  printf("Testing Sigmoid ...\n");

  int dims[] = {5,10};
  dARRAY * linear_matrix = randn(dims);
  Sigmoid * sigmoid = Sigmoid__init__(linear_matrix,1);

  dARRAY * sigmoid_op1 = NULL;
  dARRAY * sigmoid_op2 = NULL;

  sigmoid_op1 = sigmoid->forward_prop(linear_matrix);
  sigmoid_op2 = sigmoid->back_prop(linear_matrix);

  printf("Input : \n");
  display(linear_matrix);
  printf("\nForward Prop : \n");
  display(sigmoid_op1);
  printf("\nBack Prop : \n");
  display(sigmoid_op2);
  free2d(sigmoid_op1);
  free2d(sigmoid_op2);
  free2d(linear_matrix);
  free(sigmoid);
}

void test_Tanh(){
  printf("Testing Tanh ...\n");

  int dims[] = {5,10};
  dARRAY * linear_matrix = randn(dims);
  Tanh * tanh = Tanh__init__(linear_matrix,1);

  dARRAY * tanh_op1 = NULL;
  dARRAY * tanh_op2 = NULL;

  tanh_op1 = tanh->forward_prop(linear_matrix);
  tanh_op2 = tanh->back_prop(linear_matrix);

  printf("Input : \n");
  display(linear_matrix);
  printf("\nForward Prop : \n");
  display(tanh_op1);
  printf("\nBack Prop : \n");
  display(tanh_op2);
  free2d(tanh_op1);
  free2d(tanh_op2);
  free2d(linear_matrix);
  free(tanh);
}

void display(dARRAY * linear_matrix){
  for(int i=0;i<linear_matrix->shape[0];i++){
    for(int j=0;j<linear_matrix->shape[1];j++){
      printf("%lf ",linear_matrix->matrix[i*linear_matrix->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
}

int main(){
  test_ReLu();
  test_Sigmoid();
  test_Tanh();
  return 0;
}