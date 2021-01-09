#include "../neural_net/neural_net.h"

extern Computation_Graph * G;
extern char * loss_type;
extern dARRAY * Y;

int main(){
  int feature_dims[] = {5,4};
  int flag=1;
  int iterations = 30;
  dARRAY * X = randn(feature_dims);
  Input(.layer_size=5,.input_features=X,.layer_num=1);
  Dense(.layer_size=5,.activation="relu",.initializer="he",.dropout=0.5,.layer_type="hidden",.layer_num=2);
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output",.layer_num=3);
  loss_type = "cross_entropy_loss";
  
  G = G->next_layer;
  Computation_Graph * curr_layer = G;
  Computation_Graph * temp = G;

  while(G->next_layer!=NULL){
    printf("Initializing params\n");
    printf("Layer Number : %d\n",G->DENSE->layer_num);
    G->DENSE->initalize_params();
    shape(G->DENSE->weights);
    shape(G->DENSE->bias);
    G = G->next_layer;
  }
  printf("Initializing params\n");
    printf("Layer Number : %d\n",G->DENSE->layer_num);
  G->DENSE->initalize_params();
  shape(G->DENSE->weights);
  shape(G->DENSE->bias);
  G = G->prev_layer;
  // G = G->prev_layer;
  int i=1;
  while(i<=iterations){
    while(G->next_layer!=NULL){
      // printf("Layer Number : %d\n",G->DENSE->layer_num);
      // printf("FPROP\n");
      G->DENSE->forward_prop();
      // sleep(5000);
      G = G->next_layer;
    }
    // printf("Layer Number : %d\n",G->DENSE->layer_num);
    // printf("FPROP\n\n");
    G->DENSE->forward_prop();
    // sleep(5000);
    if(flag){
      int dims[] = {G->DENSE->A->shape[0],G->DENSE->A->shape[1]};
      Y = ones(dims);
      flag = 0;
    }

    printf("\033[96mIteration (%d) - Cost : \033[0m%lf\n",i,cross_entropy_loss(G->DENSE,Y));
    sleep(500);

    curr_layer->computation_graph_status = 1;
    while(G->prev_layer!=NULL){
      // printf("Layer Num : %d\n",G->DENSE->layer_num);
      G->DENSE->back_prop();
      // sleep(3000);
      // if(size(G->DENSE->dW)==size(G->DENSE->weights)) printf("\033[92mShape(dW)==Shape(W)\033[0m\n");
      // if(size(G->DENSE->db)==size(G->DENSE->bias)) printf("\033[92mShape(db)==Shape(b)\033[0m\n");
      // sleep(1000);
      G = G->prev_layer;
    }
    // sleep(3000);
    // printf("Gradient Descent!\n");
    GD(0.05);
    G = temp;
    i++;
  }
  free2d(Y);
  G = G->prev_layer;
  printComputation_Graph(G);
  destroy_G(G);
}