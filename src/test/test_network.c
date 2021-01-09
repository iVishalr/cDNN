#include "../neural_net/neural_net.h"
// #include "../loss_functions/loss_functions.h"

extern Computation_Graph * G;
extern char * loss_type;

int main(){
  int feature_dims[] = {5,4};
  dARRAY * X = randn(feature_dims);
  Input(.layer_size=5,.input_features=X);
  Dense(.layer_size=5,.activation="relu",.initializer="he",.dropout=0.5,.layer_type="hidden");
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output");
  loss_type = "cross_entropy_loss";
  shape(G->INPUT->A);
  G = G->next_layer;
  printf("\nFirst Hidden layer : \n");
  G->DENSE->initalize_params();
  printf("init W1 : \n");
  for(int i=0;i<G->DENSE->weights->shape[0];i++){
    for(int j=0;j<G->DENSE->weights->shape[1];j++){
      printf("%lf ",G->DENSE->weights->matrix[i*G->DENSE->weights->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(5000);
  printf("init B1 : \n");
  for(int i=0;i<G->DENSE->bias->shape[0];i++){
    for(int j=0;j<G->DENSE->bias->shape[1];j++){
      printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(5000);
  printf("\nForward Propagating!\n");
  //sleep(1000);
  G->DENSE->forward_prop();
  printf("W1 : \n");
  for(int i=0;i<G->DENSE->weights->shape[0];i++){
    for(int j=0;j<G->DENSE->weights->shape[1];j++){
      printf("%lf ",G->DENSE->weights->matrix[i*G->DENSE->weights->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(2000);
  printf("A1 : \n");
  for(int i=0;i<G->DENSE->A->shape[0];i++){
    for(int j=0;j<G->DENSE->A->shape[1];j++){
      printf("%lf ",G->DENSE->A->matrix[i*G->DENSE->A->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("B1 : \n");
  for(int i=0;i<G->DENSE->bias->shape[0];i++){
    for(int j=0;j<G->DENSE->bias->shape[1];j++){
      printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  G = G->next_layer;
  printf("Second Hidden Layer\nInitializer used : %s\n",G->DENSE->initializer);
  printf("\nInitializing params\n");
  G->DENSE->initalize_params();
  printf("\nForward Propagating!\n");
  G->DENSE->forward_prop();

  printf("W2 : \n");
  for(int i=0;i<G->DENSE->weights->shape[0];i++){
    for(int j=0;j<G->DENSE->weights->shape[1];j++){
      printf("%lf ",G->DENSE->weights->matrix[i*G->DENSE->weights->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("A2 : \n");
  for(int i=0;i<G->DENSE->A->shape[0];i++){
    for(int j=0;j<G->DENSE->A->shape[1];j++){
      printf("%lf ",G->DENSE->A->matrix[i*G->DENSE->A->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("B2 : \n");
  for(int i=0;i<G->DENSE->bias->shape[0];i++){
    for(int j=0;j<G->DENSE->bias->shape[1];j++){
      printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");

  int dims[] = {G->DENSE->A->shape[0],G->DENSE->A->shape[1]};
  G->Y = ones(dims);
  printf("\033[96mCost : \033[0m%lf\n",cross_entropy_loss(G->DENSE,G->Y));
  printf("\nInitiating Backprop\n");
  printf("Shape(Y) : ");
  shape(G->Y);

  G->computation_graph_status = 1;
  printf("\nStarting to backpropagate Layer 2 : \n");
  G->DENSE->back_prop();
  printf("dW2 : \n");
  for(int i=0;i<G->DENSE->dW->shape[0];i++){
    for(int j=0;j<G->DENSE->dW->shape[1];j++){
      printf("%lf ",G->DENSE->dW->matrix[i*G->DENSE->dW->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(2000);
  printf("db2 : \n");
  for(int i=0;i<G->DENSE->db->shape[0];i++){
    for(int j=0;j<G->DENSE->db->shape[1];j++){
      printf("%lf ",G->DENSE->db->matrix[i*G->DENSE->db->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");

  if(size(G->DENSE->dW)==size(G->DENSE->weights)) printf("\033[92mShape(dW2)==Shape(W2)\033[0m\n");
  if(size(G->DENSE->db)==size(G->DENSE->bias)) printf("\033[92mShape(db2)==Shape(b2)\033[0m\n");

  G = G->prev_layer;
  G->DENSE->back_prop();

  printf("dW1 : \n");
  for(int i=0;i<G->DENSE->dW->shape[0];i++){
    for(int j=0;j<G->DENSE->dW->shape[1];j++){
      printf("%lf ",G->DENSE->dW->matrix[i*G->DENSE->dW->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("db1 : \n");
  for(int i=0;i<G->DENSE->db->shape[0];i++){
    for(int j=0;j<G->DENSE->db->shape[1];j++){
      printf("%lf ",G->DENSE->db->matrix[i*G->DENSE->db->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(2000);
  printf("dA1 : \n");
  for(int i=0;i<G->DENSE->dA->shape[0];i++){
    for(int j=0;j<G->DENSE->dA->shape[1];j++){
      printf("%lf ",G->DENSE->dA->matrix[i*G->DENSE->dA->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(2000);
  if(size(G->DENSE->dW)==size(G->DENSE->weights)) printf("\033[92mShape(dW1)==Shape(W1)\033[0m\n");
  if(size(G->DENSE->db)==size(G->DENSE->bias)) printf("\033[92mShape(db1)==Shape(b1)\033[0m\n");
//sleep(2000);
  G = G->prev_layer;
//sleep(2000);
  free2d(G->Y);
  printComputation_Graph(G);
  destroy_G(G);
}