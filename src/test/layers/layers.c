// #include "../../neural_net/neural_net.h"

// int main(){
//   G = NULL;
//   Input(.layer_size=5);
//   Dense(.layer_size=5,.activation="relu",.initializer="he",.dropout=0.5);
//   Dense(.layer_size=1,.activation="sigmoid",.initializer="random");
//   printf("First Layer\nInitializer used : %s\n",G->DENSE->initializer);
//   printf("Initializing params\n");
//   int input_dims[] = {5,4};
//   G->DENSE->A = randn(input_dims);
//   shape(G->DENSE->A);
//   printf("\nInput Layer's values : \n");
//   for(int i=0;i<G->DENSE->A->shape[0];i++){
//     for(int j=0;j<G->DENSE->A->shape[1];j++){
//       printf("%lf ",G->DENSE->A->matrix[i*G->DENSE->A->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   G = G->next_layer;
//   G->DENSE->initalize_params();
//   printf("init W1 : \n");
//   for(int i=0;i<G->DENSE->weights->shape[0];i++){
//     for(int j=0;j<G->DENSE->weights->shape[1];j++){
//       printf("%lf ",G->DENSE->weights->matrix[i*G->DENSE->weights->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(5000);
//   printf("init B1 : \n");
//   for(int i=0;i<G->DENSE->bias->shape[0];i++){
//     for(int j=0;j<G->DENSE->bias->shape[1];j++){
//       printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(5000);
//   printf("\nForward Propagating!\n");
//   //sleep(1000);
//   G->DENSE->forward_prop();
//   printf("W1 : \n");
//   for(int i=0;i<G->DENSE->weights->shape[0];i++){
//     for(int j=0;j<G->DENSE->weights->shape[1];j++){
//       printf("%lf ",G->DENSE->weights->matrix[i*G->DENSE->weights->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   printf("A1 : \n");
//   for(int i=0;i<G->DENSE->A->shape[0];i++){
//     for(int j=0;j<G->DENSE->A->shape[1];j++){
//       printf("%lf ",G->DENSE->A->matrix[i*G->DENSE->A->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   printf("B1 : \n");
//   for(int i=0;i<G->DENSE->bias->shape[0];i++){
//     for(int j=0;j<G->DENSE->bias->shape[1];j++){
//       printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   G = G->next_layer;
//   //sleep(2000);
//   printf("Second Layer\nInitializer used : %s\n",G->DENSE->initializer);
//   printf("\nInitializing params\n");
//   G->DENSE->initalize_params();
//   //sleep(2000);
//   printf("\nForward Propagating!\n");
//   //sleep(2000);
//   G->DENSE->forward_prop();
//   //sleep(2000);
//   printf("W2 : \n");
//   for(int i=0;i<G->DENSE->weights->shape[0];i++){
//     for(int j=0;j<G->DENSE->weights->shape[1];j++){
//       printf("%lf ",G->DENSE->weights->matrix[i*G->DENSE->weights->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   printf("A2 : \n");
//   for(int i=0;i<G->DENSE->A->shape[0];i++){
//     for(int j=0;j<G->DENSE->A->shape[1];j++){
//       printf("%lf ",G->DENSE->A->matrix[i*G->DENSE->A->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   printf("B2 : \n");
//   for(int i=0;i<G->DENSE->bias->shape[0];i++){
//     for(int j=0;j<G->DENSE->bias->shape[1];j++){
//       printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");

//   //sleep(2000);
//   //sleep(2000);

//   printf("\nInitiating Backprop\n");
//   int dims[] = {G->DENSE->A->shape[0],G->DENSE->A->shape[1]};
//   dARRAY * Y = ones(dims);
//   printf("Shape(Y) : ");
//   shape(Y);

//   dARRAY * temp1 = NULL;
//   dARRAY * temp2 = NULL;
//   dARRAY * temp3 = NULL;
//   dARRAY * temp4 = NULL;
//   dARRAY * temp5 = NULL;
//   dARRAY * temp6 = NULL;
//   temp1 = divison(Y,G->DENSE->A);
//   temp2 = subScalar(Y,1);
//   temp3 = mulScalar(temp2,(double)-1);
//   temp4 = subScalar(G->DENSE->A,1);
//   temp5 = mulScalar(temp4,(double)-1);
//   temp6 = divison(temp3,temp5);

//   dARRAY * add_temp = subtract(temp1,temp6);

//   G->DENSE->dA = mulScalar(add_temp,(double)-1);
//   printf("\ndA2 = \n");
//   for(int i=0;i<G->DENSE->dA->shape[0];i++){
//     for(int j=0;j<G->DENSE->dA->shape[1];j++){
//       printf("%lf ",G->DENSE->dA->matrix[i*G->DENSE->dA->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   free2d(temp1);
//   free2d(temp2);
//   free2d(temp3);
//   free2d(temp4);
//   free2d(temp5);
//   free2d(temp6);
//   free2d(add_temp);

//   computation_graph_status = 1;
//   printf("\nStarting to backpropagate Layer 2 : \n");
//   //sleep(2000);
//   G->DENSE->back_prop();
//   //sleep(2000);
//   printf("dW2 : \n");
//   for(int i=0;i<G->DENSE->dW->shape[0];i++){
//     for(int j=0;j<G->DENSE->dW->shape[1];j++){
//       printf("%lf ",G->DENSE->dW->matrix[i*G->DENSE->dW->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   printf("db2 : \n");
//   for(int i=0;i<G->DENSE->db->shape[0];i++){
//     for(int j=0;j<G->DENSE->db->shape[1];j++){
//       printf("%lf ",G->DENSE->db->matrix[i*G->DENSE->db->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   printf("dA2 : \n");
//   for(int i=0;i<G->DENSE->dA->shape[0];i++){
//     for(int j=0;j<G->DENSE->dA->shape[1];j++){
//       printf("%lf ",G->DENSE->dA->matrix[i*G->DENSE->dA->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   if(size(G->DENSE->dW)==size(G->DENSE->weights)) printf("\033[92mShape(dW2)==Shape(W2)\033[0m\n");
//   if(size(G->DENSE->db)==size(G->DENSE->bias)) printf("\033[92mShape(db2)==Shape(b2)\033[0m\n");

//   G = G->prev_layer;
//   G->DENSE->back_prop();

//   printf("dW1 : \n");
//   for(int i=0;i<G->DENSE->dW->shape[0];i++){
//     for(int j=0;j<G->DENSE->dW->shape[1];j++){
//       printf("%lf ",G->DENSE->dW->matrix[i*G->DENSE->dW->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   printf("db1 : \n");
//   for(int i=0;i<G->DENSE->db->shape[0];i++){
//     for(int j=0;j<G->DENSE->db->shape[1];j++){
//       printf("%lf ",G->DENSE->db->matrix[i*G->DENSE->db->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   printf("dA1 : \n");
//   for(int i=0;i<G->DENSE->dA->shape[0];i++){
//     for(int j=0;j<G->DENSE->dA->shape[1];j++){
//       printf("%lf ",G->DENSE->dA->matrix[i*G->DENSE->dA->shape[1]+j]);
//     }
//     printf("\n");
//   }
//   printf("\n");
//   //sleep(2000);
//   if(size(G->DENSE->dW)==size(G->DENSE->weights)) printf("\033[92mShape(dW1)==Shape(W1)\033[0m\n");
//   if(size(G->DENSE->db)==size(G->DENSE->bias)) printf("\033[92mShape(db1)==Shape(b1)\033[0m\n");
// //sleep(2000);
//   G = G->prev_layer;
// //sleep(2000);
//   printComputation_Graph(G);
//   destroy_G(G);
// }