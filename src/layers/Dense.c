#include "../neural_net/neural_net.h"

void init_params(){
  Dense_layer * layer = G->DENSE;
  int weight_dims[] = {layer->num_of_computation_nodes,5};
  int bias_dims[] = {layer->num_of_computation_nodes,1};
  G->DENSE->weights = init_weights(weight_dims,layer->initializer);
  G->DENSE->bias = init_bias(bias_dims);
}

dARRAY * init_weights(int * weights_dims,const char * init_type){
  dARRAY * temp = randn(weights_dims);
  dARRAY * weights = NULL;
  if(!strcmp(init_type,"he")){
    weights = mulScalar(temp,sqrt(2.0/5.0));//5 is size of prev layer
  }
  else if(!strcmp(init_type,"xavier")){
    weights = mulScalar(temp,sqrt(2.0/(5.0+(double)weights_dims[0])));
  }
  else if(!strcmp(init_type,"random")){
    weights = mulScalar(temp,0.01);
  }
  else{
    printf("\033[93mInvalid initializer specified. Defaulting to He initialization.\033[0m\n");
    weights = mulScalar(temp,sqrt(2.0/5.0));
  }
  return weights;
}

dARRAY * init_bias(int * bias_dims){
  dARRAY * bias = zeros(bias_dims);
  return bias;
}

void forward_pass(){
  int dims[] = {5,1};//5 features and one training eg
  dARRAY * temp = randn(dims);
  // dARRAY * weight_input_res = dot(G->DENSE->weights, G->prev->DENSE->A);
  dARRAY * weight_input_res = dot(G->DENSE->weights, temp);
  dARRAY * Z = G->DENSE->cache = add(weight_input_res,G->DENSE->bias);//Z
  if(!strcmp(G->DENSE->activation,"relu")){
    G->DENSE->A = relu(.input=Z);
  }
  else if(!strcmp(G->DENSE->activation,"sigmoid")){
    G->DENSE->A = sigmoid(.input=Z);
  }
  else if(!strcmp(G->DENSE->activation,"tanh")){
    G->DENSE->A = TanH(.input=Z);
  }
  free(temp);
  free(weight_input_res);
  Z=NULL;
}
void backward_pass(Dense_layer * layer, Dense_layer * prev_layer){
  
}

void (Dense)(dense_args dense_layer_args){
  Dense_layer * layer = (Dense_layer*)malloc(sizeof(Dense_layer));
  layer->activation = dense_layer_args.activation;
  layer->num_of_computation_nodes = dense_layer_args.layer_size;
  layer->initalize_params = init_params;
  layer->initializer = dense_layer_args.initializer;
  layer->cache = NULL;
  layer->forward_prop = forward_pass;
  layer->back_prop = backward_pass;
  //finally we need to append to computation graph
  printf("Appending DENSE\n");
  G = append_graph(G,layer,"Dense");
}

int main(){
  G = NULL;
  Dense(.layer_size=5,.activation="relu",.initializer="he");
  Dense(.layer_size=5,.activation="sigmoid",.initializer="random");
  printf("First Layer\nInitializer used : %s\n",G->DENSE->initializer);
  printf("Initializing params\n");
  G->DENSE->initalize_params();
  printf("Forward Propagating!\n");
  G->DENSE->forward_prop();
  printf("Weights of first layer : \n");
  for(int i=0;i<G->DENSE->weights->shape[0];i++){
    for(int j=0;j<G->DENSE->weights->shape[1];j++){
      printf("%lf ",G->DENSE->weights->matrix[i*G->DENSE->weights->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("Activation of first layer : \n");
  for(int i=0;i<G->DENSE->A->shape[0];i++){
    for(int j=0;j<G->DENSE->A->shape[1];j++){
      printf("%lf ",G->DENSE->A->matrix[i*G->DENSE->A->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("Bias of first layer : \n");
  for(int i=0;i<G->DENSE->bias->shape[0];i++){
    for(int j=0;j<G->DENSE->bias->shape[1];j++){
      printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  G = G->next_layer;
  printf("Second Layer\nInitializer used : %s\n",G->DENSE->initializer);
  printf("Initializing params\n");
  G->DENSE->initalize_params();
  printf("Forward Propagating!\n");
  G->DENSE->forward_prop();

  printf("Weights of second layer : \n");
  for(int i=0;i<G->DENSE->weights->shape[0];i++){
    for(int j=0;j<G->DENSE->weights->shape[1];j++){
      printf("%lf ",G->DENSE->weights->matrix[i*G->DENSE->weights->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("Activation of second layer : \n");
  for(int i=0;i<G->DENSE->A->shape[0];i++){
    for(int j=0;j<G->DENSE->A->shape[1];j++){
      printf("%lf ",G->DENSE->A->matrix[i*G->DENSE->A->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");

  printf("Bias of second layer : \n");
  for(int i=0;i<G->DENSE->bias->shape[0];i++){
    for(int j=0;j<G->DENSE->bias->shape[1];j++){
      printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  printComputation_Graph(G);
  destroy_G(G);
}