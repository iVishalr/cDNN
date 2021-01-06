#include "../neural_net/neural_net.h"

void sleep(int milliseconds) {
  //Function to create a time delay. Mimicks thread.sleep() of Java
  unsigned int duration = time(0) + (milliseconds/1000);
  while(time(0)<duration);
}

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
  // int dims[] = {5,1};//5 features and one training eg
  // dARRAY * temp = randn(dims);
  // dARRAY * weight_input_res = dot(G->DENSE->weights, G->prev->DENSE->A);
  dARRAY * weight_input_res = dot(G->DENSE->weights, G->prev_layer->DENSE->A);
  dARRAY * Z = G->DENSE->cache = add(weight_input_res,G->DENSE->bias);//Z
  printf("\nFinished calculating Z\n");
  //sleep(2000);
  if(!strcmp(G->DENSE->activation,"relu")){
    G->DENSE->A = relu(.input=Z);
  }
  else if(!strcmp(G->DENSE->activation,"sigmoid")){
    G->DENSE->A = sigmoid(.input=Z);
  }
  else if(!strcmp(G->DENSE->activation,"tanh")){
    G->DENSE->A = TanH(.input=Z);
  }
  // free2d(temp);
  printf("\nCalculated Activation\n");
  //sleep(2000);
  free2d(weight_input_res);
  Z=NULL;
}
void backward_pass(){
  double m = 1;//temporarily set m=1;
  Dense_layer * layer = G->DENSE; 
  Dense_layer * prev_layer = G->prev_layer->DENSE;
  //calculate dZ 
  dARRAY * diff_layer_activation = NULL;
  if(!strcmp(layer->activation,"relu")){
    diff_layer_activation = relu(.input=layer->cache,.status=1);
  }
  else if(!strcmp(layer->activation,"sigmoid")){
    diff_layer_activation = sigmoid(.input=layer->cache,.status=1);
  }
  else if(!strcmp(layer->activation,"tanh")){
    diff_layer_activation = TanH(.input=layer->cache,.status=1);
  }
  layer->dZ = multiply(layer->dA,diff_layer_activation);
  printf("\nCalculated dZ\n");
  printf("Shape(dZ) : ");
  shape(G->DENSE->dZ);
  printf("\n");
  //sleep(2000);
  free2d(diff_layer_activation);

  //Calculate gradients with respect to the layer weights
  printf("\nCalculating Weight Gradients\n");
  dARRAY * prev_A_transpose = transpose(prev_layer->A);
  dARRAY * temp1_dW = dot(layer->dZ,prev_A_transpose);
  layer->dW = divScalar(temp1_dW,(1/(double)m));
  free2d(prev_A_transpose);
  free2d(temp1_dW);
  printf("\nCalculated dW\n");
  printf("Shape(dW) : ");
  shape(G->DENSE->dW);
  printf("\n");
  //sleep(2000);

  //calculate gradients with respect to the layer biases
  printf("\nCalculating Bias Gradients\n");
  dARRAY * temp1_db = sum(layer->dZ,1);
  layer->db = divScalar(temp1_db,(1/(double)m));
  free2d(temp1_db);
  printf("\nCalculated db\n");
  printf("Shape(db) : ");
  shape(G->DENSE->db);
  printf("\n");
  //sleep(2000);

  //calculate gradients of activation of prev layer
  if(prev_layer->cache!=NULL){
    dARRAY * weight_transpose = transpose(layer->weights);
    prev_layer->dA = dot(weight_transpose,layer->dZ);
    printf("\nCalculated dA_prev\n");
    printf("Shape(dA) : ");
    shape(G->prev_layer->DENSE->dA);
    printf("\n");
    //sleep(2000);
    free2d(weight_transpose);
  }
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
  Dense(.layer_size=5);
  Dense(.layer_size=5,.activation="relu",.initializer="he");
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random");
  printf("First Layer\nInitializer used : %s\n",G->DENSE->initializer);
  printf("Initializing params\n");
  int input_dims[] = {5,1};
  G->DENSE->A = randn(input_dims);
  shape(G->DENSE->A);
  printf("\nInput Layer's values : \n");
  for(int i=0;i<G->DENSE->A->shape[0];i++){
    for(int j=0;j<G->DENSE->A->shape[1];j++){
      printf("%lf ",G->DENSE->A->matrix[i*G->DENSE->A->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  G = G->next_layer;
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
  //sleep(2000);
  printf("B1 : \n");
  for(int i=0;i<G->DENSE->bias->shape[0];i++){
    for(int j=0;j<G->DENSE->bias->shape[1];j++){
      printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(2000);
  G = G->next_layer;
  //sleep(2000);
  printf("Second Layer\nInitializer used : %s\n",G->DENSE->initializer);
  printf("\nInitializing params\n");
  G->DENSE->initalize_params();
  //sleep(2000);
  printf("\nForward Propagating!\n");
  //sleep(2000);
  G->DENSE->forward_prop();
  //sleep(2000);
  printf("W2 : \n");
  for(int i=0;i<G->DENSE->weights->shape[0];i++){
    for(int j=0;j<G->DENSE->weights->shape[1];j++){
      printf("%lf ",G->DENSE->weights->matrix[i*G->DENSE->weights->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(2000);
  printf("A2 : \n");
  for(int i=0;i<G->DENSE->A->shape[0];i++){
    for(int j=0;j<G->DENSE->A->shape[1];j++){
      printf("%lf ",G->DENSE->A->matrix[i*G->DENSE->A->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(2000);
  printf("B2 : \n");
  for(int i=0;i<G->DENSE->bias->shape[0];i++){
    for(int j=0;j<G->DENSE->bias->shape[1];j++){
      printf("%lf ",G->DENSE->bias->matrix[i*G->DENSE->bias->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");

  //sleep(2000);
  //sleep(2000);

  printf("\nInitiating Backprop\n");
  int dims[] = {G->DENSE->A->shape[0],G->DENSE->A->shape[1]};
  dARRAY * Y =ones(dims);
  //sleep(2000);
  printf("Y = %lf\n",Y->matrix[0]);


  dARRAY * temp1 = NULL;
  dARRAY * temp2 = NULL;
  dARRAY * temp3 = NULL;
  dARRAY * temp4 = NULL;
  dARRAY * temp5 = NULL;
  dARRAY * temp6 = NULL;
  temp1 = divison(Y,G->DENSE->A);
  temp2 = subScalar(Y,1);
  temp3 = mulScalar(temp2,(double)-1);
  temp4 = subScalar(G->DENSE->A,1);
  temp5 = mulScalar(temp4,(double)-1);
  temp6 = divison(temp3,temp5);

  dARRAY * add_temp = subtract(temp1,temp6);

  G->DENSE->dA = mulScalar(add_temp,(double)-1);
  printf("\ndA2 = \n");
  for(int i=0;i<G->DENSE->dA->shape[0];i++){
    for(int j=0;j<G->DENSE->dA->shape[1];j++){
      printf("%lf ",G->DENSE->dA->matrix[i*G->DENSE->dA->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  printf("feering temp1\n");
  free2d(temp1);
  printf("feering temp2\n");
  free2d(temp2);
  printf("feering temp3\n");
  free2d(temp3);
  printf("feering temp4\n");
  free2d(temp4);
  printf("feering temp5\n");
  free2d(temp5);
  printf("feering temp6\n");
  free2d(temp6);
  printf("feering add_temp\n");
  free2d(add_temp);

  computation_graph_status = 1;
  printf("\nStarting to backpropagate Layer 2 : \n");
  //sleep(2000);
  G->DENSE->back_prop();
  //sleep(2000);
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
  //sleep(2000);
  printf("dA2 : \n");
  for(int i=0;i<G->DENSE->dA->shape[0];i++){
    for(int j=0;j<G->DENSE->dA->shape[1];j++){
      printf("%lf ",G->DENSE->dA->matrix[i*G->DENSE->dA->shape[1]+j]);
    }
    printf("\n");
  }
  printf("\n");
  //sleep(2000);
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
  //sleep(2000);
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
  printComputation_Graph(G);
  destroy_G(G);
}