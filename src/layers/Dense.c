#include "../neural_net/neural_net.h"
#include "./Dense.h"

// extern Computation_Graph * G;

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

  dARRAY * weight_input_res = NULL;
  if(G->prev_layer->type==INPUT) weight_input_res = dot(G->DENSE->weights, G->prev_layer->INPUT->A);
  else weight_input_res = dot(G->DENSE->weights, G->prev_layer->DENSE->A);
  dARRAY * Z = G->DENSE->cache = add(weight_input_res,G->DENSE->bias);//Z
  free2d(weight_input_res);
  weight_input_res = NULL;
  printf("Shape(Z) : ");
  shape(Z);
  dARRAY * activation_temp = NULL;
  if(!strcmp(G->DENSE->activation,"relu")){
    activation_temp = relu(.input=Z);
    printf("Shape(activation_temp) : ");
    shape(activation_temp);
  }
  else if(!strcmp(G->DENSE->activation,"sigmoid")){
    activation_temp = sigmoid(.input=Z);
    printf("Shape(activation_temp) : ");
    shape(activation_temp);
  }
  else if(!strcmp(G->DENSE->activation,"tanh")){
    activation_temp = TanH(.input=Z);
    printf("Shape(activation_temp) : ");
    shape(activation_temp);
  }
  if(G->DENSE->dropout<1.0 && G->DENSE->dropout>=0.0 && G->DENSE->isTraining){
    //implementation of inverted dropout layer
    int dims[] = {activation_temp->shape[0],activation_temp->shape[1]};
    G->DENSE->dropout_mask = randn(dims);
    omp_set_num_threads(4);
    #pragma omp parallel for
    for(int i=0;i<G->DENSE->dropout_mask->shape[0]*G->DENSE->dropout_mask->shape[1];i++)
      G->DENSE->dropout_mask->matrix[i] = G->DENSE->dropout_mask->matrix[i]<G->DENSE->dropout ? 1 : 0;
    dARRAY * mul_mask = multiply(activation_temp,G->DENSE->dropout_mask);
    G->DENSE->A = divScalar(mul_mask,G->DENSE->dropout);
    free2d(mul_mask);
    mul_mask = NULL;
  }
  else{
    G->DENSE->A = activation_temp;
    printf("Shape(A) : ");
    shape(G->DENSE->A);
  }
  activation_temp = NULL;
  Z=NULL;
}

void backward_pass(){
  double m = 4;//temporarily set m=1;
  Dense_layer * layer = G->DENSE; 
  Input_layer * prev_layer_in_features = NULL;
  Dense_layer * prev_layer = NULL;
  if(G->prev_layer->type==INPUT) prev_layer_in_features = G->prev_layer->INPUT;
  else prev_layer = G->prev_layer->DENSE;
  //calculate dZ
  if(strcmp(layer->layer_type,"output")!=0){ 
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
    free2d(diff_layer_activation);
    diff_layer_activation = NULL;
    printf("Shape(dZ) : ");
    shape(G->DENSE->dZ);
    printf("\n");
  }
  else{
    if(!strcmp(G->loss_type,"cross_entropy_loss")){
      G->DENSE->dZ = subtract(G->DENSE->A,G->Y);
    }
  }

  //Calculate gradients with respect to the layer weights
  printf("\nCalculating Weight Gradients\n");
  dARRAY * prev_A_transpose = NULL;
  if(G->prev_layer->type==INPUT) prev_A_transpose = transpose(prev_layer_in_features->A);
  else prev_A_transpose = transpose(prev_layer->A);
  dARRAY * temp1_dW = dot(layer->dZ,prev_A_transpose);
  free2d(prev_A_transpose);
  prev_A_transpose = NULL;
  dARRAY * regularization_grad_temp = mulScalar(layer->weights,layer->lambda);
  dARRAY * regularization_grad = divScalar(regularization_grad_temp,m);
  free2d(regularization_grad_temp);
  regularization_grad_temp = NULL;
  dARRAY * dW_temp = mulScalar(temp1_dW,(1/(double)m));
  free2d(temp1_dW);
  temp1_dW = NULL;
  layer->dW = add(dW_temp,regularization_grad);
  free2d(regularization_grad);
  free2d(dW_temp);
  regularization_grad = dW_temp = NULL;
  printf("Shape(dW) : ");
  shape(G->DENSE->dW);
  printf("\n");

  //calculate gradients with respect to the layer biases
  printf("\nCalculating Bias Gradients\n");
  dARRAY * temp1_db = sum(layer->dZ,1);
  shape(temp1_db);
  layer->db = divScalar(temp1_db,(1/(double)m));
  free2d(temp1_db);
  temp1_db = NULL;
  printf("\nCalculated db\n");
  printf("Shape(db) : ");
  shape(layer->db);
  printf("\n");
  //sleep(2000);

  //calculate gradients of activation of prev layer
  if(G->prev_layer->type!=INPUT){
    dARRAY * weight_transpose = transpose(layer->weights);
    dARRAY * prev_layer_A_temp = dot(weight_transpose,layer->dZ);
    if(layer->dropout_mask==NULL){
      prev_layer->dA = prev_layer_A_temp;
      prev_layer_A_temp = NULL;
    }
    else{
      dARRAY * prev_layer_A_masked = multiply(prev_layer_A_temp,prev_layer->dropout_mask);
      prev_layer->dA = divScalar(prev_layer_A_masked,prev_layer->dropout);
      free2d(prev_layer_A_temp);
      prev_layer_A_temp = NULL; 
      free2d(prev_layer_A_masked);
      prev_layer_A_masked = NULL;
    }
    free2d(weight_transpose);
    weight_transpose = NULL;
    printf("\nCalculated dA_prev\n");
    printf("Shape(dA) : ");
    shape(G->prev_layer->DENSE->dA);
    printf("\n");
  }
  else{
    printf("skipping calculation of dA_prev\n");
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
  layer->dropout_mask = NULL;
  layer->dropout = dense_layer_args.dropout;
  layer->lambda = dense_layer_args.lambda;
  layer->dA = layer->db = layer->dW = layer->dZ = NULL;
  layer->isTraining = 1;
  layer->layer_type = dense_layer_args.layer_type;
  //finally we need to append to computation graph
  append_graph(layer,"Dense");
}