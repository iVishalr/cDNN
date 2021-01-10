#include "../neural_net/neural_net.h"
#include "./Dense.h"
#include "../model/model.h"

extern __Model__ * m;

void init_params(){
  Dense_layer * layer = m->current_layer->DENSE;
  int weight_dims[2];
  if(m->current_layer->prev_layer->type!=INPUT){
    weight_dims[0] = layer->num_of_computation_nodes;
    weight_dims[1] = m->current_layer->prev_layer->DENSE->num_of_computation_nodes;
  }
  else{
    weight_dims[0] = layer->num_of_computation_nodes;
    weight_dims[1] = 12288;
  }
  int bias_dims[] = {layer->num_of_computation_nodes,1};
  m->current_layer->DENSE->weights = init_weights(weight_dims,layer->initializer);
  m->current_layer->DENSE->bias = init_bias(bias_dims);
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
  if(m->current_layer->prev_layer->type==INPUT) weight_input_res = dot(m->current_layer->DENSE->weights, m->current_layer->prev_layer->INPUT->A);
  else weight_input_res = dot(m->current_layer->DENSE->weights, m->current_layer->prev_layer->DENSE->A);
  dARRAY * Z = m->current_layer->DENSE->cache = add(weight_input_res,m->current_layer->DENSE->bias);//Z
  free2d(weight_input_res);
  weight_input_res = NULL;
  dARRAY * activation_temp = NULL;
  if(!strcmp(m->current_layer->DENSE->activation,"relu")){
    activation_temp = relu(.input=Z);
  }
  else if(!strcmp(m->current_layer->DENSE->activation,"sigmoid")){
    activation_temp = sigmoid(.input=Z);
  }
  else if(!strcmp(m->current_layer->DENSE->activation,"tanh")){
    activation_temp = TanH(.input=Z);
  }
  if(m->current_layer->DENSE->dropout<1.0 && m->current_layer->DENSE->dropout>=0.0 && m->current_layer->DENSE->isTraining){
    //implementation of inverted dropout layer
    int dims[] = {activation_temp->shape[0],activation_temp->shape[1]};
    m->current_layer->DENSE->dropout_mask = randn(dims);
    omp_set_num_threads(4);
    #pragma omp parallel for
    for(int i=0;i<m->current_layer->DENSE->dropout_mask->shape[0]*m->current_layer->DENSE->dropout_mask->shape[1];i++)
      m->current_layer->DENSE->dropout_mask->matrix[i] = m->current_layer->DENSE->dropout_mask->matrix[i]<m->current_layer->DENSE->dropout ? 1 : 0;
    dARRAY * mul_mask = multiply(activation_temp,m->current_layer->DENSE->dropout_mask);
    m->current_layer->DENSE->A = divScalar(mul_mask,m->current_layer->DENSE->dropout);
    free2d(mul_mask);
    mul_mask = NULL;
  }
  else{
    m->current_layer->DENSE->A = activation_temp;
  }
  activation_temp = NULL;
  Z=NULL;
  if(m->predicting)
    m->output = m->current_layer->DENSE->A;
}

void backward_pass(){
  double num_examples = 12288;//temporarily set m=1;

  Dense_layer * layer = m->current_layer->DENSE; 
  Input_layer * prev_layer_in_features = NULL;
  Dense_layer * prev_layer = NULL;

  if(m->current_layer->prev_layer->type==INPUT) 
    prev_layer_in_features = m->current_layer->prev_layer->INPUT;
  else 
    prev_layer = m->current_layer->prev_layer->DENSE;

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
  }
  else{
    // printf("Calculating dZ2\n");
    if(!strcmp(m->loss,"cross_entropy_loss")){
      m->current_layer->DENSE->dZ = subtract(m->current_layer->DENSE->A,m->Y_train);
    }
  }

  //Calculate gradients with respect to the layer weights
  // printf("\nCalculating Weight Gradients\n");
  dARRAY * prev_A_transpose = NULL;
  if(m->current_layer->prev_layer->type==INPUT) 
    prev_A_transpose = transpose(prev_layer_in_features->A);
  else prev_A_transpose = transpose(prev_layer->A);
  dARRAY * temp1_dW = dot(layer->dZ,prev_A_transpose);
  free2d(prev_A_transpose);
  prev_A_transpose = NULL;
  dARRAY * regularization_grad_temp = mulScalar(layer->weights,layer->lambda);
  dARRAY * regularization_grad = divScalar(regularization_grad_temp,num_examples);
  free2d(regularization_grad_temp);
  regularization_grad_temp = NULL;
  dARRAY * dW_temp = mulScalar(temp1_dW,(1/(double)num_examples));
  free2d(temp1_dW);
  temp1_dW = NULL;
  layer->dW = add(dW_temp,regularization_grad);
  free2d(regularization_grad);
  free2d(dW_temp);
  regularization_grad = dW_temp = NULL;

  //calculate gradients with respect to the layer biases
  // printf("\nCalculating Bias Gradients\n");
  dARRAY * temp1_db = sum(layer->dZ,1);
  layer->db = divScalar(temp1_db,(1/(double)num_examples));
  free2d(temp1_db);
  temp1_db = NULL;
  // printf("\n");
  //sleep(2000);

  //calculate gradients of activation of prev layer
  if(m->current_layer->prev_layer->type!=INPUT){
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
  }
  free2d(layer->dZ);
  layer->dZ = NULL;
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
  layer->layer_num = dense_layer_args.layer_num;
  //finally we need to append to computation graph
  append_graph(layer,"Dense");
}