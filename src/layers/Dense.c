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
    weight_dims[1] = m->input_size;
  }
  
  int bias_dims[] = {layer->num_of_computation_nodes,1};
  m->current_layer->DENSE->weights = init_weights(weight_dims,layer->initializer);
  m->current_layer->DENSE->bias = init_bias(bias_dims);
}

dARRAY * init_weights(int * weights_dims,const char * init_type){
  //Generate a random matrix with normal distribution
  dARRAY * temp = randn(weights_dims);
  
  dARRAY * weights = NULL;
  
  //create a scaling factor which will be used for proper
  //initialization of layer weights according to the
  //type of activation used for the layer.
  double scaling_factor = 0.0;
  scaling_factor = \
  m->current_layer->prev_layer->type!=INPUT ? \
  m->current_layer->prev_layer->DENSE->num_of_computation_nodes : \
  m->current_layer->prev_layer->INPUT->input_features_size;
  
  //He initialization - commonly used for ReLu activations
  if(!strcmp(init_type,"he")){
    weights = mulScalar(temp,sqrt(2.0/scaling_factor));
  }
  //Xavier initialization - commonly used for TanH activations
  else if(!strcmp(init_type,"xavier")){
    weights = mulScalar(temp,sqrt(1/scaling_factor));
  }
  //Simple random initalization - can be used for any activations
  //Warning
  //Initializing all layers to random will not make the network work properly
  //Spends lot of time for getting the optimal values of weights
  //for optimization to begin
  else if(!strcmp(init_type,"random")){
    weights = mulScalar(temp,0.01);
  }
  //Optional feature.
  //Don't use it
  //It is only to see what happens if your intialize all your weights to zeros
  //Fails to break symmetry and thus network wont train no matter what you do.
  else if(!strcmp(init_type,"zeros")){
    weights = zeros(weights_dims);
  }
  //Use He if an invalid initializer is specified
  else{
    printf("\033[93mInvalid initializer specified. Defaulting to He initialization.\033[0m\n");
    weights = mulScalar(temp,sqrt(2.0/scaling_factor));
  }

  free2d(temp);
  temp=NULL;

  return weights;
}

dARRAY * init_bias(int * bias_dims){
  //initialize biases of layer to zeros.
  //Doesn't matter if you init randomly or with only zeros.
  dARRAY * bias = zeros(bias_dims);
  return bias;
}

void forward_pass_DENSE(){
  // printf("dense forward\n");
  //Compute Z = W.A + b
  //Z is the linear output of the gate
  //W is the weights of the current layer
  //A is the activation of previous layer
  //b is a small bias offset
  dARRAY * Wx = NULL;
  //if the pevious layer is an input layer, we want to dot product the weights with the input
  //features and not the activations
  if(m->current_layer->prev_layer->type==INPUT) 
    Wx = \
    dot(m->current_layer->DENSE->weights, m->current_layer->prev_layer->INPUT->A);
  else 
    Wx = \
    dot(m->current_layer->DENSE->weights, m->current_layer->prev_layer->DENSE->A);
  //Store Z in cache as we will require it in backward pass
  dARRAY * Z = m->current_layer->DENSE->cache = add(Wx,m->current_layer->DENSE->bias);//Z
  //Done with Wx, free it.
  free2d(Wx);
  Wx = NULL;

  dARRAY * Z_drop_out = NULL;
  if(m->current_layer->DENSE->dropout<(double)1.0){
    //If dropout has been applied to the layer we need to create a dropout mask
    // printf("creating fropout msak\n");
    int dropout_mask_dims[] = {Z->shape[0],Z->shape[1]};
    dARRAY * dropout_mask_temp = NULL;
    dropout_mask_temp = randn(dropout_mask_dims);
    
    // #pragma omp parallel for num_threads(8) shared(dropout_mask_temp)
    for(int i=0;i<dropout_mask_dims[0]*dropout_mask_dims[1];i++){
      // printf("iteration %d\n",i);
      dropout_mask_temp->matrix[i] = dropout_mask_temp->matrix[i]<m->current_layer->DENSE->dropout ? (double)1.0 : (double)0.0;
    }
    m->current_layer->DENSE->dropout_mask = dropout_mask_temp;
    //we need to scale our activations so that loss doesnt change in the end as we will be forward propagating through
    //less neurons. This will change loss if activations are not scaled.
    Z_drop_out = divScalar(m->current_layer->DENSE->dropout_mask,m->current_layer->DENSE->dropout);
    // printf("done with dropout\n");
    // free2d(dropout_mask_temp);
    dropout_mask_temp = NULL;
  }
  //Compute the activation of this layer depending on the choice of activation selected.
  if(!strcasecmp(m->current_layer->DENSE->activation,"relu")){
    if(m->current_layer->DENSE->dropout==1.0) {
      m->current_layer->DENSE->A = relu(.input=Z); 
    }
    else{
      m->current_layer->DENSE->A = relu(.input=Z_drop_out);
      free2d(Z_drop_out);
      Z_drop_out = NULL;
    }
  }
  else if(!strcasecmp(m->current_layer->DENSE->activation,"sigmoid")){
    if(m->current_layer->DENSE->dropout==1.0) {
      m->current_layer->DENSE->A = sigmoid(.input=Z); 
    }
    else{
      m->current_layer->DENSE->A = sigmoid(.input=Z_drop_out);
      free2d(Z_drop_out);
      Z_drop_out = NULL;
    }
  }
  else if(!strcasecmp(m->current_layer->DENSE->activation,"tanh")){
    if(m->current_layer->DENSE->dropout==1.0) {
      m->current_layer->DENSE->A = TanH(.input=Z); 
    }
    else{
      // printf("dropout enabled!\n");
      m->current_layer->DENSE->A = TanH(.input=Z_drop_out);
      free2d(Z_drop_out);
      Z_drop_out = NULL;
    }
  }
  else{
    //if user didn't want to use any activation, pass Z itself as the output
    if(m->current_layer->DENSE->dropout==1.0) {
      
      m->current_layer->DENSE->A = Z; 
    }
    else{
      m->current_layer->DENSE->A = Z_drop_out;
      Z_drop_out = NULL;
    }
  }

  Z=NULL;

  //set output of model to be activation of the last layer
  if(m->current_layer->next_layer->type==LOSS)
    m->output = m->current_layer->DENSE->A;
}

void backward_pass_DENSE(){
  //Store the number of training examples in a variable
  double num_examples = m->num_of_training_examples;
  //Assign pointers to the respective layers
  Dense_layer * layer = m->current_layer->DENSE; 
  Input_layer * prev_layer_in_features = NULL;
  Dense_layer * prev_layer = NULL;

  if(m->current_layer->prev_layer->type==INPUT){
    prev_layer_in_features = m->current_layer->prev_layer->INPUT;
  }
  else 
    prev_layer = m->current_layer->prev_layer->DENSE;

  //Compute g'(z)
  //where g is the activation function used for the layer
  //g' is the differentiation of the activation function
  //.status=1 - indicates that the activation function must perform backward pass
  dARRAY * local_act_grad = NULL;
  if(!strcasecmp(layer->activation,"relu")){
    //local grad of relu gate
    local_act_grad = relu(.input=layer->cache,.status=1);
  }
  else if(!strcasecmp(layer->activation,"sigmoid")){
    //local grad of sigmoid gate
    local_act_grad = sigmoid(.input=layer->cache,.status=1);
  }
  else if(!strcasecmp(layer->activation,"tanh")){
    //local grad of tanh gate
    local_act_grad = TanH(.input=layer->cache,.status=1);
  }
  if(m->current_layer->next_layer->type==LOSS){
    //If we are on the last layer, then the gradient flowing
    //into the Z computation block will be by chain rule
    // dZ = local_act_grad * global_grad (loss_layer->grad_out)
    layer->dZ = multiply(local_act_grad,m->current_layer->next_layer->LOSS->grad_out);
    free2d(local_act_grad);
    free2d(m->current_layer->next_layer->LOSS->grad_out);
    local_act_grad = m->current_layer->next_layer->LOSS->grad_out = NULL;
  }
  else{
    //If we are not on the last layer then, we need to calculate dZ differently
    //dZ[1] = g'(Z[1]) * (W[2].dZ[2])
    //[1], [2] - represents layers, 1 - first layer, 2 - second layer so on...
    
    //calculating W[2].dZ[2]
    dARRAY * weight_trans = transpose(m->current_layer->next_layer->DENSE->weights);
    dARRAY * temp_dz = NULL;
    temp_dz = dot(weight_trans,m->current_layer->next_layer->DENSE->dZ);
    free2d(weight_trans);
    weight_trans = NULL;
    
    //now we have the global gradient computed. We need to chain it with
    //the local gradient and make it flow to dZ[1]
    layer->dZ = multiply(local_act_grad,temp_dz);

    free2d(temp_dz);
    temp_dz = NULL;
    free2d(local_act_grad);
    local_act_grad = NULL;
  }

  //We have calculated dZ[current layer now] we can use it to calculate the remaining grads
  //Calculate gradients with respect to the layer weights
  dARRAY * prev_A_transpose = NULL;

  if(m->current_layer->prev_layer->type==INPUT) 
    prev_A_transpose = transpose(prev_layer_in_features->A);
  else prev_A_transpose = transpose(prev_layer->A);
  dARRAY * temp1_dW = NULL;
  temp1_dW = dot(layer->dZ,prev_A_transpose);

  free2d(prev_A_transpose);
  prev_A_transpose = NULL;

  if(m->lambda>(double)0.0){
    double mul_factor = m->lambda/(double)num_examples;
    dARRAY * regularization_grad = mulScalar(temp1_dW,mul_factor);
    dARRAY * dW_temp = divScalar(temp1_dW,(double)num_examples);
    
    free2d(temp1_dW);
    temp1_dW = NULL;
    
    layer->dW = add(dW_temp,regularization_grad);
    
    free2d(regularization_grad);
    free2d(dW_temp);
    regularization_grad = dW_temp = NULL;
  }
  else if(m->lambda==0.0){
    layer->dW = divScalar(temp1_dW,(double)num_examples);
    free2d(temp1_dW);
    temp1_dW = NULL;
  }
  else{
    printf("\033[1;31mValue Error : \033[93mInvalid lambda value specified\033[0m\n");
    exit(EXIT_FAILURE);
  }
  //Now we have calculated the gradients with respect to the weights
  //calculate gradients with respect to the layer biases
  dARRAY * temp1_db = sum(layer->dZ,1);
  layer->db = divScalar(temp1_db,(double)num_examples);
  
  free2d(temp1_db);
  temp1_db = NULL;

  //Finally we need to calculate the gradients of the prev layer activation
  //calculate gradients of activation of prev layer
  if(m->current_layer->prev_layer->type!=INPUT){
    //local gradient would be just the current layer weights
    dARRAY * weight_transpose = NULL;
    weight_transpose = transpose(layer->weights);
    //chaining with the global or incomming gradient
    dARRAY * prev_layer_A_temp = NULL;
    prev_layer_A_temp = dot(weight_transpose,layer->dZ);
    if(layer->dropout==(double)1.0){
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
  layer = NULL;
  prev_layer = NULL;
}

void (Dense)(dense_args dense_layer_args){
  
  Dense_layer * layer = (Dense_layer*)malloc(sizeof(Dense_layer));
  layer->activation = dense_layer_args.activation;
  layer->num_of_computation_nodes = dense_layer_args.layer_size;
  layer->initalize_params = init_params;
  layer->initializer = dense_layer_args.initializer;
  layer->cache = NULL;
  layer->forward = forward_pass_DENSE;
  layer->backward = backward_pass_DENSE;
  layer->dropout_mask = NULL;
  layer->dropout = dense_layer_args.dropout;
  layer->lambda = dense_layer_args.lambda;
  layer->dA = layer->db = layer->dW = layer->dZ = NULL;
  layer->isTraining = 1;
  layer->layer_type = dense_layer_args.layer_type;
  //finally we need to append to computation graph
  append_graph(layer,"Dense");
}