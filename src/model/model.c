#include "./model.h"

__Model__ * m;

void __init__(){
  m->predicting = 0;
  m->test_accuracy = 0.0;
  m->train_accuracy = 0.0;
  m->train_cost = 0.0;
  m->cross_val_accuracy = 0.0;
  m->iter_cost = 0.0;
  m->output = NULL;
  __initialize_params__();
}

void __initialize_params__(){
  Computation_Graph * temp = m->graph;
  temp = temp->next_layer;
  int index = 0;
  int flag = 0;
  if(!strcasecmp(m->optimizer,"adam")) flag = 1;
  if(!strcasecmp(m->optimizer,"adagrad")) flag = 2;
  if(!strcasecmp(m->optimizer,"rmsprop")) flag = 2;
  while(temp!=NULL){
    m->current_layer = temp;
    if(temp->type==DENSE){
      temp->DENSE->initalize_params();
    }
    if(flag==1 && temp->type!=LOSS){
      int dims_dW[] = {temp->DENSE->weights->shape[0],temp->DENSE->weights->shape[1]};
      int dims_db[] = {temp->DENSE->bias->shape[0],temp->DENSE->bias->shape[1]};
      m->m_t_dW[index] = zeros(dims_dW);
      m->m_t_db[index] = zeros(dims_db);
      m->v_t_dW[index] = zeros(dims_dW);
      m->v_t_db[index] = zeros(dims_db);
      index++;
    }
    else if(flag==2 && temp->type!=LOSS){
      int dims_dW[] = {temp->DENSE->weights->shape[0],temp->DENSE->weights->shape[1]};
      int dims_db[] = {temp->DENSE->bias->shape[0],temp->DENSE->bias->shape[1]};
      m->cache_dW[index] = zeros(dims_dW);
      m->cache_db[index] = zeros(dims_db);
      index++; 
    }
    temp = temp->next_layer;
  }
  m->current_layer = m->graph;
}

void __forward__(){
  Computation_Graph * temp = m->graph;
  while(temp!=NULL){
    m->current_layer = temp;
    if(temp->type==INPUT){ temp->INPUT->forward();}
    else if(temp->type==DENSE) {temp->DENSE->forward();}
    else if(temp->type==LOSS) {temp->LOSS->forward();}
    temp = temp->next_layer;
  }
}

void __backward__(){
  Computation_Graph * temp = m->current_layer;
  while(temp->prev_layer!=NULL){
    m->current_layer = temp;
    if(temp->type==INPUT) temp->INPUT->backward();
    else if(temp->type==DENSE) temp->DENSE->backward();
    else if(temp->type==LOSS) temp->LOSS->backward();
    temp = temp->prev_layer;
  }
}

double calculate_accuracy(dARRAY * predicted, dARRAY * gnd_truth){
  int success = 0;
  dARRAY * temp = (dARRAY*)malloc(sizeof(dARRAY));
  temp->matrix = (double*)calloc(predicted->shape[0]*predicted->shape[1],sizeof(double));
  for(int i = 0;i<predicted->shape[0]*predicted->shape[1];i++){
    temp->matrix[i] = predicted->matrix[i]<0.5 ? 0 : 1;
  }
  temp->shape[0] = predicted->shape[0];
  temp->shape[1] = predicted->shape[1];

  for(int i = 0;i<predicted->shape[0]*predicted->shape[1];i++){
    if(temp->matrix[i]==gnd_truth->matrix[i]) success++;
  }
  free2d(temp);
  temp = NULL;
  return success/(double)predicted->shape[1];
}

dARRAY * relu_val(dARRAY * linear_matrix){
  dARRAY * relu_outf = NULL;
  relu_outf = (dARRAY*)malloc(sizeof(dARRAY));
  relu_outf->matrix = (double*)calloc(linear_matrix->shape[0]*linear_matrix->shape[1],sizeof(double));
  omp_set_num_threads(4);
  #pragma omp parallel for
  for(int i=0;i<linear_matrix->shape[0]*linear_matrix->shape[1];i++)
    relu_outf->matrix[i] = linear_matrix->matrix[i]>(double)0.0 ?(double)linear_matrix->matrix[i] : (double)0.0;
  relu_outf->shape[0] = linear_matrix->shape[0];
  relu_outf->shape[1] = linear_matrix->shape[1];
  return relu_outf;
}

dARRAY * sigmoid_val(dARRAY * linear_matrix){
  dARRAY * sigmoid_outf = NULL;
  sigmoid_outf = (dARRAY*)malloc(sizeof(dARRAY));
  sigmoid_outf->matrix = (double*)calloc(linear_matrix->shape[0]*linear_matrix->shape[1],sizeof(double));
  omp_set_num_threads(4);
  #pragma omp parallel for
  for(int i=0;i<linear_matrix->shape[0]*linear_matrix->shape[1];i++)
    sigmoid_outf->matrix[i] = (double)(1.0/(double)(1+exp((double)(-1.0*linear_matrix->matrix[i]))));
  sigmoid_outf->shape[0] = linear_matrix->shape[0];
  sigmoid_outf->shape[1] = linear_matrix->shape[1];
  return sigmoid_outf;
}

dARRAY * tanh_val(dARRAY * linear_matrix){
  dARRAY * tanh_out = (dARRAY*)malloc(sizeof(dARRAY));
  tanh_out->matrix = (double*)calloc(linear_matrix->shape[0]*linear_matrix->shape[1],sizeof(double));
  omp_set_num_threads(4);
  #pragma omp parallel for
  for(int i=0;i<linear_matrix->shape[0]*linear_matrix->shape[1];i++){
    //Computing the tanh function
    double exp_res1 = exp(linear_matrix->matrix[i]);
    double exp_res2 = exp(-1*linear_matrix->matrix[i]);
    tanh_out->matrix[i] = (exp_res1 - exp_res2)/(exp_res1 + exp_res2);
  }
  tanh_out->shape[0] = linear_matrix->shape[0];
  tanh_out->shape[1] = linear_matrix->shape[1];
  return tanh_out;
}

double calculate_train_val_acc(){
  dARRAY * weight_input_res = NULL;
  dARRAY * output = NULL;
  dARRAY * activation_temp = NULL;

  Computation_Graph * temp = m->graph->next_layer;
  int layer=0;
  while(temp!=NULL){
    if(temp->prev_layer->type==INPUT){ 
    weight_input_res = dot(temp->DENSE->weights, m->x_cv); 
    }
    else{ 
      weight_input_res = dot(temp->DENSE->weights, activation_temp);
    }
    if(activation_temp!=NULL){
       free2d(activation_temp);
       activation_temp = NULL;
    }
    dARRAY * Z = add(weight_input_res,temp->DENSE->bias);//Z
    free2d(weight_input_res);
    weight_input_res = NULL;
    if(!strcasecmp(temp->DENSE->activation,"relu")){
      activation_temp = relu_val(Z);
      free2d(Z);
    }
    else if(!strcasecmp(temp->DENSE->activation,"sigmoid")){
      activation_temp = sigmoid_val(Z);
      free2d(Z);
    }
    else if(!strcasecmp(temp->DENSE->activation,"tanh")){
      activation_temp = tanh_val(Z);
      free2d(Z);
    }
    else{
      activation_temp = Z;
    }
    if(temp->next_layer->type==LOSS){
      output = activation_temp;
      Z = NULL;
      break;
    }
    layer++;
    temp = temp->next_layer;
    Z = NULL;
  }
  double val_acc = calculate_accuracy(output,m->Y_cv);
  free2d(output);
  output = NULL;
  return (1-val_acc);
}

void append_to_file(double * arr ,char * filename,char * mode){
  FILE * fp = NULL;
  fp = fopen(filename,mode);
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  for(int i=0;i<m->num_iter;i++)
    fprintf(fp,"%lf ",arr[i]);
  fclose(fp);
}

void __fit__(){
  int i = 1;
  double sum_cost = 0.0;
  double sum_train_acc = 0.0;
  double sum_train_val_acc = 0.0;
  double * train_cost_arr = (double*)calloc(m->num_iter,sizeof(double));
  double * train_acc_arr = (double*)calloc(m->num_iter,sizeof(double));
  double * val_acc_arr = (double*)calloc(m->num_iter,sizeof(double));
  while(i<=m->num_iter){
    __forward__();
    __backward__();
    
    sum_cost += m->iter_cost;
    sum_train_acc += calculate_accuracy(m->output,m->Y_train);
    sum_train_val_acc += calculate_train_val_acc();
    if(m->print_cost){
      m->train_cost = sum_cost/(double)i;
      m->train_accuracy = sum_train_acc/(double)i;
      m->cross_val_accuracy = sum_train_val_acc/(double)i;
      train_cost_arr[i] = m->train_cost;
      train_acc_arr[i] = m->train_accuracy;
      val_acc_arr[i] = m->cross_val_accuracy;

      printf("\033[96m%d. Cost : \033[0m%lf ",i,m->train_cost);
      printf("\033[96m train_acc : \033[0m%lf ",m->train_accuracy);
      printf("\033[96m val_acc : \033[0m%lf\n",m->cross_val_accuracy);
    }
    GD(m->learning_rate);
    i++;
    // if(i==5) break;
  }
  append_to_file(train_cost_arr,"./bin/cost.data","ab+");
  append_to_file(train_acc_arr,"./bin/train_acc.data","ab+");
  append_to_file(val_acc_arr,"./bin/val_acc.data","ab+");
}

void __predict__(dARRAY * input_feature){
  m->graph->INPUT->A = input_feature;
  m->predicting = 1;
  __forward__();
}


void __load_model__(char * filename){
  if(strstr(filename,".t7")==NULL){
    printf("\033[1;31mFileExtension Error : \033[93m Please use \".t7\" extension only. Other extensions are not supported currently.\033[0m\n");
    exit(EXIT_FAILURE);
  }
  char destpath[1024];
  snprintf (destpath, sizeof(destpath), "./model/%s", filename);
  dARRAY weights[m->number_of_layers-1];
  dARRAY biases[m->number_of_layers-1];

  Computation_Graph * temp = m->graph->next_layer;

  for(int i=0;i<m->number_of_layers-1 && temp!=NULL;i++){
    weights[i].matrix = (double*)calloc(temp->DENSE->weights->shape[0]*temp->DENSE->weights->shape[1],sizeof(double));
    biases[i].matrix = (double*)calloc(temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1],sizeof(double));
    temp = temp->next_layer;
  }
  temp = m->graph->next_layer;
  FILE * fp = NULL;
  fp = fopen(destpath,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  for(int i=0;i<m->number_of_layers-1;i++){
    if(temp==NULL) printf("null\n");
    for(int j=0;j<temp->DENSE->weights->shape[0]*temp->DENSE->weights->shape[1];j++){
      fscanf(fp,"%lf ",&weights[i].matrix[j]);
    }
    weights[i].shape[0] = temp->DENSE->weights->shape[0];
    weights[i].shape[1] = temp->DENSE->weights->shape[1];
    temp = temp->next_layer;
  }
  temp = m->graph->next_layer;
  for(int i=0;i<m->number_of_layers-1;i++){
    for(int j=0;j<temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1];j++){
      fscanf(fp,"%lf ",&biases[i].matrix[j]);
    }
    biases[i].shape[0] = temp->DENSE->bias->shape[0];
    biases[i].shape[1] = temp->DENSE->bias->shape[1];
    temp = temp->next_layer;
  }
  fclose(fp);
  temp = m->graph->next_layer;
  int index = 0;
  while(temp!=NULL){
    free2d(temp->DENSE->weights);
    free2d(temp->DENSE->bias);
    
    temp->DENSE->weights = NULL;
    temp->DENSE->bias = NULL;

    temp->DENSE->weights = (dARRAY*)malloc(sizeof(dARRAY));
    temp->DENSE->weights->matrix = (double*)calloc(weights[index].shape[0]*weights[index].shape[1],sizeof(double));

    temp->DENSE->bias = (dARRAY*)malloc(sizeof(dARRAY));
    temp->DENSE->bias->matrix = (double*)calloc(biases[index].shape[0]*biases[index].shape[1],sizeof(double));
    
    temp->DENSE->weights->matrix = weights[index].matrix;

    temp->DENSE->weights->shape[0] = weights[index].shape[0];
    temp->DENSE->weights->shape[1] = weights[index].shape[1];

    temp->DENSE->bias->matrix = biases[index].matrix;

    temp->DENSE->bias->shape[0] = biases[index].shape[0];
    temp->DENSE->bias->shape[1] = biases[index].shape[1];
    index++;
    temp = temp->next_layer;
  }
  temp = NULL;
}

void __save_model__(char * filename){
  if(strstr(filename,".t7")==NULL){
    printf("\033[1;31mFileExtension Error : \033[93m Please use \".t7\" extension only. Other extensions are not supported currently.\033[0m\n");
    exit(EXIT_FAILURE);
  }
  char destpath[1024];
  snprintf (destpath, sizeof(destpath), "./model/%s", filename);
  FILE * fp = NULL;
  if( access(destpath,F_OK)==0) {
    if(remove(destpath)==0){}
  } 
  fp = fopen(destpath,"ab+");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  Computation_Graph * temp = m->graph->next_layer;
  for(int i=0;i<m->number_of_layers-1;i++){
    for(int j=0;j<temp->DENSE->weights->shape[0]*temp->DENSE->weights->shape[1];j++)
      fprintf(fp,"%lf ",temp->DENSE->weights->matrix[j]);
    temp = temp->next_layer;
  }
  temp = m->graph->next_layer;
  for(int i=0;i<m->number_of_layers-1;i++){
    for(int j=0;j<temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1];j++)
      fprintf(fp,"%lf ",temp->DENSE->bias->matrix[j]);
    temp = temp->next_layer;
  }
  fclose(fp);
  fp = NULL;
}

void load_x_train(int * dims){
  FILE * fp = NULL;
  fp = fopen("X_train.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * x_train = (dARRAY*)malloc(sizeof(dARRAY));
  x_train->matrix = (double*)calloc(dims[0]*dims[1],sizeof(double));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%lf ",&x_train->matrix[j]);
  }
  x_train->shape[0] = dims[1];
  x_train->shape[1] = dims[0];

  m->x_train = transpose(x_train);
  free2d(x_train);
  x_train = NULL;
  fclose(fp);
}

void load_y_train(int * dims){
  FILE * fp = NULL;
  fp = fopen("y_train.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * Y_train = (dARRAY*)malloc(sizeof(dARRAY));
  Y_train->matrix = (double*)calloc(dims[0]*dims[1],sizeof(double));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%lf ",&Y_train->matrix[j]);
  }
  Y_train->shape[0] = dims[1];
  Y_train->shape[1] = dims[0];

  m->Y_train = transpose(Y_train);
  free2d(Y_train);
  Y_train = NULL;
  fclose(fp);
}

void load_x_cv(int * dims){
  FILE * fp = NULL;
  fp = fopen("X_cv.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * x_cv = (dARRAY*)malloc(sizeof(dARRAY));
  x_cv->matrix = (double*)calloc(dims[0]*dims[1],sizeof(double));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%lf ",&x_cv->matrix[j]);
  }
  x_cv->shape[0] = dims[1];
  x_cv->shape[1] = dims[0];

  m->x_cv = transpose(x_cv);
  free2d(x_cv);
  x_cv = NULL;
  fclose(fp);
}

void load_y_cv(int * dims){
  FILE * fp = NULL;
  fp = fopen("y_cv.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * Y_cv = (dARRAY*)malloc(sizeof(dARRAY));
  Y_cv->matrix = (double*)calloc(dims[0]*dims[1],sizeof(double));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%lf ",&Y_cv->matrix[j]);
  }
  Y_cv->shape[0] = dims[1];
  Y_cv->shape[1] = dims[0];

  m->Y_cv = transpose(Y_cv);
  free2d(Y_cv);
  Y_cv = NULL;
  fclose(fp);
}

void __summary__(){}

long int get_total_params(){
  long int total_params = 0;
  Computation_Graph * temp = m->graph->next_layer;
  while(temp!=NULL){
    if(temp->type==DENSE){
      total_params+= temp->DENSE->weights->shape[0]*temp->DENSE->weights->shape[1];
      total_params+= temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1];
    }
    temp = temp->next_layer;
  }
  return total_params;
}

void (create_model)(){
  m = NULL;
  m = (__Model__*)malloc(sizeof(__Model__));
  m->graph = NULL;
  m->graph = NULL;
  m->current_layer = NULL;
  m->number_of_layers = 0;
  m->input_size = 0;
  m->output_size = 0;
}

void (destroy_model)(){

}

void (Model)(Model_args model_args){
  m->x_train = model_args.x_train;
  m->x_test = model_args.x_test;
  m->x_cv = model_args.x_cv;

  m->Y_train = model_args.Y_train;
  m->Y_test = model_args.Y_test;
  m->Y_cv = model_args.Y_cv;

  //initialize regualrization hyperparameters
  m->loss = model_args.loss;
  m->lambda = model_args.lambda;
  m->regularization = model_args.regularization;

  if(!strcasecmp(m->loss,"cross_entropy_loss")) cross_entropy_loss();

  //initialize hyperparameters for various optimizers
  m->optimizer = model_args.optimizer; // Optimizer choice
  m->learning_rate = model_args.learning_rate; // hyperparameter for step size for optimization algorithm
  m->time_step = 0; // timestep - required for Adam update, for bias correction
  m->beta = model_args.beta; //hyperparameter - used for RMSProp, Denotes Decay_rate for weighted average
  m->beta1 = model_args.beta1; // hyperparameter - used for Adam and RMSProp, Decay rate for estimation of first moment
  m->beta2 = model_args.beta2; // hyperparameter - used for Adam, Decay rate used for estimation of second moment.
  m->epsilon = 1e-8; // small value to prevent divison by zero during parameter updates
  for(int i=0;i<m->number_of_layers;i++){
    m->m_t_dW[i] = NULL;  // for Adam and RMSProp - to calculate first moment
    m->v_t_dW[i] = NULL; // for Adam - to calculate second moment
    m->m_t_db[i] = NULL;
    m->v_t_db[i] = NULL;
    m->cache_dW[i] = NULL; // for AdaGrad
    m->cache_db[i] = NULL;
  }
  m->mini_batch_size = model_args.mini_batch_size;
  m->num_iter = model_args.num_iter;

  m->input_size = model_args.x_train->shape[0];
  m->output_size = model_args.Y_train->shape[0];

  if(m->input_size!=m->graph->INPUT->input_features_size){
    printf("\033[1;31mModel Error : \033[93m Size of Input Layer does not match the size of x_train.\033[0m\n");
    printf("\033[96mHint : \033[93mCheck if layer_size of input layer == x_train->shape[0]\033[0m\n");
    exit(EXIT_FAILURE);
  }

  m->num_of_training_examples = model_args.x_train->shape[1];

  m->print_cost = model_args.print_cost;
  m->init = __init__;
  m->forward = __forward__;
  m->backward = __backward__;
  m->load_model = __load_model__;
  m->save_model = __save_model__;
  m->fit = __fit__;
  m->predict = __predict__;
  m->summary = __summary__;
  m->init();
  m->total_parameters = get_total_params();
  printf("Total Trainable Parameters : %ld\n",m->total_parameters);
}