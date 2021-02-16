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
    if(temp->type==INPUT) temp->INPUT->forward();
    else if(temp->type==DENSE) temp->DENSE->forward();
    else if(temp->type==LOSS) temp->LOSS->forward();
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

float calculate_accuracy(dARRAY * predicted, dARRAY * gnd_truth){
  int success = 0;
  dARRAY * temp = (dARRAY*)malloc(sizeof(dARRAY));
  temp->matrix = (float*)calloc(predicted->shape[0]*predicted->shape[1],sizeof(float));
  for(int i = 0;i<predicted->shape[0]*predicted->shape[1];i++){
    temp->matrix[i] = predicted->matrix[i]<0.5 ? 0 : 1;
  }
  temp->shape[0] = predicted->shape[0];
  temp->shape[1] = predicted->shape[1];

  for(int i = 0;i<predicted->shape[0]*predicted->shape[1];i++){
    if(temp->matrix[i]==gnd_truth->matrix[i]) success++;
  }
  if(predicted->shape[0]!=1) success = success/predicted->shape[0];
  free2d(temp);
  temp = NULL;
  return success/(float)predicted->shape[1];
}

dARRAY * relu_val(dARRAY * linear_matrix){
  dARRAY * relu_outf = NULL;
  relu_outf = (dARRAY*)malloc(sizeof(dARRAY));
  relu_outf->matrix = (float*)calloc(linear_matrix->shape[0]*linear_matrix->shape[1],sizeof(float));
  #pragma omp parallel for num_threads(8) shared(relu_outf,linear_matrix)
  for(int i=0;i<linear_matrix->shape[0]*linear_matrix->shape[1];i++)
    relu_outf->matrix[i] = linear_matrix->matrix[i]>(float)0.0 ?(float)linear_matrix->matrix[i] : (float)0.0;
  relu_outf->shape[0] = linear_matrix->shape[0];
  relu_outf->shape[1] = linear_matrix->shape[1];
  return relu_outf;
}

dARRAY * sigmoid_val(dARRAY * linear_matrix){
  dARRAY * sigmoid_outf = NULL;
  sigmoid_outf = (dARRAY*)malloc(sizeof(dARRAY));
  sigmoid_outf->matrix = (float*)calloc(linear_matrix->shape[0]*linear_matrix->shape[1],sizeof(float));
  #pragma omp parallel for num_threads(8) shared(sigmoid_outf,linear_matrix)
  for(int i=0;i<linear_matrix->shape[0]*linear_matrix->shape[1];i++)
    sigmoid_outf->matrix[i] = (float)(1.0/(float)(1+exp((float)(-1.0*linear_matrix->matrix[i]))));
  sigmoid_outf->shape[0] = linear_matrix->shape[0];
  sigmoid_outf->shape[1] = linear_matrix->shape[1];
  return sigmoid_outf;
}

dARRAY * tanh_val(dARRAY * linear_matrix){
  dARRAY * tanh_out = (dARRAY*)malloc(sizeof(dARRAY));
  tanh_out->matrix = (float*)calloc(linear_matrix->shape[0]*linear_matrix->shape[1],sizeof(float));
  #pragma omp parallel for num_threads(8) shared(tanh_out,linear_matrix)
  for(int i=0;i<linear_matrix->shape[0]*linear_matrix->shape[1];i++){
    float exp_res1 = exp(linear_matrix->matrix[i]);
    float exp_res2 = exp(-1*linear_matrix->matrix[i]);
    tanh_out->matrix[i] = (exp_res1 - exp_res2)/(exp_res1 + exp_res2);
  }
  tanh_out->shape[0] = linear_matrix->shape[0];
  tanh_out->shape[1] = linear_matrix->shape[1];
  return tanh_out;
}

dARRAY * softmax_val(dARRAY * linear_matrix){
  dARRAY * softmax_outf = NULL;
  dARRAY * exp_sub_max = exponentional(linear_matrix);

  dARRAY * div_factor = (dARRAY*)malloc(sizeof(dARRAY));
  div_factor->matrix = (float*)calloc(exp_sub_max->shape[1],sizeof(float));
  
  dARRAY * temp = transpose(exp_sub_max);
  for(int i=0;i<temp->shape[0];i++){
    float sum_of_exps=0.0;
    for(int j=0;j<temp->shape[1];j++){
      sum_of_exps+= temp->matrix[i*temp->shape[1]+j];
    }
    div_factor->matrix[i] = sum_of_exps;
  }
  div_factor->shape[0] = 1;
  div_factor->shape[1] = exp_sub_max->shape[1];

  softmax_outf = divison(exp_sub_max,div_factor);

  free2d(exp_sub_max);
  free2d(div_factor);
  free2d(temp);
  temp = exp_sub_max = div_factor = NULL; 

  return softmax_outf;
}

dARRAY * calculate_val_test_acc(dARRAY * input_features,dARRAY * gnd_truth){
  dARRAY * weight_input_res = NULL;
  dARRAY * output = NULL;
  dARRAY * activation_temp = NULL;

  Computation_Graph * temp = m->graph->next_layer;
  int layer=0;
  while(temp!=NULL){
    if(temp->prev_layer->type==INPUT){ 
    weight_input_res = dot(temp->DENSE->weights, input_features); 
    }
    else{ 
      weight_input_res = dot(temp->DENSE->weights, activation_temp);
    }
    if(activation_temp!=NULL){
       free2d(activation_temp);
       activation_temp = NULL;
    }
    dARRAY * Z = add(weight_input_res,temp->DENSE->bias);
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
    else if(!strcasecmp(temp->DENSE->activation,"softmax")){
      activation_temp = softmax_val(Z);
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
  if(m->predicting) return output;
  else{
    float acc = calculate_accuracy(output,gnd_truth);
    free2d(output);
    output = NULL;

    dARRAY * return_val = (dARRAY*)malloc(sizeof(dARRAY));
    return_val->matrix = &acc;

    return return_val;
  }
}

void append_to_file(float * arr ,char * filename,char * mode){
  FILE * fp = NULL;
  fp = fopen(filename,mode);
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  for(int i=0;i<m->num_iter;i++)
    fprintf(fp,"%f ",arr[i]);
  fclose(fp);
}

void __fit__(){
  int i = 1;
  int iterations=0;
  float sum_cost = 0.0;
  float sum_train_acc = 0.0;
  float sum_train_val_acc = 0.0;
  float * train_cost_arr = (float*)calloc(m->num_iter==-1?1000000:m->num_iter,sizeof(float));
  float * train_acc_arr = (float*)calloc(m->num_iter==-1?1000000:m->num_iter,sizeof(float));
  float * val_acc_arr = (float*)calloc(m->num_iter==-1?1000000:m->num_iter,sizeof(float));
  if(m->num_iter==-1){
    iterations = i;
  }
  else iterations = m->num_iter;
  while(i<=iterations){
    __forward__();
    sum_cost += m->iter_cost;
    sum_train_acc += calculate_accuracy(m->output,m->Y_train);
    // dARRAY * temp = calculate_val_test_acc(m->x_cv,m->Y_cv);
    // sum_train_val_acc += *temp->matrix;
    if(m->print_cost){
      m->train_cost = sum_cost/(float)i;
      m->train_accuracy = sum_train_acc/(float)i;
      m->cross_val_accuracy = sum_train_val_acc/(float)i;
      train_cost_arr[i-1] = m->train_cost;
      train_acc_arr[i-1] = m->train_accuracy;
      val_acc_arr[i-1] = m->cross_val_accuracy;
      printf("\033[96m%d. Cost : \033[0m%f ",i,m->train_cost);
      printf("\033[96m train_acc : \033[0m%f ",m->train_accuracy);
      printf("\033[96m val_acc : \033[0m%f\n",m->cross_val_accuracy);
    }
    __backward__();
    if(!strcasecmp(m->optimizer,"adam")){
      m->time_step += 1;
      adam();
    }
    else if(!strcasecmp(m->optimizer,"rmsprop")){
      RMSProp();
    }
    else if(!strcasecmp(m->optimizer,"adagrad")){
      adagrad();
    }
    else if(!strcasecmp(m->optimizer,"sgd")){
      SGD();
    }
    if(m->ckpt_every!=-1 && (i%m->ckpt_every==0 || (i==m->num_iter && m->num_iter!=-1))){
      time_t rawtime;
      struct tm * timeinfo;

      time ( &rawtime );
      timeinfo = localtime ( &rawtime );
  
      char buffer[1024];
      snprintf(buffer,sizeof(buffer),"model_%d_%s.t7",i,asctime (timeinfo));
      __save_model__(buffer);
    }
    i++;
    if(m->num_iter==-1) iterations = i;
    // free(temp);
    // temp = NULL;
  }
  append_to_file(train_cost_arr,"./bin/cost.data","ab+");
  append_to_file(train_acc_arr,"./bin/train_acc.data","ab+");
  append_to_file(val_acc_arr,"./bin/val_acc.data","ab+");
}

void __predict__(dARRAY * input_feature){
  m->graph->INPUT->A = input_feature;
  m->predicting = 1;
  dARRAY * prediction = calculate_val_test_acc(input_feature,NULL);
  printf("Score : [%f,%f]\n",prediction->matrix[0],prediction->matrix[1]);
  if(prediction->matrix[0]>=0.5 && prediction->matrix[1]<0.5) printf("CAT\n");
  else printf("DOG\n");
  m->predicting=0;
}

void __test__(){
  dARRAY * acc = calculate_val_test_acc(m->x_test,m->Y_test);
  printf("\033[96mTest Accuracy : \033[0m%f\n",acc->matrix[0]);
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
    weights[i].matrix = (float*)calloc(temp->DENSE->weights->shape[0]*temp->DENSE->weights->shape[1],sizeof(float));
    biases[i].matrix = (float*)calloc(temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1],sizeof(float));
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
      fscanf(fp,"%f ",&weights[i].matrix[j]);
    }
    weights[i].shape[0] = temp->DENSE->weights->shape[0];
    weights[i].shape[1] = temp->DENSE->weights->shape[1];
    temp = temp->next_layer;
  }
  temp = m->graph->next_layer;
  for(int i=0;i<m->number_of_layers-1;i++){
    for(int j=0;j<temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1];j++){
      fscanf(fp,"%f ",&biases[i].matrix[j]);
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
    temp->DENSE->weights->matrix = (float*)calloc(weights[index].shape[0]*weights[index].shape[1],sizeof(float));

    temp->DENSE->bias = (dARRAY*)malloc(sizeof(dARRAY));
    temp->DENSE->bias->matrix = (float*)calloc(biases[index].shape[0]*biases[index].shape[1],sizeof(float));
    
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
      fprintf(fp,"%f ",temp->DENSE->weights->matrix[j]);
    temp = temp->next_layer;
  }
  temp = m->graph->next_layer;
  for(int i=0;i<m->number_of_layers-1;i++){
    for(int j=0;j<temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1];j++)
      fprintf(fp,"%f ",temp->DENSE->bias->matrix[j]);
    temp = temp->next_layer;
  }
  fclose(fp);
  fp = NULL;
}

dARRAY * load_x_train(int * dims){
  FILE * fp = NULL;
  fp = fopen("./data/X_train.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * x_train = (dARRAY*)malloc(sizeof(dARRAY));
  x_train->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&x_train->matrix[j]);
  }
  x_train->shape[0] = dims[1];
  x_train->shape[1] = dims[0];

  dARRAY * X_train = transpose(x_train);
  free2d(x_train);
  x_train = NULL;
  fclose(fp);
  return X_train;
}

dARRAY * load_y_train(int * dims){
  FILE * fp = NULL;
  fp = fopen("./data/y_train.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * Y_train = (dARRAY*)malloc(sizeof(dARRAY));
  Y_train->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&Y_train->matrix[j]);
  }
  Y_train->shape[0] = dims[1];
  Y_train->shape[1] = dims[0];

  dARRAY * y_train = transpose(Y_train);
  free2d(Y_train);
  Y_train = NULL;
  fclose(fp);
  return y_train;
}

dARRAY * load_x_cv(int * dims){
  FILE * fp = NULL;
  fp = fopen("./data/X_cv.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * x_cv = (dARRAY*)malloc(sizeof(dARRAY));
  x_cv->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&x_cv->matrix[j]);
  }
  x_cv->shape[0] = dims[1];
  x_cv->shape[1] = dims[0];

  dARRAY * X_CV = transpose(x_cv);
  free2d(x_cv);
  x_cv = NULL;
  fclose(fp);
  return X_CV;
}

dARRAY * load_y_cv(int * dims){
  FILE * fp = NULL;
  fp = fopen("./data/y_cv.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * Y_cv = (dARRAY*)malloc(sizeof(dARRAY));
  Y_cv->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&Y_cv->matrix[j]);
  }
  Y_cv->shape[0] = dims[1];
  Y_cv->shape[1] = dims[0];

  dARRAY * y_cv = transpose(Y_cv);
  free2d(Y_cv);
  Y_cv = NULL;
  fclose(fp);
  return y_cv;
}

dARRAY * load_x_test(int * dims){
  FILE * fp = NULL;
  fp = fopen("./data/X_test.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * x_test_temp = (dARRAY*)malloc(sizeof(dARRAY));
  x_test_temp->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&x_test_temp->matrix[j]);
  }
  x_test_temp->shape[0] = dims[1];
  x_test_temp->shape[1] = dims[0];
  dARRAY * x_test = transpose(x_test_temp);
  free2d(x_test_temp);
  x_test_temp = NULL;
  fclose(fp);
  return x_test;
}

dARRAY * load_y_test(int * dims){
  FILE * fp = NULL;
  fp = fopen("./data/y_test.data","rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * Y_test = (dARRAY*)malloc(sizeof(dARRAY));
  Y_test->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&Y_test->matrix[j]);
  }
  Y_test->shape[0] = dims[1];
  Y_test->shape[1] = dims[0];

  dARRAY * y_test = transpose(Y_test);
  free2d(Y_test);
  Y_test = NULL;
  fclose(fp);
  return y_test;
}

dARRAY * load_test_image(char * filename){
  FILE * fp = NULL;
  fp = fopen(filename,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  dARRAY * image = (dARRAY*)malloc(sizeof(dARRAY));
  image->matrix = (float*)calloc(12288*1,sizeof(float));
  for(int j=0;j<12288;j++){
    fscanf(fp,"%f ",&image->matrix[j]);
  }
  image->shape[0] = 12288;
  image->shape[1] = 1;
  fclose(fp);
  fp=NULL;
  return image;
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
  destroy_Graph(m->graph);
  free2d(m->x_train);
  free2d(m->Y_train);
  free2d(m->x_cv);
  free2d(m->Y_cv);
  free2d(m->output);
  m = NULL;
}

void (Model)(Model_args model_args){
  m->x_train = model_args.x_train;
  m->x_test = model_args.x_test;
  m->x_cv = model_args.x_cv;

  m->Y_train = model_args.Y_train;
  m->Y_test = model_args.Y_test;
  m->Y_cv = model_args.Y_cv;

  m->ckpt_every = model_args.checkpoint_every;

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
  m->test = __test__;
  m->init();
  m->total_parameters = get_total_params();
  printf("Total Trainable Parameters : %ld\n",m->total_parameters);
}