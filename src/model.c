#include <cdnn/model.h>
#include <cdnn/progressbar.h>

__Model__ * m;

void __init__(){
  m->predicting = 0;
  m->testing=0;
  m->test_accuracy = 0.0;
  m->train_accuracy = 0.0;
  m->train_cost = 0.0;
  m->cross_val_accuracy = 0.0;
  m->iter_cost = 0.0;
  m->current_iter = 1;
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
  if(!strcasecmp(m->optimizer,"rmsprop")) flag = 3;
  if(!strcasecmp(m->optimizer,"momentum")) flag = 4;
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
    else if(flag==3 && temp->type!=LOSS){
      int dims_dW[] = {temp->DENSE->weights->shape[0],temp->DENSE->weights->shape[1]};
      int dims_db[] = {temp->DENSE->bias->shape[0],temp->DENSE->bias->shape[1]};
      m->v_t_dW[index] = zeros(dims_dW);
      m->v_t_db[index] = zeros(dims_db);
      index++;
    }
    else if(flag==4 && temp->type!=LOSS){
      int dims_dW[] = {temp->DENSE->weights->shape[0],temp->DENSE->weights->shape[1]};
      int dims_db[] = {temp->DENSE->bias->shape[0],temp->DENSE->bias->shape[1]};
      m->m_t_dW[index] = zeros(dims_dW);
      m->m_t_db[index] = zeros(dims_db);
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

  dARRAY * div_factor = sum(exp_sub_max,0);

  softmax_outf = divison(exp_sub_max,div_factor);

  free2d(exp_sub_max);
  free2d(div_factor);
  exp_sub_max = div_factor = NULL; 

  return softmax_outf;
}

dARRAY * calculate_val_test_acc(dARRAY * input_features,dARRAY * gnd_truth){
  dARRAY * weight_input_res = NULL;
  dARRAY * output = NULL;
  dARRAY * activation_temp = NULL;

  Computation_Graph * temp = m->graph->next_layer;
  int layer=0;
  while(temp->type!=LOSS){
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
    }
    layer++;
    temp = temp->next_layer;
    Z = NULL;
  }
  if(m->predicting) return output;
  else{
    float acc = calculate_accuracy(output,gnd_truth);
    if(m->testing)
      printf("Accuracy : %f\n",acc);
    free2d(output);
    output = NULL;

    dARRAY * return_val = (dARRAY*)malloc(sizeof(dARRAY));
    return_val->matrix = (float*)calloc(1,sizeof(float));
    return_val->matrix[0] = acc;
    return_val->shape[0] = 1;
    return_val->shape[1] = 1;
    if(m->testing)
      printf("Accuracy : %f\n",return_val->matrix[0]);
    return return_val;
  }
}

void dump_to_file(float * arr ,char * filename,char * mode){
  struct stat st = {0};
  if (stat("./bin/", &st) == -1) {
    mkdir("./bin/", 0700);
  }
  FILE * fp = NULL;
  fp = fopen(filename,mode);
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93mCould not open %s file!\033[0m\n",filename);
    exit(EXIT_FAILURE);
  }
  for(int i=0;i<m->num_iter*m->num_mini_batches;i++)
    fprintf(fp,"%f ",arr[i]);
  fclose(fp);
}

void __fit__(){
  int i = 1;
  int index = 0;
  int iterations=0;
  int flag_cost = 0;
  float sum_cost = m->train_cost*m->current_iter;
  float sum_train_acc = m->train_accuracy*m->current_iter;
  float sum_train_val_acc = m->cross_val_accuracy*m->current_iter;
  float * train_cost_arr = (float*)calloc(m->num_iter==-1?1000000:m->num_iter*m->num_mini_batches,sizeof(float));
  float * train_acc_arr = (float*)calloc(m->num_iter==-1?1000000:m->num_iter*m->num_mini_batches,sizeof(float));
  float * val_acc_arr = NULL;

  if(m->Y_cv!=NULL && m->x_cv!=NULL){
    val_acc_arr = (float*)calloc(m->num_iter==-1?1000000:m->num_iter*m->num_mini_batches,sizeof(float));
  }
  if(m->num_iter==-1){
    iterations = i;
  }
  else iterations = m->num_iter;
  signal(SIGINT,early_stopping_handler);
  while(i<=iterations){
    flag_cost=0;
    int * shuffle = permutation(m->num_mini_batches);
    for(int mini_batch=0;mini_batch<m->num_mini_batches;mini_batch++){
      m->current_mini_batch = shuffle[mini_batch];
      __forward__();
      sum_cost += m->iter_cost;
      sum_train_acc += calculate_accuracy(m->output,m->y_train_mini_batch[m->current_mini_batch]);
      dARRAY * temp = NULL;
      if(m->Y_cv!=NULL && m->x_cv!=NULL){
        temp = calculate_val_test_acc(m->x_cv,m->Y_cv);
        sum_train_val_acc += temp->matrix[0];
      }
      if(m->print_cost){
        m->train_cost = i==m->current_iter ? sum_cost/(float)i : sum_cost/(float)m->current_iter;
        m->train_accuracy = i==m->current_iter ? sum_train_acc/(float)i : sum_train_acc/(float)m->current_iter;
        train_cost_arr[index] = m->train_cost;
        train_acc_arr[index] = m->train_accuracy;
        if(m->Y_cv!=NULL && m->x_cv!=NULL){
          m->cross_val_accuracy = i==m->current_iter ? sum_train_val_acc/(float)i : sum_train_val_acc/(float)m->current_iter;
          val_acc_arr[index] = m->cross_val_accuracy;
        }
        printf("\033[96m%d. %d. Cost : \033[0m%f ",i,mini_batch,m->train_cost);
        printf("\033[96m train_acc : \033[0m%f ",m->train_accuracy);
        if(m->Y_cv!=NULL && m->x_cv!=NULL){
          printf("\033[96m val_acc : \033[0m%f",m->cross_val_accuracy);
        }
        printf("\n");
        if(train_cost_arr[index]>=2.5*train_cost_arr[0]){
          printf("\033[96mCost is exploding hence exiting. Try reducing your learning rate and try again!\033[0m\n");
          exit(EXIT_FAILURE);
        }
        index++;
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
      else if(!strcasecmp(m->optimizer,"momentum")){
        Momentum();
      }
      else{
        printf("Optimizer selected is not available\n");
        exit(EXIT_FAILURE);
      }
      m->current_iter += 1;
      free(temp);
      temp = NULL;
    }
    if(m->ckpt_every!=-1 && (i%m->ckpt_every==0 || (i==m->num_iter && m->num_iter!=-1))){
      time_t rawtime;
      struct tm * timeinfo;

      time ( &rawtime );
      timeinfo = localtime ( &rawtime );
  
      char buffer[1024];
      snprintf(buffer,sizeof(buffer),"model_%d_%s.t7",i,asctime (timeinfo));
      __save_model__(buffer);
      if(flag_cost==0){
        dump_to_file(train_cost_arr,"./bin/cost.data","ab+");
        dump_to_file(train_acc_arr,"./bin/train_acc.data","ab+");
        if(m->x_cv!=NULL || m->Y_cv!=NULL)
          dump_to_file(val_acc_arr,"./bin/val_acc.data","wb+");
        flag_cost=1;
      }
    }
    if(!flag_cost){
      dump_to_file(train_cost_arr,"./bin/cost.data","ab+");
      dump_to_file(train_acc_arr,"./bin/train_acc.data","ab+");
      if(m->x_cv!=NULL || m->Y_cv!=NULL)
        dump_to_file(val_acc_arr,"./bin/val_acc.data","wb+");
    }
    i++;
    if(m->num_iter==-1) iterations = i;
    free(shuffle);
    shuffle=NULL;
  }
  if(flag_cost==0){
    dump_to_file(train_cost_arr,"./bin/cost.data","wb+");
    dump_to_file(train_acc_arr,"./bin/train_acc.data","wb+");
    if(m->x_cv!=NULL || m->Y_cv!=NULL)
      dump_to_file(val_acc_arr,"./bin/val_acc.data","wb+");
    flag_cost=1;
  }
}

/**
 * Function that is used to initiate model training. 
*/
void Fit(){
  __fit__();
}

void early_stopping_handler(){
  printf("\nModel Saved!\n");
  time_t rawtime;
  struct tm * timeinfo;

  time ( &rawtime );
  timeinfo = localtime ( &rawtime );

  char buffer[1024];
  snprintf(buffer,sizeof(buffer),"model_%s.t7",asctime (timeinfo));
  __save_model__(buffer);
  Destroy_Model();
  exit(EXIT_SUCCESS);  
}

dARRAY * __predict__(dARRAY * input_feature,int verbose){
  m->graph->INPUT->A = input_feature;
  m->predicting = 1;
  dARRAY * prediction = calculate_val_test_acc(input_feature,NULL);
  dARRAY * prediction_trans = transpose(prediction);
  if(verbose){
    printf("[");
    for(int i = 0;i<prediction_trans->shape[0];i++){
      for(int j=0;j<prediction_trans->shape[1];j++){
        if(j<prediction_trans->shape[1]-1)
        printf("%f, ",prediction_trans->matrix[i*prediction_trans->shape[1]+j]);
        else
        printf("%f",prediction_trans->matrix[i*prediction_trans->shape[1]+j]);
      }
    }
    printf("]\n");
  }
  free2d(prediction);
  m->predicting=0;
  return prediction_trans;
}

/**!
 * Function that is used to test your model on arbitrary input_features. 
 * @param input_feature Features to be passed to your model. 
 * @param verbose Specifies whether or not to print the classification scores (verbose=1 or verbose=0)
 * @return Returns a dARRAY pointing to the output values of the model.
*/
dARRAY * Predict(dARRAY * input_feature,int verbose){
  return __predict__(input_feature,verbose);
}

void __test__(){
  if(m->x_test==NULL || m->Y_test==NULL){
    printf("\033[1;31mDataset Error : \033[93mTest set has not been provided.\033[0m\n");
    return;
  }
  dARRAY * acc = calculate_val_test_acc(m->x_test,m->Y_test);
  printf("\033[96mTest Accuracy : \033[0m%f\n",acc->matrix[0]);
  free2d(acc);
  acc = NULL;
}

/**
 * Function that is used to test your model against your test set. 
*/
void Test(){
  __test__();
}

void __load_model__(char * filename){
  if(strstr(filename,".t7")==NULL){
    printf("\033[1;31mFileExtension Error : \033[93mPlease use \".t7\" extension only. Other extensions are not supported currently.\033[0m\n");
    exit(EXIT_FAILURE);
  }
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
  fp = fopen(filename,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93mCould not open %s file!\033[0m\n",filename);
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
  fscanf(fp,"%f ",&m->train_cost);
  fscanf(fp,"%f ",&m->train_accuracy);
  fscanf(fp,"%f ",&m->cross_val_accuracy);
  fscanf(fp,"%d ",&m->current_iter);
  fclose(fp);
  temp = m->graph->next_layer;
  int index = 0;
  while(temp!=NULL){
    if(temp->DENSE->weights!=NULL)
      free2d(temp->DENSE->weights);
    if(temp->DENSE->bias!=NULL)
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

/**!
 * Function that is used to load a model. 
 * @param filename Name of the file where the model resides. 
*/
void Load_Model(char * filename){
  __load_model__(filename);
}

void __save_model__(char * filename){
  if(strstr(filename,".t7")==NULL){
    printf("\033[1;31mFileExtension Error : \033[93m%s uses a different extension. Please use \".t7\" extension only. Other extensions are not supported currently.\033[0m\n",filename);
    exit(EXIT_FAILURE);
  }
  FILE * fp = NULL;
  if( access(filename,F_OK)==0) {
    if(remove(filename)==0){}
  } 
  fp = fopen(filename,"wb+");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93mCould not open %s file!\033[0m\n",filename);
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
  fprintf(fp,"%f ",m->train_cost);
  fprintf(fp,"%f ",m->train_accuracy);
  fprintf(fp,"%f ",m->cross_val_accuracy);
  fprintf(fp,"%d ",m->current_iter-1);
  fclose(fp);
  fp = NULL;
}

/**!
 * Function that is used to save a model. 
 * @param filename Name of the file where the model will be saved. 
*/
void Save_Model(char * filename){
  __save_model__(filename);
}

/**!
 * Function that is used to load X_train from file. 
 * @param filename Name of the file where the X_train resides. 
 * @param dims Dimensions of your training set (Features,Number of examples)
*/
dARRAY * load_x_train(char * filename,int * dims){
  FILE * fp = NULL;
  fp = fopen(filename,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93mCould not open %s file!\033[0m\n",filename);
    exit(EXIT_FAILURE);
  }
  dARRAY * x_train = (dARRAY*)malloc(sizeof(dARRAY));
  x_train->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  progressbar * progress = progressbar_new("\033[1;32mLoading X_train :\033[0m",dims[1]);
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&x_train->matrix[j]);
    if(j%dims[0]==0)
      progressbar_inc(progress);
  }
  progressbar_finish(progress);
  x_train->shape[0] = dims[1];
  x_train->shape[1] = dims[0];
  
  
  dARRAY * X_train = transpose(x_train);
  free2d(x_train);
  x_train = NULL;
  fclose(fp);
  return X_train;
}

/**!
 * Function that is used to load y_train from file. 
 * @param filename Name of the file where the y_train resides. 
 * @param dims Dimensions of your training set (Number of classes,Number of examples)
*/
dARRAY * load_y_train(char * filename,int * dims){
  FILE * fp = NULL;
  fp = fopen(filename,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93mCould not open %s file!\033[0m\n",filename);
    exit(EXIT_FAILURE);
  }
  dARRAY * Y_train = (dARRAY*)malloc(sizeof(dARRAY));
  Y_train->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  progressbar * progress = progressbar_new("\033[1;32mLoading y_train :\033[0m",dims[1]);
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&Y_train->matrix[j]);
    if(j%dims[0]==0)
      progressbar_inc(progress);
  }
  progressbar_finish(progress);
  Y_train->shape[0] = dims[1];
  Y_train->shape[1] = dims[0];

  dARRAY * y_train = transpose(Y_train);
  free2d(Y_train);
  Y_train = NULL;
  fclose(fp);
  return y_train;
}

/**!
 * Function that is used to load X_cv from file. 
 * @param filename Name of the file where the X_cv resides. 
 * @param dims Dimensions of your validation set (Features,Number of examples)
*/
dARRAY * load_x_cv(char * filename,int * dims){
  FILE * fp = NULL;
  fp = fopen(filename,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93mCould not open %s file!\033[0m\n",filename);
    exit(EXIT_FAILURE);
  }
  dARRAY * x_cv = (dARRAY*)malloc(sizeof(dARRAY));
  x_cv->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  progressbar * progress = progressbar_new("\033[1;32mLoading X_cv :\033[0m",dims[1]);
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&x_cv->matrix[j]);
    if(j%dims[0]==0)
      progressbar_inc(progress);
  }
  progressbar_finish(progress);
  x_cv->shape[0] = dims[1];
  x_cv->shape[1] = dims[0];

  dARRAY * X_CV = transpose(x_cv);
  free2d(x_cv);
  x_cv = NULL;
  fclose(fp);
  return X_CV;
}

/**!
 * Function that is used to load y_cv from file. 
 * @param filename Name of the file where the y_cv resides. 
 * @param dims Dimensions of your validation set (Number of classes,Number of examples)
*/
dARRAY * load_y_cv(char * filename,int * dims){
  FILE * fp = NULL;
  fp = fopen(filename,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93mCould not open %s file!\033[0m\n",filename);
    exit(EXIT_FAILURE);
  }
  dARRAY * Y_cv = (dARRAY*)malloc(sizeof(dARRAY));
  Y_cv->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  progressbar * progress = progressbar_new("\033[1;32mLoading y_cv :\033[0m",dims[1]);
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&Y_cv->matrix[j]);
    if(j%dims[0]==0)
      progressbar_inc(progress);
  }
  progressbar_finish(progress);
  Y_cv->shape[0] = dims[1];
  Y_cv->shape[1] = dims[0];

  dARRAY * y_cv = transpose(Y_cv);
  free2d(Y_cv);
  Y_cv = NULL;
  fclose(fp);
  return y_cv;
}

/**!
 * Function that is used to load X_test from file. 
 * @param filename Name of the file where the X_test resides. 
 * @param dims Dimensions of your test set (Features,Number of examples)
*/
dARRAY * load_x_test(char * filename,int * dims){
  FILE * fp = NULL;
  fp = fopen(filename,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open %s file!\033[0m\n",filename);
    exit(EXIT_FAILURE);
  }
  dARRAY * x_test_temp = (dARRAY*)malloc(sizeof(dARRAY));
  x_test_temp->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  progressbar * progress = progressbar_new("\033[1;32mLoading X_test :\033[0m",dims[1]);
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&x_test_temp->matrix[j]);
    if(j%dims[0]==0)
      progressbar_inc(progress);
  }
  progressbar_finish(progress);
  x_test_temp->shape[0] = dims[1];
  x_test_temp->shape[1] = dims[0];
  dARRAY * x_test = transpose(x_test_temp);
  free2d(x_test_temp);
  x_test_temp = NULL;
  fclose(fp);
  return x_test;
}

/**!
 * Function that is used to load y_test from file. 
 * @param filename Name of the file where the y_test resides. 
 * @param dims Dimensions of your test set (Number of classes,Number of examples)
*/
dARRAY * load_y_test(char * filename,int * dims){
  FILE * fp = NULL;
  fp = fopen(filename,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93mCould not open %s file!\033[0m\n",filename);
    exit(EXIT_FAILURE);
  }
  dARRAY * Y_test = (dARRAY*)malloc(sizeof(dARRAY));
  Y_test->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  progressbar * progress = progressbar_new("\033[1;32mLoading y_test :\033[0m",dims[1]);
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&Y_test->matrix[j]);
    if(j%dims[0]==0)
      progressbar_inc(progress);
  }
  progressbar_finish(progress);
  Y_test->shape[0] = dims[1];
  Y_test->shape[1] = dims[0];

  dARRAY * y_test = transpose(Y_test);
  free2d(Y_test);
  Y_test = NULL;
  fclose(fp);
  return y_test;
}

dARRAY * load_image(char * filename,int * dims){
  FILE * fp = NULL;
  fp = fopen(filename,"rb");
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open %s file!\033[0m\n",filename);
    exit(EXIT_FAILURE);
  }
  dARRAY * image = (dARRAY*)malloc(sizeof(dARRAY));
  image->matrix = (float*)calloc(dims[0]*dims[1],sizeof(float));
  for(int j=0;j<dims[0]*dims[1];j++){
    fscanf(fp,"%f ",&image->matrix[j]);
  }
  image->shape[0] = dims[0];
  image->shape[1] = dims[1];
  fclose(fp);
  fp=NULL;
  return image;
}

void create_mini_batches(){
  if(m->x_train->shape[1]!=m->Y_train->shape[1]){
    printf("\033[1;31mShape Error : \033[93mNumber of examples in X_train[%d,\033[1;31m%d\033[93m] != Y_train[%d,\033[1;31m%d\033[93m]!\033[0m\n",m->x_train->shape[0],m->x_train->shape[1],m->Y_train->shape[0],m->Y_train->shape[1]);
    exit(EXIT_FAILURE);
  }

  if(m->mini_batch_size>m->Y_train->shape[1]){
    printf("\033[1;31mMini Batch Error : \033[93mNumber of examples in train_set is less than the mini_batch_size specified!\033[1;31m [examples:%d < mini_batch_size: %d]\033[93m\nPlease make sure that the mini batch size is less than or equal to the number of examples in train_set.\033[0m\n",m->x_train->shape[1],m->mini_batch_size);
    exit(EXIT_FAILURE);
  }

  if(m->x_train->shape[1]==0 || m->Y_train->shape[1]==0){
    printf("\033[1;31mShape Error : \033[93mDataset Empty!\033m[0m\n");
    exit(EXIT_FAILURE);
  }

  int num_mini_batches = (m->x_train->shape[1]%m->mini_batch_size==0)?\
                          m->x_train->shape[1]/m->mini_batch_size :\
                          (int)floor(m->x_train->shape[1]/m->mini_batch_size) + 1;

  m->x_train_mini_batch = (dARRAY**)malloc(sizeof(dARRAY*)*num_mini_batches);
  m->y_train_mini_batch = (dARRAY**)malloc(sizeof(dARRAY*)*num_mini_batches);

  m->num_mini_batches = num_mini_batches;

  int remaining_mem_block = (m->x_train->shape[1]-((num_mini_batches-1)*m->mini_batch_size));
  
  for(int i=0;i<num_mini_batches;i++){
    m->x_train_mini_batch[i] = NULL;
    m->y_train_mini_batch[i] = NULL;
  }

  dARRAY * X_train = transpose(m->x_train);
  dARRAY * Y_train = transpose(m->Y_train);
  
  free2d(m->x_train);
  free2d(m->Y_train);
  m->x_train = NULL;
  m->Y_train = NULL;

  progressbar * progress = progressbar_new("\033[1;32mCreating Mini Batches :\033[0m",num_mini_batches*2);
  for(int i=0;i<num_mini_batches;i++){
    int row = 0;
    dARRAY * temp = (dARRAY*)malloc(sizeof(dARRAY));
    temp->matrix = (float*)calloc(m->mini_batch_size*X_train->shape[1],sizeof(float));
    if(i<num_mini_batches-1 || num_mini_batches==1){
      temp->shape[0] = m->mini_batch_size;
      temp->shape[1] = X_train->shape[1];
      for(int j=i*m->mini_batch_size;j<(i+1)*m->mini_batch_size && j<X_train->shape[0];j++){
        for(int k = 0;k<X_train->shape[1];k++){
          temp->matrix[row*X_train->shape[1]+k] = X_train->matrix[j*X_train->shape[1]+k];
        }
        row++;
      }
    }
    else{
      temp->shape[0] = remaining_mem_block;
      temp->shape[1] = X_train->shape[1];
      for(int j=i*m->mini_batch_size;j<X_train->shape[0] && j<X_train->shape[0];j++){
        for(int k = 0;k<X_train->shape[1];k++){
          temp->matrix[row*X_train->shape[1]+k] = X_train->matrix[j*X_train->shape[1]+k];
        }
        row++;
      }
    }
    m->x_train_mini_batch[i] = transpose(temp);
    free2d(temp);
    temp = NULL;
    progressbar_inc(progress);
  }

  for(int i=0;i<num_mini_batches;i++){
    int row = 0;
    dARRAY * temp = (dARRAY*)malloc(sizeof(dARRAY));
    temp->matrix = (float*)calloc(m->mini_batch_size*Y_train->shape[1],sizeof(float));
    if(i<num_mini_batches-1 || num_mini_batches==1){
      temp->shape[0] = m->mini_batch_size;
      temp->shape[1] = Y_train->shape[1];
      for(int j=i*m->mini_batch_size;j<(i+1)*m->mini_batch_size && j<Y_train->shape[0];j++){
        for(int k = 0;k<Y_train->shape[1];k++){
          temp->matrix[row*Y_train->shape[1]+k] = Y_train->matrix[j*Y_train->shape[1]+k];
        }
        row++;
      }
    }
    else{
      temp->shape[0] = remaining_mem_block;
      temp->shape[1] = Y_train->shape[1];
      for(int j=i*m->mini_batch_size;j<Y_train->shape[0] && j<Y_train->shape[0];j++){
        for(int k = 0;k<Y_train->shape[1];k++){
          temp->matrix[row*Y_train->shape[1]+k] = Y_train->matrix[j*Y_train->shape[1]+k];
        }
        row++;
      }
    }
    m->y_train_mini_batch[i] = transpose(temp);
    free2d(temp);
    temp = NULL;
    progressbar_inc(progress);
  }
  progressbar_finish(progress);
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

void (Create_Model)(){
  m = NULL;
  m = (__Model__*)malloc(sizeof(__Model__));
  m->graph = NULL;
  m->graph = NULL;
  m->current_layer = NULL;
  m->number_of_layers = 0;
  m->input_size = 0;
  m->output_size = 0;
}

void segfault_handler(){
  exit(EXIT_SUCCESS);
}

void (Destroy_Model)(){
  signal(SIGSEGV,segfault_handler);
  destroy_Graph(m->graph);
  if(m->x_cv!=NULL)
  free2d(m->x_cv);
  if(m->Y_cv!=NULL)
  free2d(m->Y_cv);
  if(m->x_test!=NULL)
  free2d(m->x_test);
  if(m->Y_test!=NULL)
  free2d(m->Y_test);
  for(int i=0;i<m->number_of_layers;i++){
    if(m->m_t_dW[i] != NULL) free2d(m->m_t_dW[i]);
    if(m->m_t_db[i] != NULL) free2d(m->m_t_db[i]);
    if(m->v_t_dW[i] != NULL) free2d(m->v_t_dW[i]);
    if(m->v_t_db[i] != NULL) free2d(m->v_t_db[i]);
    if(m->cache_dW[i] != NULL) free2d(m->cache_dW[i]);
    if(m->cache_db[i] != NULL) free2d(m->cache_db[i]);
    m->m_t_dW[i] = NULL;
    m->v_t_dW[i] = NULL;
    m->m_t_db[i] = NULL;
    m->v_t_db[i] = NULL;
    m->cache_dW[i] = NULL;
    m->cache_db[i] = NULL;
  }
  for(int i=0;i<m->num_mini_batches;i++){
    if(m->x_train_mini_batch[i]!=NULL) free2d(m->x_train_mini_batch[i]);
    if(m->y_train_mini_batch[i]!=NULL) free2d(m->y_train_mini_batch[i]);
    m->x_train_mini_batch[i] = NULL;
    m->y_train_mini_batch[i] = NULL;
  }
  free(m->x_train_mini_batch);
  free(m->y_train_mini_batch);
  m->x_train_mini_batch = NULL;
  m->y_train_mini_batch = NULL;
  m->x_train = NULL;
  m->Y_train = NULL;
  m->x_cv = NULL;
  m->x_test = NULL;
  m->Y_test = NULL;
  m->Y_cv = NULL;
  m->output = NULL;
  free(m);
  m = NULL;
}

void dump_image(dARRAY * images,char * filename){
  FILE * fp = NULL;
  fp = fopen(filename,"a+");
  if(fp==NULL){
    exit(EXIT_FAILURE);
  }
  else{
    dARRAY * temp = transpose(images);
    for(int i=0;i<temp->shape[0];i++){
      for(int j=0;j<temp->shape[1];j++){
        fprintf(fp,"%lf ",temp->matrix[i*temp->shape[1]+j]);
      }
    }
    free2d(temp);
    temp = NULL;
  }
}

void (Model)(Model_args model_args){
  get_safe_nn_threads();

  if(model_args.X_train==NULL || model_args.y_train==NULL){
    printf("\033[1;31mDataset Error : \033[93mNo dataset passed to the model constructor.\033[0m\n");
    exit(EXIT_FAILURE);
  }

  m->x_train = model_args.X_train;
  m->x_test = model_args.X_test;
  m->x_cv = model_args.X_cv;

  m->Y_train = model_args.y_train;
  m->Y_test = model_args.y_test;
  m->Y_cv = model_args.y_cv;

  m->ckpt_every = model_args.checkpoint_every;

  //initialize regualrization hyperparameters
  m->loss = model_args.loss;
  m->lambda = model_args.weight_decay;
  m->regularization = model_args.regularization;

  m->num_of_training_examples = m->x_train->shape[1];
  //initialize hyperparameters for various optimizers
  m->optimizer = model_args.optimizer; // Optimizer choice
  m->learning_rate = model_args.lr; // hyperparameter for step size for optimization algorithm
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
  m->mini_batch_size = model_args.batch_size;
  m->num_iter = model_args.epochs;

  create_mini_batches();
  if(!strcasecmp(m->loss,"cross_entropy_loss")) cross_entropy_loss();
  if(!strcasecmp(m->loss,"MSELoss")) MSELoss();

  m->input_size = m->x_train_mini_batch[0]->shape[0];
  
  m->print_cost = model_args.print_cost;
  m->init = __init__;
  m->forward = __forward__;
  m->backward = __backward__;
  m->load_model = __load_model__;
  m->save_model = __save_model__;
  m->fit = __fit__;
  // m->predict = __predict__;
  m->summary = __summary__;
  m->test = __test__;
  m->init();
  m->total_parameters = get_total_params();
}