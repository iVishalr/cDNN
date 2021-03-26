#include <model.h>

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
  if(m->testing){
    printf("Ground Truth\n");
    for(int  i=0;i<gnd_truth->shape[0];i++){
      for(int j=0;j<gnd_truth->shape[1];j++){
        printf("%d ",(int)gnd_truth->matrix[i*gnd_truth->shape[1]+j]);
      }
      printf("\n");
    }
  }
  int success = 0;
  dARRAY * temp = (dARRAY*)malloc(sizeof(dARRAY));
  temp->matrix = (float*)calloc(predicted->shape[0]*predicted->shape[1],sizeof(float));
  for(int i = 0;i<predicted->shape[0]*predicted->shape[1];i++){
    temp->matrix[i] = predicted->matrix[i]<0.5 ? 0 : 1;
  }
  temp->shape[0] = predicted->shape[0];
  temp->shape[1] = predicted->shape[1];

  if(m->testing){
    printf("Predicted array : \n");
    if(temp==NULL) printf("predicted was null for some reason\n");
    for(int  i=0;i<temp->shape[0];i++){
      for(int j=0;j<temp->shape[1];j++){
        if((int)temp->matrix[i*temp->shape[1]+j]==(int)gnd_truth->matrix[i*gnd_truth->shape[1]+j])
          printf("\033[92m%d\033[0m ",(int)temp->matrix[i*temp->shape[1]+j]);
        else{
          printf("\033[1;31m%d\033[0m ",(int)temp->matrix[i*temp->shape[1]+j]);
        }
      }
      printf("\n");
    }
  }  

  for(int i = 0;i<predicted->shape[0]*predicted->shape[1];i++){
    if(temp->matrix[i]==gnd_truth->matrix[i]) success++;
  }
  if(m->testing)
    printf("number of success : %d\n",success);
  if(predicted->shape[0]!=1) success = success/predicted->shape[0];
  if(m->testing)
    printf("number of success/2 : %d\n",success);
  free2d(temp);
  temp = NULL;
  if(m->testing)
    printf("Avg accuracy : %f\n",success/(float)predicted->shape[1]);
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

void append_to_file(float * arr ,char * filename,char * mode){
  FILE * fp = NULL;
  fp = fopen(filename,mode);
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
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
  float sum_cost = m->train_cost*m->current_iter;
  float sum_train_acc = m->train_accuracy*m->current_iter;
  float sum_train_val_acc = m->cross_val_accuracy*m->current_iter;
  float * train_cost_arr = (float*)calloc(m->num_iter==-1?1000000:m->num_iter*m->num_mini_batches,sizeof(float));
  float * train_acc_arr = (float*)calloc(m->num_iter==-1?1000000:m->num_iter*m->num_mini_batches,sizeof(float));
  float * val_acc_arr = (float*)calloc(m->num_iter==-1?1000000:m->num_iter*m->num_mini_batches,sizeof(float));
  if(m->num_iter==-1){
    iterations = i;
  }
  else iterations = m->num_iter;
  while(i<=iterations){
    for(int mini_batch=0;mini_batch<m->num_mini_batches;mini_batch++){
      m->current_mini_batch = mini_batch;
      __forward__();
      sum_cost += m->iter_cost;
      sum_train_acc += calculate_accuracy(m->output,m->y_train_mini_batch[m->current_mini_batch]);
      // sum_train_acc += calculate_accuracy(m->output,m->Y_train);
      dARRAY * temp = calculate_val_test_acc(m->x_cv,m->Y_cv);
      sum_train_val_acc += temp->matrix[0];
      if(m->print_cost){
        m->train_cost = i==m->current_iter ? sum_cost/(float)i : sum_cost/(float)m->current_iter;
        m->train_accuracy = i==m->current_iter ? sum_train_acc/(float)i : sum_train_acc/(float)m->current_iter;
        m->cross_val_accuracy = i==m->current_iter ? sum_train_val_acc/(float)i : sum_train_val_acc/(float)m->current_iter;
        train_cost_arr[index] = m->train_cost;
        train_acc_arr[index] = m->train_accuracy;
        val_acc_arr[index] = m->cross_val_accuracy;
        printf("\033[96m%d. %d. Cost : \033[0m%f ",i,mini_batch,m->train_cost);
        // printf("\033[96m%d. Cost : \033[0m%f ",i,m->train_cost);
        printf("\033[96m train_acc : \033[0m%f ",m->train_accuracy);
        printf("\033[96m val_acc : \033[0m%f\n",m->cross_val_accuracy);
        if(train_cost_arr[index]>=2.5*train_cost_arr[0]){
          printf("Cost is exploding hence exiting. Try reducing your learning rate and try again!\n");
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
      if(m->ckpt_every!=-1 && (i%m->ckpt_every==0 || (i==m->num_iter && m->num_iter!=-1))){
        time_t rawtime;
        struct tm * timeinfo;

        time ( &rawtime );
        timeinfo = localtime ( &rawtime );
    
        char buffer[1024];
        snprintf(buffer,sizeof(buffer),"model_%d_%s.t7",i,asctime (timeinfo));
        __save_model__(buffer);

        // append_to_file(train_cost_arr,"./bin/cost.data","ab+");
        // append_to_file(train_acc_arr,"./bin/train_acc.data","ab+");
        // append_to_file(val_acc_arr,"./bin/val_acc.data","ab+");
      }
      m->current_iter += 1;
      free(temp);
      temp = NULL;
    }
    i++;
    if(m->num_iter==-1) iterations = i;
  }
  append_to_file(train_cost_arr,"./bin/cost.data","ab+");
  append_to_file(train_acc_arr,"./bin/train_acc.data","ab+");
  append_to_file(val_acc_arr,"./bin/val_acc.data","ab+");
}

void __predict__(dARRAY * input_feature){
  m->graph->INPUT->A = input_feature;
  m->predicting = 1;
  dARRAY * prediction = calculate_val_test_acc(input_feature,NULL);
  
  if(prediction->shape[0]!=1){
    printf("Score : [%f,%f]\n",prediction->matrix[0],prediction->matrix[1]);
    if(prediction->matrix[0]>=0.5 && prediction->matrix[1]<0.5) printf("CAT\n");
    else printf("DOG\n");
  }
  else{
    printf("Score : %f\n",prediction->matrix[0]);
    if(prediction->matrix[0]>=0.5) printf("DOG\n");
    else printf("CAT\n");
  }
  m->predicting=0;
}

void __test__(){
  // m->testing=1;
  dARRAY * acc = calculate_val_test_acc(m->x_test,m->Y_test);
  printf("\033[96mTest Accuracy : \033[0m%f\n",acc->matrix[0]);
  free2d(acc);
  acc = NULL;
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
  fprintf(fp,"%f ",m->train_cost);
  fprintf(fp,"%f ",m->train_accuracy);
  fprintf(fp,"%f ",m->cross_val_accuracy);
  fprintf(fp,"%d ",m->current_iter-1);
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
  
  printf("Mini-Batch-Size : %d, Number of Batches to Create : %d\n",m->mini_batch_size,num_mini_batches);
  printf("Remaining block size: %d\n",remaining_mem_block);
  
  for(int i=0;i<num_mini_batches;i++){
    m->x_train_mini_batch[i] = NULL;
    m->y_train_mini_batch[i] = NULL;
  }

  printf("DATA IN Y_TRAIN : %f %f %f %f\n",m->Y_train->matrix[0],m->Y_train->matrix[5],m->Y_train->matrix[10],m->Y_train->matrix[15]);

  dARRAY * X_train = transpose(m->x_train);
  dARRAY * Y_train = transpose(m->Y_train);
  printf("Shape(X_train) : ");
  shape(X_train);
  printf("Shape(Y_train) : ");
  shape(Y_train);
  
  free2d(m->x_train);
  free2d(m->Y_train);
  m->x_train = NULL;
  m->Y_train = NULL;

  
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
  }

  shape(m->x_train_mini_batch[0]);

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
    shape(m->y_train_mini_batch[i]);
  }
  printf("outside loop\n");
  for(int i=0;i<num_mini_batches;i++){
    // printf("Shape(X_mini[%d]) : ",i);
    // shape(m->x_train_mini_batch[i]);
    // printf("Shape(Y_mini[%d]) : ",i);
    shape(m->y_train_mini_batch[i]);
    printf("\n");
  }
  // printf("DATA IN Y_TRAIN MINI BATCH [0] : %f %f %f %f\n",m->y_train_mini_batch[0]->matrix[0],m->y_train_mini_batch[0]->matrix[5],m->y_train_mini_batch[0]->matrix[10],m->y_train_mini_batch[0]->matrix[15]);
  printf("Created Mini Batches!\n");
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
  if(m->x_cv!=NULL)
  free2d(m->x_cv);
  if(m->Y_cv!=NULL)
  free2d(m->Y_cv);
  if(m->output!=NULL)
  // free2d(m->output);
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
  m->x_train = m->Y_train = m->x_cv = m->x_test = m->Y_test = m->Y_cv = m->output = NULL;
  free(m);
  m = NULL;
}

void dump_image(dARRAY * images){
  FILE * fp = NULL;
  fp = fopen("./mini_batchy.data","a+");
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

  m->num_of_training_examples = m->x_train->shape[1];
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

  create_mini_batches();
  printf("Adding loss functions\n");
  if(!strcasecmp(m->loss,"cross_entropy_loss")) cross_entropy_loss();
  if(!strcasecmp(m->loss,"MSELoss")) MSELoss();

  printf("Setting up input size\n");
  m->input_size = m->x_train_mini_batch[0]->shape[0];
  // m->input_size = m->x_train->shape[0];
  // m->output_size = model_args.Y_train->shape[0];

  // if(m->input_size!=m->graph->INPUT->input_features_size){
  //   printf("\033[1;31mModel Error : \033[93m Size of Input Layer does not match the size of x_train.\033[0m\n");
  //   printf("\033[96mHint : \033[93mCheck if layer_size of input layer == x_train->shape[0]\033[0m\n");
  //   exit(EXIT_FAILURE);
  // }
  
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