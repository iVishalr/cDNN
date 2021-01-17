#include "./model.h"

__Model__ * m;

void __init__(){
  m->predicting = 0;
  m->test_accuracy = 0.0;
  m->train_accuracy = 0.0;
  m->train_cost = 0.0;
  m->cross_val_accuracy = 0.0;
  m->output = NULL;
  __initialize_params__();
}

void __initialize_params__(){
  Computation_Graph * temp = m->graph;
  temp = temp->next_layer;
  while(temp!=NULL){
    m->current_layer = temp;
    temp->DENSE->initalize_params();
    temp = temp->next_layer;
  }
  m->current_layer = m->graph;
}

void __forward__(){
  Computation_Graph * temp = m->graph;
  temp = temp->next_layer;
  while(temp!=NULL){
    m->current_layer = temp;
    temp->DENSE->forward_prop();
    temp = temp->next_layer;
  }
}

void __backward__(){
  Computation_Graph * temp = m->current_layer;
  while(temp->prev_layer!=NULL){
    m->current_layer = temp;
    temp->DENSE->back_prop();
    temp = temp->prev_layer;
  }
}

double calculate_accuracy(dARRAY * predicted, dARRAY * gnd_truth){
  int success = 0;
  dARRAY * temp = (dARRAY*)malloc(sizeof(dARRAY));
  temp->matrix = (double*)calloc(predicted->shape[0]*predicted->shape[1],sizeof(double));
  for(int i = 0;i<predicted->shape[0]*predicted->shape[1];i++){
    temp->matrix[i] = round(predicted->matrix[i]);
  }
  temp->shape[0] = predicted->shape[0];
  temp->shape[1] = predicted->shape[1];
  for(int i = 0;i<predicted->shape[0]*predicted->shape[1];i++){
    if(temp->matrix[i]==gnd_truth->matrix[i]) success++;
  }
  free2d(temp);
  temp = NULL;
  return success/(double)gnd_truth->shape[1];
}

void append_to_file(double value,char * filename,char * mode){
  FILE * fp = NULL;
  fp = fopen(filename,mode);
  if(fp==NULL){
    printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
    exit(EXIT_FAILURE);
  }
  fprintf(fp,"%lf ",value);
  fclose(fp);
}

void __fit__(){
  int i = 1;
  while(i<=m->num_iter){
    __forward__();
    if(i%100==0 && m->print_cost){
      m->train_cost = cross_entropy_loss(m->current_layer->DENSE,m->Y_train);
      m->train_accuracy = calculate_accuracy(m->output,m->Y_train);
      printf("\033[96m%d. Cost : \033[0m%lf ",i,m->train_cost);
      printf("\033[96m Accuracy : \033[0m%lf\n",m->train_accuracy);
      // append_to_file(m->train_cost,"./bin/cost.data","ab+");
      // append_to_file(m->train_accuracy,"./bin/train_acc.data","ab+");
    }
    __backward__();
    GD(m->learning_rate);
    i++;
  }
  // system("python3 ./src/plot/plot_scores.py");
}

void __predict__(dARRAY * input_feature){
  m->graph->INPUT->A = input_feature;
  m->predicting = 1;
  __forward__();
  printf("Output : ");
  for(int i=0;i<m->output->shape[0];i++){
    for(int j=0;j<m->output->shape[1];j++){
      printf("%lf ",m->output->matrix[i*m->output->shape[1]+j]);
    }
    printf("\n");
  }
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
  // fp = fopen("./bin/model_biases1.t7","ab+");
  // if(fp==NULL){
  //   printf("\033[1;31mFile Error : \033[93m Could not open the specified file!\033[0m\n");
  //   exit(EXIT_FAILURE);
  // }
  // temp = m->graph->next_layer;
  // for(int i=0;i<m->number_of_layers-1;i++){
  //   for(int j=0;j<temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1];j++)
  //     fprintf(fp,"%lf ",temp->DENSE->bias->matrix[j]);
  //   temp = temp->next_layer;
  // }
  fclose(fp);
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
  // printf("shape of X_train : ");
  // shape(m->x_train);
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

  // printf("Shape of Y_train : ");
  // shape(m->Y_train);
  free2d(Y_train);
  Y_train = NULL;
  fclose(fp);
}

void __summary__(){}

long int get_total_params(){
  long int total_params = 0;
  Computation_Graph * temp = m->graph->next_layer;
  while(temp!=NULL){
    total_params+= temp->DENSE->weights->shape[0]*temp->DENSE->weights->shape[1];
    total_params+= temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1];
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

  m->loss = model_args.loss;
  m->lambda = model_args.lambda;
  m->regularization = model_args.regularization;

  m->optimizer = model_args.optimizer;
  m->learning_rate = model_args.learning_rate;

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