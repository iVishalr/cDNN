#include "./model.h"

__Model__ * m;

void __init__(){
  m->input_size = 0;
  m->output_size = 0;
  m->predicting = 0;
  m->number_of_layers = 0;
  m->test_accuracy = 0.0;
  m->train_accuracy = 0.0;
  m->train_cost = 0.0;
  m->cross_val_accuracy = 0.0;
  m->model_layer_weights = NULL;
  m->model_layer_biases = NULL;
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
  // m->model_layer_weights = (dARRAY *)malloc(sizeof(dARRAY)*m->number_of_layers);
  // m->model_layer_biases = (dARRAY *)malloc(sizeof(dARRAY)*m->number_of_layers);
  // temp = m->graph;
  // int index=0;
  // while(temp!=NULL){
  //   m->current_layer = temp;
  //   m->model_layer_weights[index] = temp->DENSE->weights;
  //   m->model_layer_biases[index] = temp->DENSE->bias;
  //   temp = temp->next_layer;
  // }
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

void __fit__(){
  int i = 1;
  while(i<=m->num_iter){
    __forward__();
    if(m->print_cost && i%100==0)
      printf("\033[96mIteration (%d) - Cost : \033[0m%lf\n",i,cross_entropy_loss(m->current_layer->DENSE,m->Y_train));
    __backward__();
    GD(m->learning_rate);
    i++;
  }
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

void __load_model__(){
  SAVE_MODEL * loaded_model = (SAVE_MODEL*)malloc(sizeof(SAVE_MODEL));
  FILE * file_ptr;
  file_ptr = fopen("weights.data","rb");
  if(file_ptr==NULL){
    printf("ERROR!\n");
  exit(EXIT_FAILURE);
  }
  fread(loaded_model,sizeof(SAVE_MODEL),1,file_ptr);
  Computation_Graph * temp = m->graph->next_layer;
  int index = 0;
  while(temp!=NULL){
    printf("Attempting to load weights!\n");
    __weights__ * new_w = loaded_model->weights_arr[index];
    printf("Attempting to load weights!\n");
    __biases__ * new_b = loaded_model->biases_arr[index];
    printf("Attempting to load weights!\n");
    temp->DENSE->weights = new_w->weight;
    printf("Attempting to load weights!\n");
    temp->DENSE->bias = new_b->bias;
    printf("Attempting to load weights!\n");
    temp = temp->next_layer;
    index++;
    new_w=NULL;
    new_b=NULL;
  }
  m->learning_rate = loaded_model->learning_rate;
  m->lambda = loaded_model->lambda;
  printf("Learning rate : %lf\n",loaded_model->learning_rate);
}

void __save_model__(char * file_name){
  Computation_Graph * temp = m->graph->next_layer;
  SAVE_MODEL * __save_model__ = (SAVE_MODEL *)malloc(sizeof(SAVE_MODEL));
  __save_model__->num_layers = m->number_of_layers;
  __save_model__->lambda = m->lambda;
  __save_model__->learning_rate = m->learning_rate;
  int index = 0;
  while(temp!=NULL){
    __weights__ * new_w = (__weights__*)malloc(sizeof(__weights__));
    new_w->weight = temp->DENSE->weights;
    __save_model__->weights_arr[index] = new_w;
    __biases__  * new_b = (__biases__*)malloc(sizeof(__biases__));
    new_b->bias = temp->DENSE->bias;
    __save_model__->biases_arr[index]=new_b;
    new_w = NULL;
    new_b = NULL;
    index++;
    temp = temp->next_layer;
  }
  FILE * file_ptr;
  file_ptr = fopen("weights.data","wb+");
  if(file_ptr==NULL){
    printf("ERROR!\n");
    exit(EXIT_FAILURE);
  }
  fwrite(__save_model__,sizeof(SAVE_MODEL),1,file_ptr);
  fclose(file_ptr);
}

void __summary__(){}

void (create_model)(){
  m = NULL;
  m = (__Model__*)malloc(sizeof(__Model__));
  m->graph = NULL;
  m->graph = NULL;
  m->current_layer = NULL;
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
}