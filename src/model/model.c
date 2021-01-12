#include "./model.h"

__Model__ * m;

void __init__(){
  m->predicting = 0;
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

void load_weights(){
  
}

void load_biases(){
  // FILE * fp1;
  // Computation_Graph * temp = m->graph->next_layer;
  
}

void __load_model__(){
  dARRAY weights[m->number_of_layers-1];
  dARRAY biases[m->number_of_layers-1];

  Computation_Graph * temp = m->graph->next_layer;

  // printf("allocating memory\n");
  for(int i=0;i<m->number_of_layers-1 && temp!=NULL;i++){
    shape(temp->DENSE->weights);
    shape(temp->DENSE->bias);
    weights[i].matrix = (double*)calloc(temp->DENSE->weights->shape[0]*temp->DENSE->weights->shape[1],sizeof(double));
    biases[i].matrix = (double*)calloc(temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1],sizeof(double));
    temp = temp->next_layer;
  }
  // printf("Finished memory\n");
  temp = m->graph->next_layer;

  FILE * fp = NULL;
  fp = fopen("./bin/model_weights.t7","rb");

  for(int i=0;i<m->number_of_layers-1;i++){
    if(temp==NULL) printf("null\n");
    for(int j=0;j<temp->DENSE->weights->shape[0]*temp->DENSE->weights->shape[1];j++){
      fscanf(fp,"%lf ",&weights[i].matrix[j]);
    }
    weights[i].shape[0] = temp->DENSE->weights->shape[0];
    weights[i].shape[1] = temp->DENSE->weights->shape[1];
    temp = temp->next_layer;
  }
  // printf("%lf\n",weights[0].matrix[0]);
  // printf("%lf\n",weights[1].matrix[0]);

  printf("weight matrix of layer 3 : \n");
  for(int i=0;i<weights[1].shape[0];i++){
    for(int j=0;j<weights[1].shape[1];j++){
      printf("%lf ",weights[1].matrix[i*weights[1].shape[1]+j]);
    }
    printf("\n");
  }
  printf("W3 (%d,%d)\n",weights[1].shape[0],weights[1].shape[1]);
  // shape((dARRAY*)weights[0]);
  // shape((dARRAY*)weights[1]);
  fclose(fp);


  sleep(1000);
  

  
  temp = m->graph->next_layer;

  FILE * fp1 = NULL;
  fp1 = fopen("./bin/model_biases.t7","rb");

  temp = m->graph->next_layer;

  for(int i=0;i<m->number_of_layers-1;i++){
    for(int j=0;j<temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1];j++){
      fscanf(fp1,"%lf ",&biases[i].matrix[j]);
    }
    biases[i].shape[0] = temp->DENSE->bias->shape[0];
    biases[i].shape[1] = temp->DENSE->bias->shape[1];
    temp = temp->next_layer;
  }

  // printf("%lf\n",biases[0].matrix[0]);
  // printf("%lf\n",biases[1].matrix[0]);

  printf("bias matrix of layer 2 : \n");
  for(int i=0;i<biases[0].shape[0];i++){
    for(int j=0;j<biases[0].shape[1];j++){
      printf("%lf ",biases[0].matrix[i*biases[0].shape[1]+j]);
    }
    printf("\n");
  }
  fclose(fp1);
}

void __save_model__(char * file_name){
  FILE * fp = NULL;
  fp = fopen("./bin/model_weights.t7","ab+");
  Computation_Graph * temp = m->graph->next_layer;
  printf("first ele : %lf\n",temp->DENSE->weights->matrix[0]);
  shape(temp->DENSE->weights);
  for(int i=0;i<m->number_of_layers-1;i++){
    for(int j=0;j<temp->DENSE->weights->shape[0]*temp->DENSE->weights->shape[1];j++)
      fprintf(fp,"%lf ",temp->DENSE->weights->matrix[j]);
    temp = temp->next_layer;
  }
  fclose(fp);
  fp = NULL;
  fp = fopen("./bin/model_biases.t7","ab+");
  temp = m->graph->next_layer;
  for(int i=0;i<m->number_of_layers-1;i++){
    for(int j=0;j<temp->DENSE->bias->shape[0]*temp->DENSE->bias->shape[1];j++)
      fprintf(fp,"%lf ",temp->DENSE->bias->matrix[j]);
    temp = temp->next_layer;
  }
  fclose(fp);
}

void __summary__(){}

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
}