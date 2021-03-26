#include <model.h>
#include <cross_entropy_loss.h>

extern __Model__ * m;
cross_entropy_loss_layer * loss_layer = NULL;

void forward_pass_L2_LOSS(){
  int number_of_examples = m->y_train_mini_batch[m->current_mini_batch]->shape[1];
  // int number_of_examples = m->Y_train->shape[1];
  dARRAY * loss = NULL;
  dARRAY * Y = m->y_train_mini_batch[m->current_mini_batch];
  // dARRAY * Y = m->Y_train;
  int act_dims[] = {m->output->shape[0],m->output->shape[1]};

  if(!strcasecmp(m->current_layer->prev_layer->DENSE->activation,"softmax")){
    dARRAY * log_y_hat= NULL;
    log_y_hat = (dARRAY *)malloc(sizeof(dARRAY));
    log_y_hat->matrix = (float*)calloc(act_dims[0]*act_dims[1],sizeof(float));

    for(int i=0;i<act_dims[0]*act_dims[1];i++){
      log_y_hat->matrix[i] = log(m->output->matrix[i]);
    }
    log_y_hat->shape[0] = act_dims[0];
    log_y_hat->shape[1] = act_dims[1];

    dARRAY * loss_term_temp = multiply(Y,log_y_hat);
    
    free2d(log_y_hat);
    log_y_hat = NULL;

    loss = sum(loss_term_temp,0);

    free2d(loss_term_temp);
    loss_term_temp = NULL;
  }
  else{
    dARRAY * log_a = NULL;
    log_a = (dARRAY*)malloc(sizeof(dARRAY));
    log_a->matrix = (float*)calloc(act_dims[0]*act_dims[1],sizeof(float));

    #pragma omp parallel for num_threads(8)
    for(int i=0;i<act_dims[0]*act_dims[1];i++){
      log_a->matrix[i] = logf(m->output->matrix[i]);
    }
    log_a->shape[0] = act_dims[0];
    log_a->shape[1] = act_dims[1];

    dARRAY * temp_ones = ones(act_dims);
    dARRAY * temp_sub = subtract(temp_ones,m->output);
    
    dARRAY * log_one_y_hat = NULL;
    log_one_y_hat = (dARRAY*)malloc(sizeof(dARRAY));
    log_one_y_hat->matrix = (float*)calloc(act_dims[0]*act_dims[1],sizeof(float));
    
    #pragma omp parallel for num_threads(8)
    for(int i=0;i<act_dims[0]*act_dims[1];i++){
      log_one_y_hat->matrix[i] = logf(temp_sub->matrix[i]);
    }
    log_one_y_hat->shape[0] = act_dims[0];
    log_one_y_hat->shape[1] = act_dims[1];

    free2d(temp_sub);
    temp_sub = NULL;

    dARRAY * loss_term_1 = multiply(Y,log_a);
    
    free2d(log_a);
    log_a = NULL;

    temp_sub = subtract(temp_ones,Y);
    
    free2d(temp_ones);
    temp_ones = NULL;

    dARRAY * loss_term_2 = multiply(temp_sub,log_one_y_hat);
    
    free2d(temp_sub);
    temp_sub = NULL;

    free2d(log_one_y_hat);
    log_one_y_hat=NULL;

    loss = add(loss_term_1,loss_term_2);

    free2d(loss_term_1);
    free2d(loss_term_2);
    loss_term_1 = NULL;
    loss_term_2 = NULL;
  }

  // dARRAY * sum_of_losses_arr = (dARRAY*)malloc(sizeof(dARRAY));
  // sum_of_losses_arr->matrix = (float*)calloc(1,sizeof(float));
  // sum_of_losses_arr->shape[0] = 1;
  // sum_of_losses_arr->shape[1] = 1;
  
  // cblas_saxpy(loss->shape[0]*loss->shape[1],1.f,loss->matrix,0,sum_of_losses_arr->matrix,0);
  dARRAY * sum_of_losses_arr = sum(loss,1);
  float sum_of_losses = sum_of_losses_arr->matrix[0];

  free2d(sum_of_losses_arr);
  free2d(loss);
  sum_of_losses_arr = NULL;
  loss = NULL;

  float cost = 0.0f;
  float cross_entropy_cost = -1 * sum_of_losses/(float)number_of_examples;

  float reg_cost=0.0;
  Computation_Graph * temp = NULL;
  
  if(m->regularization!=NULL){
    temp = m->graph->next_layer;
    if(!strcasecmp(m->regularization,"L2")){
      float layer_frobenius = 0.0;
      while(temp->next_layer->type!=LOSS){
        layer_frobenius += frobenius_norm(temp->DENSE->weights);
        temp = temp->next_layer;
      }
      reg_cost = m->lambda*layer_frobenius/(2.0*number_of_examples);
    }
    else if(!strcasecmp(m->regularization,"L1")){
      float layer_manhattan = 0.0;
      while(temp->next_layer->type!=LOSS){
        layer_manhattan += Manhattan_distance(temp->DENSE->weights);
        temp = temp->next_layer;
      }
      reg_cost = m->lambda * layer_manhattan/(float)(2.0*number_of_examples);
    }
    cost = cross_entropy_cost + reg_cost;
    temp = NULL;
    
    m->iter_cost = cost;
  }
  else{
    cost = cross_entropy_cost;
    m->iter_cost = cost;
  }
}

void backward_pass_L2_LOSS(){
  int act_dims[] = {m->output->shape[0],m->output->shape[1]};
  if(!strcasecmp(m->current_layer->prev_layer->DENSE->activation,"softmax")){
    dARRAY * temp = divison(m->y_train_mini_batch[m->current_mini_batch],m->output);
    // dARRAY * temp = divison(m->Y_train,m->output);
    dARRAY * class_sum = sum(temp,0);
    free2d(temp);
    temp = NULL;
    loss_layer->grad_out = mulScalar(class_sum,-1.0);
    free2d(class_sum);
    class_sum = NULL;
  }
  else{
    dARRAY * one = ones(act_dims);
    dARRAY * temp1 = subtract(m->y_train_mini_batch[m->current_mini_batch],m->output);
    // dARRAY * temp1 = subtract(m->Y_train,m->output);
    dARRAY * temp2 = subtract(one,m->output);
    dARRAY * temp3 = multiply(m->output,temp2);
    free2d(one);
    one = NULL;

    dARRAY * lgrad2 = divison(temp1,temp3);

    free2d(temp1);
    free2d(temp2);
    free2d(temp3);
    temp1 = NULL;
    temp2 = NULL;
    temp3 = NULL;

    loss_layer->grad_out = mulScalar(lgrad2,-1.0f);
    free2d(lgrad2);
    lgrad2 = NULL;
  }
}

void (cross_entropy_loss)(cross_entropy_loss_args args){
  loss_layer = (cross_entropy_loss_layer*)malloc(sizeof(cross_entropy_loss_layer));
  loss_layer->cost = 0.0;
  loss_layer->grad_out = NULL;
  loss_layer->forward = forward_pass_L2_LOSS;
  loss_layer->backward = backward_pass_L2_LOSS;
  // loss_layer->gnd_truth = m->y_train_mini_batch[m->current_mini_batch];
  append_graph(loss_layer,"loss");
}