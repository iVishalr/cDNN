#include "../model/model.h"
#include "./cross_entropy_loss.h"

extern __Model__ * m;
cross_entropy_loss_layer * loss_layer = NULL;

void forward_pass_L2_LOSS(){
  //Store the number of training examples in a variable
  int number_of_examples = m->Y_train->shape[1];
  dARRAY * Y = loss_layer->gnd_truth;
  //we will also store prev layer activation dims
  int act_dims[] = {m->current_layer->prev_layer->DENSE->A->shape[0],m->current_layer->prev_layer->DENSE->A->shape[0]};

  dARRAY * log_y_hat= NULL;
  log_y_hat = (dARRAY *)malloc(sizeof(dARRAY));
  log_y_hat->matrix = (double*)calloc(act_dims[0]*act_dims[1],sizeof(double));
  //calculate log(y_pred)
  omp_set_num_threads(4);
  #pragma omp parallel for
  for(int i=0;i<act_dims[0]*act_dims[1];i++){
    log_y_hat->matrix[i] = log(m->current_layer->prev_layer->DENSE->A->matrix[i]);
  }
  log_y_hat->shape[0] = act_dims[0];
  log_y_hat->shape[1] = act_dims[1];

  //calculating (1-y_pred)
  dARRAY * temp_ones = ones(act_dims);
  dARRAY * temp_sub = subtract(temp_ones,m->current_layer->prev_layer->DENSE->A);
  
  dARRAY * log_one_y_hat = NULL;
  log_one_y_hat = (dARRAY *)malloc(sizeof(dARRAY));
  log_one_y_hat->matrix = (double*)calloc(act_dims[0]*act_dims[1],sizeof(double));
  //calculate log(1-y_pred)
  omp_set_num_threads(4);
  #pragma omp parallel for
  for(int i=0;i<act_dims[0]*act_dims[1];i++){
    log_one_y_hat->matrix[i] = log(temp_sub->matrix[i]);
  }
  log_one_y_hat->shape[0] = act_dims[0];
  log_one_y_hat->shape[1] = act_dims[1];

  free2d(temp_sub);
  temp_sub = NULL;
  //calculate y*log(y^)
  //Y could be Y_train, Y_test, Y_cv
  dARRAY * loss_term_1 = multiply(Y,log_y_hat);
  
  free2d(log_y_hat);
  log_y_hat = NULL;

  //calculate (1-Y)
  temp_sub = subtract(temp_ones,Y);
  
  free2d(temp_ones);
  temp_ones = NULL;

  //calculate (1-Y)*log(1-Y^)
  dARRAY * loss_term_2 = multiply(temp_sub,log_one_y_hat);
  
  free2d(temp_sub);
  temp_sub = NULL;

  free2d(log_one_y_hat);
  log_one_y_hat=NULL;

  dARRAY * loss = add(loss_term_1,loss_term_2);
  
  free2d(loss_term_1);
  free2d(loss_term_2);
  loss_term_1 = loss_term_2 = NULL;

  dARRAY * sum_of_losses = sum(loss,1);

  free2d(loss);
  loss = NULL;

  dARRAY * cost = NULL;
  dARRAY * cross_entropy_cost = divScalar(sum_of_losses,(double)(-1 * number_of_examples));

  double reg_cost=0.0;
  Computation_Graph * temp = NULL;
  
  if(m->regularization!=NULL){
    temp = m->graph->next_layer;
    if(!strcasecmp(m->regularization,"L2")){
      double layer_frobenius = 0.0;
      while(temp!=NULL){
        layer_frobenius += frobenius_norm(temp->DENSE->weights);
        temp = temp->next_layer;
      }
      reg_cost = m->lambda*layer_frobenius/(2*number_of_examples);
    }
    else if(!strcasecmp(m->regularization,"L1")){
      double layer_manhattan = 0.0;
      while(temp!=NULL){
        layer_manhattan += Manhattan_distance(temp->DENSE->weights);
        temp = temp->next_layer;
      }
      reg_cost = m->lambda * layer_manhattan/(2*number_of_examples);
    }
  }
  if(m->regularization!=NULL){
    cost = addScalar(cross_entropy_cost,reg_cost);
    temp = NULL;

    free2d(cross_entropy_cost);
    cross_entropy_cost = NULL;

    free2d(sum_of_losses);
    sum_of_losses = NULL;

    double total_cost = cost->matrix[0];
    
    free2d(cost);
    cost = NULL;
    
    m->iter_cost = total_cost;
  }
  else{
    cost = cross_entropy_cost;
    double total_cost = cost->matrix[0];

    free2d(cost);
    cost = NULL;

    m->iter_cost = total_cost;
  }
}

void backward_pass_L2_LOSS(){
  dARRAY * lgrad1 = divison(m->Y_train,m->current_layer->prev_layer->DENSE->A);

  int act_dims[] = {m->current_layer->prev_layer->DENSE->A->shape[0],m->current_layer->prev_layer->DENSE->A->shape[0]};
  dARRAY * one = ones(act_dims);
  dARRAY * temp1 = subtract(one,m->Y_train);
  dARRAY * temp2 = subtract(one,m->current_layer->prev_layer->DENSE->A);

  free2d(one);
  one = NULL;

  dARRAY * lgrad2 = divison(temp1,temp2);

  free2d(temp1);
  free2d(temp2);
  temp1 = temp2 = NULL;

  dARRAY * temp_local_grad = subtract(lgrad1,lgrad2);
  loss_layer->grad_out = mulScalar(temp_local_grad,-1.0);

  free2d(lgrad1);
  free2d(lgrad2);
  free2d(temp_local_grad);
  lgrad1 = lgrad2 = temp_local_grad = NULL;
}

void (cross_entropy_loss)(){
  loss_layer = (cross_entropy_loss_layer*)malloc(sizeof(cross_entropy_loss_layer));
  loss_layer->cost = 0.0;
  loss_layer->grad_out = NULL;
  loss_layer->forward = forward_pass_L2_LOSS;
  loss_layer->backward = backward_pass_L2_LOSS;
  loss_layer->gnd_truth = NULL;
}