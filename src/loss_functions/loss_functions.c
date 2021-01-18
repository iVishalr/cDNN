#include "../model/model.h"
#include "./loss_functions.h"

extern __Model__ * m;

double cross_entropy_loss(dARRAY * output, dARRAY * Y){
  int number_of_examples = Y->shape[1];
  
  dARRAY * log_y_hat= NULL;
  log_y_hat = (dARRAY *)malloc(sizeof(dARRAY));
  log_y_hat->matrix = (double*)calloc(output->shape[0]*output->shape[1],sizeof(double));
  //calculate log(y^)
  for(int i=0;i<output->shape[0]*output->shape[1];i++){
    log_y_hat->matrix[i] = log(output->matrix[i]);
  }
  log_y_hat->shape[0] = output->shape[0];
  log_y_hat->shape[1] = output->shape[1];

  int dims[] = {output->shape[0],output->shape[1]};
  dARRAY * temp_ones = ones(dims);
  dARRAY * temp_sub = subtract(temp_ones,output);
  
  dARRAY * log_one_y_hat = NULL;
  log_one_y_hat = (dARRAY *)malloc(sizeof(dARRAY));
  log_one_y_hat->matrix = (double*)calloc(output->shape[0]*output->shape[1],sizeof(double));
  //calculate log(1-y^)
  for(int i=0;i<output->shape[0]*output->shape[1];i++){
    log_one_y_hat->matrix[i] = log(temp_sub->matrix[i]);
  }
  log_one_y_hat->shape[0] = output->shape[0];
  log_one_y_hat->shape[1] = output->shape[1];

  free2d(temp_sub);
  temp_sub = NULL;
  //calculate y*log(y^)
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
    if(!strcmp(m->regularization,"L2")){
      double layer_frobenius = 0.0;
      while(temp!=NULL){
        layer_frobenius += frobenius_norm(temp->DENSE->weights);
        temp = temp->next_layer;
      }
      reg_cost = m->lambda*layer_frobenius/(2*number_of_examples);
    }
    else if(!strcmp(m->regularization,"L1")){
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
    return total_cost;
  }
  else{
    cost = cross_entropy_cost;
    double total_cost = cost->matrix[0];

    free2d(cost);
    cost = NULL;
    
    return total_cost;
  }
}