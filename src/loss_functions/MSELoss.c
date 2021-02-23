#include "../model/model.h"
#include "./MSELoss.h"

extern __Model__ * m;
mse_loss_layer * loss_layer_mse = NULL;

void forward_pass_MSE_LOSS(){
  //Store the number of training examples in a variable
  int number_of_examples = m->Y_train->shape[1];
  dARRAY * Y = loss_layer_mse->gnd_truth;
  dARRAY * loss = NULL;

  dARRAY * temp_sub = subtract(m->output,Y);
  loss = power(temp_sub,2);

  free2d(temp_sub);
  temp_sub = NULL;

  dARRAY * sum_of_losses = sum(loss,1);

  free2d(loss);
  loss = NULL;

  dARRAY * cost = NULL;
  dARRAY * mse_cost = divScalar(sum_of_losses,(float)(number_of_examples));
  
  free2d(sum_of_losses);
  sum_of_losses = NULL;

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
      reg_cost = m->lambda * layer_manhattan/(2.0*number_of_examples);
    }
  }
  if(m->regularization!=NULL){
    cost = addScalar(mse_cost,reg_cost);
    temp = NULL;

    free2d(mse_cost);
    mse_cost = NULL;

    float total_cost = cost->matrix[0];
    free2d(cost);
    cost = NULL;
    
    m->iter_cost = total_cost;
  }
  else{
    cost = mse_cost;
    float total_cost = cost->matrix[0];
    free2d(cost);
    cost = NULL;
    m->iter_cost = total_cost;
  }
}

void backward_pass_MSE_LOSS(){
  dARRAY * Y = loss_layer_mse->gnd_truth;
  dARRAY * temp = subtract(m->output,Y);
  loss_layer_mse->grad_out = mulScalar(temp,2.0);
  free2d(temp);
  temp = NULL;
}

void (MSELoss)(mse_loss_args args){
  loss_layer_mse = (mse_loss_layer*)malloc(sizeof(mse_loss_layer));
  loss_layer_mse->cost = 0.0;
  loss_layer_mse->grad_out = NULL;
  loss_layer_mse->forward = forward_pass_MSE_LOSS;
  loss_layer_mse->backward = backward_pass_MSE_LOSS;
  loss_layer_mse->gnd_truth = m->Y_train;
  append_graph(loss_layer_mse,"loss");
}