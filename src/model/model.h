#ifndef MODEL_H
#define MODEL_H

#include "../neural_net/neural_net.h"

typedef struct model_args{
  dARRAY * x_train;
  dARRAY * Y_train;
  dARRAY * x_cv;
  dARRAY * Y_cv;
  dARRAY * x_test;
  dARRAY * Y_test;
  double learning_rate;
  int mini_batch_size;
  int num_iter;
  char * optimizer;
  char * loss;
  char * regularization;
  double lambda;
  int print_cost;
}Model_args;

typedef struct weight_arr{
  dARRAY * weight;
}__weights__;

typedef struct bias_arr{
  dARRAY * bias;
}__biases__;

typedef struct save_model{
  __weights__ * weights_arr[200];
  __biases__ * biases_arr[200];
  double learning_rate;
  double lambda;
  int num_layers;
}SAVE_MODEL;

#ifdef __cplusplus
extern "C"{
#endif
  typedef void (*__model_init__)();
  typedef void (*__model_fit__)();
  typedef void (*__model_predict__)(dARRAY * input_feature);
  typedef void (*__model_save__)(char * file_name);
  typedef void (*__model_load__)();
  typedef void (*__model_summary__)();
  typedef void (*__model_forward__)();
  typedef void (*__model_backward__)();
  void __initialize_params__();
  void (create_model)();
  void (Model)(Model_args model_args);
  void __init__();
  void __forward__();
  void __backward__();
  void __fit__();
  void __load_model__();
  void __save_model__();
  void __summary__();
  void __predict__();
  void (destroy_model)();
#ifdef __cplusplus
}
#endif

typedef struct model{
  Computation_Graph * graph;
  Computation_Graph * current_layer;
  int number_of_layers;

  double learning_rate;
  int mini_batch_size;
  dARRAY * x_train;
  dARRAY * Y_train;
  dARRAY * x_test;
  dARRAY * Y_test;
  dARRAY * x_cv;
  dARRAY * Y_cv;
  dARRAY * output;
  int print_cost;
  int num_iter;
  char * optimizer;
  char * regularization;
  double lambda;
  char * loss;

  int input_size;
  int output_size;
  long int total_parameters;
  double train_cost;
  double train_accuracy;
  double test_accuracy;
  double cross_val_accuracy;
  int predicting;
  dARRAY * model_layer_weights;
  dARRAY * model_layer_biases;
  __model_init__ init;
  __model_fit__ fit;
  __model_predict__ predict;
  __model_save__ save_model;
  __model_load__ load_model;
  __model_summary__ summary;
  __model_forward__ forward;
  __model_backward__ backward;
}__Model__;

// __Model__ * m;

#define create_model(...) create_model();

#define Model(...) Model((Model_args){\
.x_train=NULL,.Y_train=NULL,\
.x_cv=NULL,.Y_cv=NULL,\
.x_test=NULL,.Y_test=NULL,\
.num_iter=10,\
.mini_batch_size=128,\
.optimizer="Adam",\
.regularization=NULL,\
.lambda=0.0,\
.learning_rate=3e-4,\
.print_cost=1,\
.loss="cross_entropy_loss",__VA_ARGS__});

#define destroy_model(...) destroy_model();

#endif