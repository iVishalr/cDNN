 /** 
  * File:    model.h 
  * 
  * Author:  Vishal R (vishalr@pesu.pes.edu or vishalramesh01@gmail.com) 
  * 
  * Summary of File: 
  *   This file contains all the function headers and model objects. 
  *   Functions allow the user of the software to create simple Artificial Neural Networks in C. 
  */ 

#ifndef MODEL_H
#define MODEL_H

#include "neural_net.h"
#include "utils.h"

typedef struct model_args{
  dARRAY * X_train;
  dARRAY * y_train;
  dARRAY * X_cv;
  dARRAY * y_cv;
  dARRAY * X_test;
  dARRAY * y_test;
  float lr;
  int batch_size;
  int epochs;
  char * optimizer;
  char * loss;
  char * regularization;
  float weight_decay;
  int print_cost;
  float beta;
  float beta1;
  float beta2;
  int checkpoint_every;
}Model_args;

#ifdef __cplusplus
extern "C"{
#endif
  typedef void (*__model_init__)();
  typedef void (*__model_fit__)();
  typedef void (* __model_predict__)(dARRAY * input_feature);
  typedef void (*__model_save__)(char * file_name);
  typedef void (*__model_load__)();
  typedef void (*__model_summary__)();
  typedef void (*__model_forward__)();
  typedef void (*__model_backward__)();
  typedef void (*__model_test__)();
  void __initialize_params__();
  void (Create_Model)();
  void (Model)(Model_args model_args);
  void __init__();
  void __forward__();
  void __backward__();
  void __fit__();
  void __load_model__();
  void __save_model__();
  void __summary__();
  dARRAY * __predict__();
  void __test__();
  float calculate_accuracy(dARRAY * predicted, dARRAY * gnd_truth);
  float calculate_test_val_acc(dARRAY * input_features,dARRAY * gnd_truth);
  dARRAY * relu_val(dARRAY * linear_matrix);
  dARRAY * sigmoid_val(dARRAY * linear_matrix);
  dARRAY * tanh_val(dARRAY * linear_matrix);
  dARRAY * softmax_val(dARRAY * linear_matrix);
  dARRAY * load_x_train(char * filename, int * dims);
  dARRAY * load_y_train(char * filename, int * dims);
  dARRAY * load_x_cv(char * filename, int * dims);
  dARRAY * load_y_cv(char * filename, int * dims);
  dARRAY * load_x_test(char * filename, int * dims);
  dARRAY * load_y_test(char * filename, int * dims);
  dARRAY * Predict(dARRAY * input_feature,int verbose);
  void Fit();
  void Test();
  void Load_Model(char * filename);
  void Save_Model(char * filename);
  void early_stopping_handler(int num);
  void create_mini_batches();
  void dump_to_file(float * arr ,char * filename,char * mode);
  void dump_image(dARRAY * images,char * filename);
  dARRAY * load_image(char * filename,int * dims);
  void (Destroy_Model)();
#ifdef __cplusplus
}
#endif

typedef struct model{
  Computation_Graph * graph;
  Computation_Graph * current_layer;
  
  int number_of_layers;

  float learning_rate;
  int mini_batch_size;
  int num_mini_batches;
  int current_mini_batch;

  dARRAY * x_train;
  dARRAY * Y_train;
  dARRAY * x_test;
  dARRAY * Y_test;
  dARRAY * x_cv;
  dARRAY * Y_cv;
  dARRAY ** x_train_mini_batch;
  dARRAY ** y_train_mini_batch;
  dARRAY * output;

  int num_of_training_examples;
  int print_cost;
  int num_iter;
  char * optimizer;
  char * regularization;
  float lambda;
  char * loss;
  float beta;
  float beta1;
  float beta2;
  float epsilon;
  int time_step;

  dARRAY * m_t_dW[1024];
  dARRAY * v_t_dW[1024];
  dARRAY * m_t_db[1024];
  dARRAY * v_t_db[1024];
  dARRAY * cache_dW[1024];
  dARRAY * cache_db[1024];

  int input_size;
  int output_size;
  long int total_parameters;
  float train_cost;
  float iter_cost;
  float train_accuracy;
  float test_accuracy;
  float cross_val_accuracy;
  int predicting;
  int testing;
  int ckpt_every;
  int current_iter;

  __model_init__ init;
  __model_fit__ fit;
  __model_predict__ predict;
  __model_test__ test;
  __model_save__ save_model;
  __model_load__ load_model;
  __model_summary__ summary;
  __model_forward__ forward;
  __model_backward__ backward;
}__Model__;

/**! 
 * @brief  Function : create_model - Constructor that creates a model object that is responsible for the overall functioning of the network.
 * 
 * @return void
*/
#define Create_Model(...) Create_Model();

/**! 
 * @brief  Function Model - Constructor that defines the model parameters and constructs 
 * and initializes the layers with the arguments specified.
 * 
 * @param X_train dARRAY pointing to training set (X_train)
 * @param y_train dARRAY pointing to the labels for training set (y_train) 
 * @param X_cv dARRAY pointing to validation set (X_val)
 * @param y_cv dARRAY pointing to the labels for validation set (y_val)
 * @param X_test dARRAY pointing to test set (X_test)
 * @param y_test dARRAY pointing to the labels for test set. (y_test) 
 * 
 * @param epochs Number of epochs the model must train for. (Hyperparameter)
 * @param batch_size Specifies the batch_size to be used for training. (Hyperparameter) 
 * @param lr Specifies the learning rate for the model. (Hyperparameter)
 * @param weight_decay Specifies the weight decay to be used in regularization. (Hyperparameter) 
 * @param beta Specifies the decay value that will be used during parameter updates. (Hyperparameter) 
 * @param beta1 Specifies the decay value that will be used during parameter updates. (Hyperparameter) 
 * @param beta2 Specifies the decay value that will be used during parameter updates. (Hyperparameter) 
 * 
 * @param loss Specifies the loss function to be used.  
 * @param optimizer Specifies the optimizer to be used for training. 
 * @param regularization Specifies the type of regularization to be applied during training. 
 * @param checkpoint_every Specifies the interval at which the model will be saved.
*/
#define Model(...) Model((Model_args){\
.X_train=NULL,.y_train=NULL,\
.X_cv=NULL,.y_cv=NULL,\
.X_test=NULL,.y_test=NULL,\
.epochs=10,\
.batch_size=64,\
.optimizer="Adam",\
.regularization=NULL,\
.weight_decay=0.0,\
.lr=3e-4,\
.print_cost=1,\
.beta=0.9,\
.beta1=0.9,\
.beta2=0.999,\
.loss="cross_entropy_loss",\
.checkpoint_every=2500,__VA_ARGS__});

/**! 
 * @brief  Function : destroy_model - Constructor that is responsible for deallocating memory
 * assigned to the model.
 * @return void
*/
#define Destroy_Model(...) Destroy_Model();

#endif