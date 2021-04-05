 /** 
  * File:    Input.h 
  * 
  * Author:  Vishal R (vishalr@pesu.pes.edu or vishalramesh01@gmail.com) 
  * 
  * Summary of File: 
  *   This file contains all the function headers required for the implementation of Input Layer. 
  */ 

#ifndef INPUT_H
#define INPUT_H

#include "utils.h"

typedef struct input_args{
  int layer_size;
}Input_args; 

#ifdef __cplusplus
extern "C"{
#endif
  typedef void (*__compute)();
  void (Input)(Input_args input_layer_args);
#ifdef __cplusplus
}
#endif

typedef struct input_layer{
  int input_features_size;
  dARRAY * A;
  int isTraining;
  __compute forward;
  __compute backward;
}Input_layer;

/**!
* @brief Function : Input - Constructor that constructs the Input Layer.
* @param layer_size Specifies the number of nodes in the layer which is equivalent to number of features in training set.
* @return void
*/ 
#define Input(...) Input((Input_args){.layer_size=20,__VA_ARGS__});

#endif //INPUT_H