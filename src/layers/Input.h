#ifndef INPUT_H
#define INPUT_H

#include "../utils/utils.h"

typedef struct input_layer{
  int input_features_size;
  dARRAY * A;
  int isTraining;
}Input_layer;

typedef struct input_args{
  dARRAY * input_features;
  int layer_size;
}Input_args; 

#define Input(...) Input((Input_args){.layer_size=20,.input_features=NULL,__VA_ARGS__});

#ifdef __cplusplus
extern "C"{
#endif
  void (Input)(Input_args input_layer_args);
#ifdef __cplusplus
}
#endif
#endif //INPUT_H