#ifndef INPUT_H
#define INPUT_H

#include "../utils/utils.h"

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

#define Input(...) Input((Input_args){.layer_size=20,__VA_ARGS__});

#endif //INPUT_H