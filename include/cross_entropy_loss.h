#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "utils.h"

#ifdef __cplusplus
extern "C"{
#endif
  typedef void (*__compute)();
  void (cross_entropy_loss)();
  void backward_pass_L2_LOSS();
  void forward_pass_L2_LOSS();
#ifdef __cplusplus
}
#endif

typedef struct cross_entropy_loss_layer{
  double cost;
  dARRAY * grad_out;
  dARRAY * gnd_truth;
  __compute forward;
  __compute backward;
}cross_entropy_loss_layer;

typedef struct cross_entropy_loss_args{
  dARRAY * gnd_truth;
  int isValdiation;
}cross_entropy_loss_args;

#define cross_entropy_loss(...) cross_entropy_loss((cross_entropy_loss_args){.gnd_truth=m->Y_train,.isValdiation=0,__VA_ARGS__});

#endif //CROSS_ENTROPY_LOSS_H