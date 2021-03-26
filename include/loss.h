#ifndef LOSS_H
#define LOSS_H

#include "utils.h"

#ifdef __cplusplus
extern "C"{
#endif
  typedef void (*__compute)();
  void (cross_entropy_loss)();
  void backward_pass_L2_LOSS();
  void forward_pass_L2_LOSS();
  void (MSELoss)();
  void backward_pass_MSE_LOSS();
  void forward_pass_MSE_LOSS();
#ifdef __cplusplus
}
#endif

typedef struct loss_layer{
  double cost;
  dARRAY * grad_out;
  __compute forward;
  __compute backward;
}loss_layer;

typedef struct loss_args{
  dARRAY * gnd_truth;
  int isValdiation;
}loss_args;

#define cross_entropy_loss(...) cross_entropy_loss((loss_args){.isValdiation=0,__VA_ARGS__});
#define MSELoss(...) MSELoss((loss_args){.isValdiation=0,__VA_ARGS__});

#endif //LOSS_H