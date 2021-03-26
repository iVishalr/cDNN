#ifndef MSELOSS_H
#define MSELOSS_H

#include "utils.h"

#ifdef __cplusplus
extern "C"{
#endif
  typedef void (*__compute)();
  void (MSELoss)();
  void backward_pass_MSE_LOSS();
  void forward_pass_MSE_LOSS();
#ifdef __cplusplus
}
#endif

typedef struct mse_loss_layer{
  double cost;
  dARRAY * grad_out;
  dARRAY * gnd_truth;
  __compute forward;
  __compute backward;
}mse_loss_layer;

typedef struct mse_loss_args{
  dARRAY * gnd_truth;
  int isValdiation;
}mse_loss_args;

#define MSELoss(...) MSELoss((mse_loss_args){.gnd_truth=m->Y_train,.isValdiation=0,__VA_ARGS__});

#endif //MSELOSS_H