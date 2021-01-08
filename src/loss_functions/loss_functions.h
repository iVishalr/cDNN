#ifndef LOSS_FUNCTIONS_H
#define LOSS_FUNCTIONS_H

#include "../layers/Dense.h"
#include "../utils/utils.h"

#ifdef __cplusplus
extern "C"{
#endif
  double cross_entropy_loss(Dense_layer * layer, dARRAY * Y);
#ifdef __cplusplus
}
#endif
#endif //LOSS_FUNCTIONS_H