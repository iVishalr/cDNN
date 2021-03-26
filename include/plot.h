#ifndef PLOT_H
#define PLOT_H

#ifdef __cplusplus
extern "C"{
#endif
  void (plot_train_scores)();
#ifdef __cplusplus
}
#endif

#define plot_train_scores(...) plot_train_scores()

#endif //PLOT_H