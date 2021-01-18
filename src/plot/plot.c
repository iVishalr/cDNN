#include "./plot.h"
#include "../model/model.h"

extern __Model__ * m;

void (plot_train_scores)(){
  printf("plotting\n");
  int iterations = m->num_iter;
  char command[1024] = "python3 ./src/plot/Plot_cost_acc.py ";
  char iter[100];
  sprintf(iter, "%d", iterations);
  strcat(command,iter);
  system(command);
}