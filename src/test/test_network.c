#include "../model/model.h"

extern __Model__ * m;

int main(){
  
  create_model();
  
  int x_train_dims[] = {12288,10};
  load_x_train(x_train_dims);
  int x_cv_dims[] = {12288,10};
  load_x_cv(x_cv_dims);
  int y_train_dims[] = {1,10};
  load_y_train(y_train_dims);
  int y_cv_dims[] = {1,10};
  load_y_cv(y_cv_dims);

  Input(.layer_size=12288);
  Dense(.layer_size=10,.activation="relu",.initializer="he",.layer_type="hidden",.dropout=0.5);
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output");
  Model(.x_train=m->x_train,.Y_train=m->Y_train,.x_cv=m->x_cv,.Y_cv=m->Y_cv,.num_iter=100,.learning_rate=0.009);
  m->fit();
  plot_train_scores();
  destroy_model();
}