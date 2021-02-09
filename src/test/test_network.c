#include "../model/model.h"

extern __Model__ * m;

int main(){
  
  create_model();
  
  int x_train_dims[] = {12288,4000};
  load_x_train(x_train_dims);
  int x_cv_dims[] = {12288,100};
  load_x_cv(x_cv_dims);
  int y_train_dims[] = {1,4000};
  load_y_train(y_train_dims);
  int y_cv_dims[] = {1,100};
  load_y_cv(y_cv_dims);
  
  int x_test_dims[] = {12288,1000};
  int y_test_dims[] = {1,1000};

  load_x_test(x_test_dims);
  load_y_test(y_test_dims);

  Input(.layer_size=12288);
  Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden",.dropout=0.5);
  Dense(.layer_size=32,.activation="relu",.initializer="he",.layer_type="hidden",.dropout=0.5);
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output");
  Model(.x_train=m->x_train,.Y_train=m->Y_train,.x_cv=m->x_cv,.Y_cv=m->Y_cv,.x_test=m->x_test,.Y_test=m->Y_test,.num_iter=2000,.learning_rate=0.007);
  m->fit();
  m->save_model("DOGS_VS_CATS.t7");
  m->test();
  plot_train_scores();
  destroy_model();
}