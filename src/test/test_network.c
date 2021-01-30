#include "../model/model.h"

extern __Model__ * m;

int main(){
  
  create_model();
  
  int x_train_dims[] = {12288,100};
  load_x_train(x_train_dims);
  int x_cv_dims[] = {12288,100};
  load_x_cv(x_cv_dims);
  int y_train_dims[] = {1,100};
  load_y_train(y_train_dims);
  int y_cv_dims[] = {1,100};
  load_y_cv(y_cv_dims);

  Input(.layer_size=12288);
  // Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
  // Dense(.layer_size=32,.activation="relu",.initializer="he",.layer_type="hidden");
  // Dense(.layer_size=16,.activation="relu",.initializer="he",.layer_type="hidden");
  // Dense(.layer_size=8,.activation="relu",.initializer="he",.layer_type="hidden");
  // Dense(.layer_size=20,.activation="relu",.initializer="he",.layer_type="hidden");
  // Dense(.layer_size=5,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output");
  cross_entropy_loss();
  Model(.x_train=m->x_train,.Y_train=m->Y_train,.x_cv=m->x_cv,.Y_cv=m->Y_cv,.num_iter=5000,.learning_rate=0.05);
  printComputation_Graph(m->graph);
  // m->load_model("model_dogsvscats64.t7");
  m->fit();
  // m->save_model("model_dogsvscats64_with_dropout.t7");
  plot_train_scores();
  printComputation_Graph(m->graph);
  destroy_G(m->graph);
}