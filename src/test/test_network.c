#include "../model/model.h"

extern __Model__ * m;

int main(){
  // int feature_dims[] = {19200,200};
  // int output_dims[] = {1,200};
  // int test_prediction[] = {19200,1};
  // dARRAY * test = randn(test_prediction);
  // dARRAY * temp = randn(feature_dims);
  // dARRAY * X = mulScalar(temp,0.01);
  // free2d(temp);
  // temp = NULL;
  // dARRAY * Y = eye(output_dims);
  
  create_model();
  
  int x_train_dims[] = {19200,200};
  load_x_train(x_train_dims);
  load_x_cv(x_train_dims);
  int y_train_dims[] = {1,200};
  load_y_train(y_train_dims);
  load_y_cv(y_train_dims);

  Input(.layer_size=19200,.input_features=m->x_train);
  Dense(.layer_size=20,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=10,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output");
  Model(.x_train=m->x_train,.Y_train=m->Y_train,.x_cv=m->x_cv,.Y_cv=m->Y_cv,.num_iter=5000,.learning_rate=0.008,.lambda=0.05,.regularization="L2");
  m->fit();
  m->save_model("model_cats_vs_dogs5_new.t7");
  plot_train_scores();
  // m->load_model("model1.t7");
  // m->predict(test);
  // free2d(Y);
  printComputation_Graph(m->graph);
  destroy_G(m->graph);
}