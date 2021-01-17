#include "../model/model.h"

extern __Model__ * m;

int main(){
  int feature_dims[] = {19200,200};
  int output_dims[] = {1,200};
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
  int y_train_dims[] = {1,200};
  load_y_train(y_train_dims);
  Input(.layer_size=19200,.input_features=m->x_train);
  // Dense(.layer_size=50,.activation="relu",.initializer="he",.layer_type="hidden");
  // Dense(.layer_size=10);
  Dense(.layer_size=5);
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output");
  Model(.x_train=m->x_train,.Y_train=m->Y_train,.num_iter=2500,.learning_rate=0.005,.lambda=0.003,.regularization="L2");
  m->fit();
  m->save_model("model_cats_vs_dogs2.t7");
  // m->load_model("model1.t7");
  // m->predict(test);
  // free2d(Y);
  printComputation_Graph(m->graph);
  destroy_G(m->graph);
}