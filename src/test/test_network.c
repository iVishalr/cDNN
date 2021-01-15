#include "../model/model.h"

extern __Model__ * m;

int main(){
  int feature_dims[] = {12288,209};
  int output_dims[] = {1,209};
  int test_prediction[] = {12288,1};
  dARRAY * test = randn(test_prediction);
  dARRAY * temp = randn(feature_dims);
  dARRAY * X = mulScalar(temp,0.01);
  free2d(temp);
  temp = NULL;
  dARRAY * Y = eye(output_dims);
  create_model();
  Input(.layer_size=12288,.input_features=X);
  Dense(.layer_size=50,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=10);
  Dense(.layer_size=5);
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output");
  Model(.x_train=X,.Y_train=Y,.num_iter=100,.learning_rate=3e-4);
  int x_train_dims[] = {9,2};
  load_x_train(x_train_dims);
  // m->fit();
  // m->save_model("model1.t7");
  // m->load_model("model1.t7");
  // m->predict(test);
  free2d(Y);
  printComputation_Graph(m->graph);
  destroy_G(m->graph);
}