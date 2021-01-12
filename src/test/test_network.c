#include "../model/model.h"

extern __Model__ * m;

int main(){
  int feature_dims[] = {12288,209};
  int output_dims[] = {1,209};
  int test_prediction[] = {12288,1};
  dARRAY * test = randn(test_prediction);
  dARRAY * X = randn(feature_dims);
  dARRAY * Y = ones(output_dims);
  create_model();
  Input(.layer_size=12288,.input_features=X,.layer_num=1);
  Dense(.layer_size=5,.activation="relu",.initializer="he",.layer_type="hidden",.layer_num=2);
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output",.layer_num=3);
  Model(.x_train=X,.Y_train=Y,.num_iter=2500);
  // m->fit();
  // sleep(1000);
  // m->save_model("blala");
  m->load_model();
  printf("in main\n");
  // printf("%d\n",m->number_of_layers);
  // m->predict(test);
  free2d(Y);
  printComputation_Graph(m->graph);
  destroy_G(m->graph);
}