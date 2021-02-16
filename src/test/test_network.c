#include "../model/model.h"

extern __Model__ * m;

int main(){
  
  create_model();
  
  int x_train_dims[] = {12288,100};
  dARRAY * x_train = load_x_train(x_train_dims);

  int x_cv_dims[] = {12288,100};
  dARRAY * x_cv = load_x_cv(x_cv_dims);
  
  int y_train_dims[] = {2,100};
  dARRAY * y_train = load_y_train(y_train_dims);
  
  int y_cv_dims[] = {2,100};
  dARRAY * y_cv = load_y_cv(y_cv_dims);
  
  int x_test_dims[] = {12288,1000};
  int y_test_dims[] = {2,1000};

  dARRAY * x_test = load_x_test(x_test_dims);
  dARRAY * y_test = load_y_test(y_test_dims);

  Input(.layer_size=12288);
  Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=32,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=2,.activation="softmax",.initializer="random",.layer_type="output");
  Model(.x_train=x_train,.Y_train=y_train,.x_cv=x_cv,.Y_cv=y_cv,.x_test=x_test,.Y_test=y_test,\
        .num_iter=3000,.learning_rate=0.007,.optimizer="sgd",.checkpoint_every=-1
       );

  m->fit();
  // m->save_model("DOGS_VS_CATS.t7");
  m->test();
  // m->load_model("DOGS_VS_CATS.t7");

  dARRAY * test_img1 = load_test_image("test_img1.data");
  dARRAY * test_img2 = load_test_image("test_img2.data");
  m->predict(test_img1);
  m->predict(test_img2);
  
  free2d(test_img1);
  free2d(test_img2);
  plot_train_scores();
  destroy_model();
}