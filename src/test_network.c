#include "model.h"

extern __Model__ * m;

int main(){
  
  create_model();
  
  int x_train_dims[] = {12288,100};
  dARRAY * x_train = load_x_train(x_train_dims);
  
  int y_train_dims[] = {1,100};
  dARRAY * y_train = load_y_train(y_train_dims);

  int x_cv_dims[] = {12288,50};
  dARRAY * x_cv = load_x_cv(x_cv_dims);
  
  int y_cv_dims[] = {1,50};
  dARRAY * y_cv = load_y_cv(y_cv_dims);
  
  // int x_test_dims[] = {12288,10};
  // int y_test_dims[] = {1,10};

  // dARRAY * x_test = load_x_test(x_test_dims);
  // dARRAY * y_test = load_y_test(y_test_dims);

  Input(.layer_size=12288);
  Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=32,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=32,.activation="relu",.layer_type="hidden");
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output");
  Model(.x_train=x_train,.Y_train=y_train,.x_cv=x_cv,.Y_cv=y_cv,\
        .num_iter=1000,.learning_rate=4.67e-3,.optimizer="momentum",.checkpoint_every=-1,.mini_batch_size=64,\
       );
      //  .loss="MSELoss"
      // .regularization="L2",.lambda=5e-4,
      //SGD, Momentum - lr = 4.67e-3
// .x_test=x_test,.Y_test=y_test,
  // m->load_model("test_patch2.t7");
  // shape(m->x_train_mini_batch[1]);
  // dump_image(m->y_train_mini_batch[1]);
  m->fit();
  // m->save_model("test_patch3.t7");
  // m->test();

  dARRAY * test_img1 = load_test_image("test_img1.data");
  dARRAY * test_img2 = load_test_image("test_img2.data");

  m->predict(test_img1);
  m->predict(test_img2);
  
  free2d(test_img1);
  free2d(test_img2);
  // plot_train_scores();
  destroy_model();
}