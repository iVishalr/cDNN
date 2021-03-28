#include <model.h>
#include <plot.h>
#include <progressbar.h>

extern __Model__ * m;

int main(){
  
  create_model();
  
  int x_train_dims[] = {12288,10000};
  dARRAY * x_train = load_x_train("./data/X_train.t7",x_train_dims);
  
  int y_train_dims[] = {2,10000};
  dARRAY * y_train = load_y_train("./data/y_train.t7",y_train_dims);

  int x_cv_dims[] = {12288,2000};
  dARRAY * x_cv = load_x_cv("./data/X_cv.t7",x_cv_dims);
  
  int y_cv_dims[] = {2,2000};
  dARRAY * y_cv = load_y_cv("./data/y_cv.t7",y_cv_dims);
  
  int x_test_dims[] = {12288,2000};
  int y_test_dims[] = {2,2000};

  dARRAY * x_test = load_x_test("./data/X_test.t7",x_test_dims);
  dARRAY * y_test = load_y_test("./data/y_test.t7",y_test_dims);

  Input(.layer_size=12288);
  Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=32,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=32,.activation="relu",.layer_type="hidden");
  Dense(.layer_size=2,.activation="softmax",.initializer="random",.layer_type="output");
  Model(.x_train=x_train,.Y_train=y_train,.x_cv=x_cv,.Y_cv=y_cv,.x_test=x_test,.Y_test=y_test,\
        .num_iter=200,.learning_rate=4.67e-6,.optimizer="adam",.checkpoint_every=-1,.mini_batch_size=50,\
        .regularization="L2",.lambda=5e-4
       );
      //  .loss="MSELoss"
      // .regularization="L2",.lambda=5e-4,
      //SGD, Momentum - lr = 4.67e-3
  m->fit();
  m->test();

  dARRAY * test_img1 = load_test_image("test_img1.data");
  dARRAY * test_img2 = load_test_image("test_img2.data");

  m->predict(test_img1);
  m->predict(test_img2);
  
  free2d(test_img1);
  free2d(test_img2);
  plot_train_scores();
  destroy_model();
}