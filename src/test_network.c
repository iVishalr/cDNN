#include <model.h>
#include <plot.h>

extern __Model__ * m;

int main(){
  
  create_model();
  
  int x_train_dims[] = {12288,100};
  dARRAY * x_train = load_x_train("./data/X_train.data",x_train_dims);
  
  int y_train_dims[] = {1,100};
  dARRAY * y_train = load_y_train("./data/y_train.data",y_train_dims);

  int x_cv_dims[] = {12288,100};
  dARRAY * x_cv = load_x_cv("./data/X_cv.data",x_cv_dims);
  
  int y_cv_dims[] = {1,100};
  dARRAY * y_cv = load_y_cv("./data/y_cv.data",y_cv_dims);
  
  int x_test_dims[] = {12288,100};
  int y_test_dims[] = {1,100};

  dARRAY * x_test = load_x_test("./data/X_test.data",x_test_dims);
  dARRAY * y_test = load_y_test("./data/y_test.data",y_test_dims);

  Input(.layer_size=12288);
  Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=32,.activation="relu",.initializer="he",.layer_type="hidden",.dropout=0.7);
  Dense(.layer_size=32,.activation="relu",.layer_type="hidden",.dropout=0.5);
  Dense(.layer_size=16,.activation="relu",.layer_type="hidden");
  Dense(.layer_size=1,.activation="sigmoid",.initializer="random",.layer_type="output");
  Model(.X_train=x_train,.y_train=y_train,.X_cv=x_cv,.y_cv=y_cv,.X_test=x_test,.y_test=y_test,\
        .epochs=1000,.lr=4.67e-5,.optimizer="adam",.checkpoint_every=-1,.batch_size=32);
  
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