#include <cdnn.h>

int main(){
  
  Create_Model();
  
  int x_train_dims[] = {12288,100};
  int x_cv_dims[] = {12288,100};
  int x_test_dims[] = {12288,100};
  
  int y_train_dims[] = {2,100};
  int y_cv_dims[] = {2,100};
  int y_test_dims[] = {2,100};
  
  dARRAY * x_train = load_x_train("./data/X_train.t7",x_train_dims);
  dARRAY * x_cv = load_x_cv("./data/X_cv.t7",x_cv_dims);
  dARRAY * x_test = load_x_test("./data/X_test.t7",x_test_dims);
  
  dARRAY * y_train = load_y_train("./data/y_train.t7",y_train_dims);
  dARRAY * y_cv = load_y_cv("./data/y_cv.t7",y_cv_dims);
  dARRAY * y_test = load_y_test("./data/y_test.t7",y_test_dims);

  Input(.layer_size=12288);
  Dense(.layer_size=64,.activation="relu",.initializer="he",.layer_type="hidden");
  Dense(.layer_size=32,.activation="relu",.initializer="he",.layer_type="hidden",.dropout=0.7);
  Dense(.layer_size=32,.activation="relu",.layer_type="hidden",.dropout=0.5);
  Dense(.layer_size=16,.activation="relu",.layer_type="hidden");
  Dense(.layer_size=2,.activation="softmax",.initializer="random",.layer_type="output");
  Model(.X_train=x_train,.y_train=y_train,.X_cv=x_cv,.y_cv=y_cv,.X_test=NULL,.y_test=NULL,\
        .epochs=100,.lr=4.67e-5,.optimizer="rmsprop",.checkpoint_every=-1,.batch_size=32);
  
  Load_Model("test.t7");
  
  Fit();
  Test();

  Save_Model("test.t7");

  int img_dims[] = {12288,1};
  dARRAY * test_img1 = load_image("./test_img1.data",img_dims);
  dARRAY * test_img2 = load_image("./test_img2.data",img_dims);

  dARRAY * prediction1 = Predict(test_img1,1);
  dARRAY * prediction2 = Predict(test_img2,1);
  
  free2d(test_img1);
  free2d(test_img2);
  free2d(prediction1);
  free2d(prediction2);

  Destroy_Model();
}