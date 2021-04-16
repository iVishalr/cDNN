#include <cdnn.h>

int main(){
    Create_Model();
    
    int train_x_dims[] = {784,55000};
    int train_y_dims[] = {10,55000};

    int val_x_dims[] = {784,5000};
    int val_y_dims[] = {10,5000};

    int test_x_dims[] = {784,10000};
    int test_y_dims[] = {10,10000};

    dARRAY * x_train = load_x_train("./train/train_x.t7",train_x_dims);
    dARRAY * y_train = load_y_train("./train/train_y.t7",train_y_dims);
    dARRAY * x_val = load_x_cv("./val/val_x.t7",val_x_dims);
    dARRAY * y_val = load_y_cv("./val/val_y.t7",val_y_dims);
    dARRAY * x_test = load_x_test("./test/test_x.t7",test_x_dims);
    dARRAY * y_test = load_y_test("./test/test_y.t7",test_y_dims);

    Input(.layer_size=784);
    Dense(.layer_size=64,.activation="relu",.layer_type="hidden");
    Dense(.layer_size=32,.activation="relu",.layer_type="hidden",.dropout=0.5);
    Dense(.layer_size=32,.activation="relu",.layer_type="hidden",.dropout=0.5);
    Dense(.layer_size=16,.activation="relu",.layer_type="hidden");
    Dense(.layer_size=10,.activation="softmax",.layer_type="output",.initializer="random");
    Model(.X_train=x_train,.y_train=y_train,.X_cv=x_val,.y_cv=y_val,.X_test=x_test,.y_test=y_test,\
    .lr=3.67e-5,.optimizer="momentum",.epochs=1000,.checkpoint_every=100,.batch_size=55,.regularization="L2",.weight_decay=5e-4);
    
    Load_Model("./MODEL.t7");

    Fit();
    Test();

    Destroy_Model();
    return 0;
}