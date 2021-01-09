CC=gcc-10
CCL=clang 
ATTR= -funroll-loops -O3 -fopenmp
CFLAGS=-c -Wall -Wrestrict
BUILD=build
SRC=src
UTILS=utils
TEST=test
ACTIVATIONS=Activations
RELU=relu
SIGMOID=sigmoid
TANH=tanh
LAYERS=layers
LOSS=loss_functions
OPTIMIZERS=optimizers
all: 
	@if ! test -d $(BUILD); \
		then echo "\033[93msetting up build directory...\033[0m"; mkdir -p build;\
  	fi
	@if ! test -d bin; \
		then echo "\033[93msetting up bin directory...\033[0m"; mkdir -p bin; \
  	fi;
	@$(MAKE) start
# $(BUILD)/test_utils.o	

start:  $(BUILD)/utils.o $(BUILD)/relu.o $(BUILD)/sigmoid.o $(BUILD)/tanh.o $(BUILD)/neural_net.o $(BUILD)/Dense.o $(BUILD)/Input.o $(BUILD)/loss_functions.o $(BUILD)/gradient_descent.o $(BUILD)/test_network.o
	$(CC) $(ATTR) $(BUILD)/utils.o $(BUILD)/relu.o $(BUILD)/sigmoid.o $(BUILD)/tanh.o $(BUILD)/neural_net.o $(BUILD)/Dense.o $(BUILD)/Input.o $(BUILD)/loss_functions.o $(BUILD)/gradient_descent.o $(BUILD)/test_network.o
	@echo "\033[92mBuild Successful\033[0m"
	@echo "\033[92mCompiled Test\033[0m"
$(BUILD)/utils.o: $(SRC)/$(UTILS)/utils.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/relu.o: $(SRC)/$(UTILS)/$(ACTIVATIONS)/$(RELU)/relu.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/sigmoid.o: $(SRC)/$(UTILS)/$(ACTIVATIONS)/$(SIGMOID)/sigmoid.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/tanh.o: $(SRC)/$(UTILS)/$(ACTIVATIONS)/$(TANH)/tanh.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/Dense.o: $(SRC)/$(LAYERS)/Dense.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/Input.o: $(SRC)/$(LAYERS)/Input.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/neural_net.o: $(SRC)/neural_net/neural_net.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/loss_functions.o: $(SRC)/$(LOSS)/loss_functions.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/gradient_descent.o: $(SRC)/$(OPTIMIZERS)/gradient_descent.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/test_network.o: $(SRC)/$(TEST)/test_network.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
clean:
	@rm -rf $(BUILD) a.out a.exe
	@echo "\033[92mDone\033[0m"

# -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib