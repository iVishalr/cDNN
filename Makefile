#-----------------------
#Compiler Configurations
#-----------------------
CC=gcc-10
CCL=clang 
ATTR= -funroll-loops -O3 -fopenmp -Ofast -ffp-contract=fast
CFLAGS=-c -Wall -Wrestrict

#-----------
#Directories
#-----------
BUILD=build
SRC=src

#----------------
#Code Directories
#----------------
UTILS=utils
TEST=test
ACTIVATIONS=activations
LAYERS=layers
LOSS=loss_functions
OPTIMIZERS=optimizers
MODEL=model
PLOT=plot

#Default target to compile the whole project
all: 
	@if ! test -d $(BUILD); \
		then echo "\033[93msetting up build directory...\033[0m"; mkdir -p build;\
  	fi
	@if ! test -d bin; \
		then echo "\033[93msetting up bin directory...\033[0m"; mkdir -p bin; \
  	fi;
	@$(MAKE) project

.PHONY: targets
targets:
	@echo "Available targets in the Makefile";
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

project:  $(BUILD)/utils.o $(BUILD)/relu.o $(BUILD)/sigmoid.o $(BUILD)/tanh.o $(BUILD)/softmax.o \
$(BUILD)/neural_net.o $(BUILD)/Dense.o $(BUILD)/Input.o $(BUILD)/cross_entropy_loss.o \
$(BUILD)/gradient_descent.o $(BUILD)/adam.o $(BUILD)/adagrad.o $(BUILD)/rmsprop.o \
$(BUILD)/model.o \
$(BUILD)/plot.o $(BUILD)/test_network.o 
	$(CC) $(ATTR) $(BUILD)/utils.o $(BUILD)/relu.o $(BUILD)/sigmoid.o $(BUILD)/tanh.o $(BUILD)/softmax.o \
	$(BUILD)/neural_net.o $(BUILD)/Dense.o $(BUILD)/Input.o $(BUILD)/cross_entropy_loss.o \
	$(BUILD)/gradient_descent.o $(BUILD)/adam.o $(BUILD)/adagrad.o $(BUILD)/rmsprop.o \
	$(BUILD)/model.o $(BUILD)/plot.o $(BUILD)/test_network.o
	@echo "\033[92mBuild Successful\033[0m"
$(BUILD)/utils.o: $(SRC)/$(UTILS)/utils.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/relu.o: $(SRC)/$(ACTIVATIONS)/relu.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/sigmoid.o: $(SRC)/$(ACTIVATIONS)/sigmoid.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/tanh.o: $(SRC)/$(ACTIVATIONS)/tanh.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/softmax.o: $(SRC)/$(ACTIVATIONS)/softmax.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/Dense.o: $(SRC)/$(LAYERS)/Dense.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/Input.o: $(SRC)/$(LAYERS)/Input.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/neural_net.o: $(SRC)/neural_net/neural_net.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/cross_entropy_loss.o: $(SRC)/$(LOSS)/cross_entropy_loss.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/gradient_descent.o: $(SRC)/$(OPTIMIZERS)/gradient_descent.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/adam.o: $(SRC)/$(OPTIMIZERS)/adam.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/adagrad.o: $(SRC)/$(OPTIMIZERS)/adagrad.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/rmsprop.o : $(SRC)/$(OPTIMIZERS)/rmsprop.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/model.o: $(SRC)/$(MODEL)/model.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/plot.o: $(SRC)/$(PLOT)/plot.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
$(BUILD)/test_network.o: $(SRC)/$(TEST)/test_network.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<

%:
	@echo "\033[96mPlease enter a valid target\033[0m";
	@$(MAKE) targets

clean:
	@rm -rf $(BUILD) a.out a.exe
	@echo "\033[92mDone\033[0m"