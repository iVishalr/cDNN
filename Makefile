#-----------------------
#Compiler Configurations
#-----------------------
CC=gcc
CCL=clang 
ATTR=-I ./include/ -lm -funroll-loops -O3 -fopenmp -Ofast -ffp-contract=fast --fast-math -I /opt/OpenBLAS/include/ -L/opt/OpenBLAS/lib -lopenblas -lncurses
CFLAGS=-c -Wall -Wrestrict

# 
#-----------
#Directories
#-----------
BUILD=build
SRC=src

#----------------
#Code Directories
#----------------
# UTILS=utils
# TEST=test
# ACTIVATIONS=activations
# LAYERS=layers
# LOSS=loss_functions
# OPTIMIZERS=optimizers
# MODEL=model
# PLOT=plot

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
$(BUILD)/neural_net.o $(BUILD)/Dense.o $(BUILD)/Input.o $(BUILD)/cross_entropy_loss.o $(BUILD)/MSELoss.o\
$(BUILD)/sgd.o $(BUILD)/momentum.o $(BUILD)/adam.o $(BUILD)/adagrad.o $(BUILD)/rmsprop.o \
$(BUILD)/progressbar.o $(BUILD)/statusbar.o \
$(BUILD)/model.o \
$(BUILD)/plot.o $(BUILD)/test_network.o 
	$(CC) $(BUILD)/utils.o $(BUILD)/relu.o $(BUILD)/sigmoid.o $(BUILD)/tanh.o $(BUILD)/softmax.o \
	$(BUILD)/neural_net.o $(BUILD)/Dense.o $(BUILD)/Input.o $(BUILD)/cross_entropy_loss.o $(BUILD)/MSELoss.o \
	$(BUILD)/sgd.o $(BUILD)/momentum.o $(BUILD)/adam.o $(BUILD)/adagrad.o $(BUILD)/rmsprop.o \
	$(BUILD)/model.o $(BUILD)/plot.o $(BUILD)/test_network.o $(BUILD)/progressbar.o $(BUILD)/statusbar.o $(ATTR)
	@echo "\033[92mBuild Successful\033[0m"
$(BUILD)/utils.o: $(SRC)/utils.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/relu.o: $(SRC)/relu.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/sigmoid.o: $(SRC)/sigmoid.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/tanh.o: $(SRC)/tanh.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/softmax.o: $(SRC)/softmax.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/Dense.o: $(SRC)/Dense.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/Input.o: $(SRC)/Input.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/neural_net.o: $(SRC)/neural_net.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/cross_entropy_loss.o: $(SRC)/cross_entropy_loss.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/MSELoss.o: $(SRC)/MSELoss.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/sgd.o: $(SRC)/sgd.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/momentum.o: $(SRC)/momentum.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/adam.o: $(SRC)/adam.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/adagrad.o: $(SRC)/adagrad.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/rmsprop.o : $(SRC)/rmsprop.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/model.o: $(SRC)/model.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/plot.o: $(SRC)/plot.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)

$(BUILD)/progressbar.o: $(SRC)/progressbar.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)
$(BUILD)/statusbar.o: $(SRC)/statusbar.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)

$(BUILD)/test_network.o: $(SRC)/test_network.c
	$(CC) $(CFLAGS) -o $@ $< $(ATTR)

%:
	@echo "\033[96mPlease enter a valid target\033[0m";
	@$(MAKE) targets

clean:
	@rm -rf $(BUILD) a.out a.exe
	@echo "\033[92mDone\033[0m"