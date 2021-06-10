CC=gcc
CFLAGS=-Wall -Wrestrict -O3 -I ./include/ -funroll-loops -fopenmp -Ofast -ffp-contract=fast --fast-math -I /opt/OpenBLAS/include/ -fpic
LDFLAGS=-L /opt/OpenBLAS/lib -lopenblas -lncurses -lgomp -lm

SRC=src
BUILD=build
SRCS=$(wildcard $(SRC)/*.c)
OBJECTS=$(patsubst $(SRC)/%.c, $(BUILD)/%.o,$(SRCS))
INCLUDE=include/cdnn
INCLUDE_DIR=/usr/local/include
INCLUDE_HEADERS=$(wildcard $(INCLUDE)/*.h)
HEADERS=$(patsubst $(INCLUDE)/%.h,$(INCLUDE_DIR)/cdnn/%.h, $(INCLUDE_HEADERS))

PLATFORM=$(shell uname -s)
SHARED_SUFFIX=dll
LIB_NAME=cdnn

INSTALL_DIR=/usr/local/lib

ifeq "$(PLATFORM)" "Darwin"
    SHARED_SUFFIX=dylib
endif
ifeq "$(PLATFORM)" "Linux"
    SHARED_SUFFIX=so
    INCLUDE_DIR=/usr/include/
    INSTALL_DIR=/usr/lib/
endif


all: 
	@if ! test -d $(BUILD); \
		then echo "\033[93msetting up build directory...\033[0m"; mkdir -p build;\
  	fi
	@if ! test -d bin; \
		then echo "\033[93msetting up bin directory...\033[0m"; mkdir -p bin; \
  	fi;
	@export OPENBLAS_NUM_THREADS=2
	@$(MAKE) project

project: $(OBJECTS)
	$(CC) -fpic -o lib$(LIB_NAME).$(SHARED_SUFFIX) $(LDFLAGS) $(OBJECTS) -shared 
$(BUILD)/%.o: $(SRC)/%.c
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

install: lib$(LIB_NAME).$(SHARED_SUFFIX)
	@install -d $(INSTALL_DIR)
	@install lib$(LIB_NAME).$(SHARED_SUFFIX) $(INSTALL_DIR)
	@install -d $(INCLUDE_DIR)/$(LIB_NAME)
	@install $(INCLUDE)/*.h $(INCLUDE_DIR)/$(LIB_NAME)
	@echo "#include <$(LIB_NAME)/model.h>" > $(INCLUDE_DIR)/$(LIB_NAME).h
	@echo "\033[92mInstalled Successfully!\033[0m"

clean:
	rm -rf $(BUILD) a.out a.exe $(INSTALL_DIR)/lib$(LIB_NAME).$(SHARED_SUFFIX) $(HEADERS) $(INCLUDE_DIR)/$(LIB_NAME).h $(INCLUDE_DIR)/$(LIB_NAME)
