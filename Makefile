CC=gcc-10
CCL=clang 
ATTR= -funroll-loops -O3 -fopenmp
CFLAGS=-c -Wall -Wrestrict
BUILD=build
SRC=src
UTILS=utils
TEST=test
all: 
	@if ! test -d $(BUILD); \
		then echo "\033[93msetting up build directory...\033[0m"; mkdir -p build;\
  	fi
	@if ! test -d bin; \
		then echo "\033[93msetting up bin directory...\033[0m"; mkdir -p bin; \
  	fi;
	@$(MAKE) start	
start: $(BUILD)/test_utils.o $(BUILD)/utils.o
	$(CC) $(ATTR) $(BUILD)/test_utils.o $(BUILD)/utils.o
	@echo "\033[92mBuild Successful\033[0m"
$(BUILD)/test_utils.o: $(SRC)/$(TEST)/test_utils.c
	$(CC) $(CFLAGS) -o $@ $<
	@echo "\033[92mCompiled Test\033[0m"
$(BUILD)/utils.o: $(SRC)/$(UTILS)/utils.c
	$(CC) $(CFLAGS) $(ATTR) -o $@ $<
clean:
	@rm -rf $(BUILD) a.out a.exe
	@echo "\033[92mDone\033[0m"

# -Xpreprocessor -fopenmp -I/usr/local/include -L/usr/local/lib