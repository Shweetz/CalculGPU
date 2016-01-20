CC := g++

EXEC := mult_mat_vect

INC := -I/usr/local/cuda/include

LDFLAGS := -Xlinker -rpath=".:"
NVCCLDFLAGS := -Xlinker -rpath=".:"

ifeq ($(DEBUG),yes)
	CFLAGS := -g -pg -Wall -DDEBUG
	NVCCFLAGS := -g -G
else
	CFLAGS := -O2 -Wall
	NVCCFLAGS := -Xptxas -v -O2 --compiler-options -Wall
endif


all: $(EXEC)

mult_mat_vect: ../src/mult_mat_vect.cpp libcommon.so libcudaimpl.so
	$(CC) -o $@ $(CFLAGS) $(INC) "../src/$@.cpp" libcommon.so libcudaimpl.so $(LDFLAGS) 

%: ../src/%.cpp libcommon.so
	$(CC) -o $@ $(CFLAGS) $(INC) "../src/$@.cpp" libcommon.so $(LDFLAGS) 

libcudaimpl.so: ../src/*.cu libcommon.so
	PATH=/usr/local/cuda/bin:$(PATH) LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(LD_LIBRARY_PATH) nvcc --shared  -o libcudaimpl.so  $+  --compiler-options '-fPIC'  $(NVCCLDFLAGS)

libcommon.so: ../src/common.cpp ../src/tools.cpp
	$(CC) -o $@ -shared -fPIC $(CFLAGS) $(INC) $+ $(LDFLAGS) 


.PHONY: clean mrproper tags

clean:
	rm -f $(EXEC) libcommon.so libcudaimpl.so

help:
	@echo
	@echo "make  -> build target in release mode"
	@echo "make DEBUG=yes  -> build target in debug mode"
	@echo "make test  -> to debug Makefile"
	@echo

test:
	@echo CC=$(CC) 
	@echo EXEC=$(EXEC)
	@echo INC=$(INC) 
	@echo LDFLAGS=$(LDFLAGS) 

