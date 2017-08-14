
BIN:=xct
GPU_ARCH=sm_60


$(BIN):proj.o utility.o main.o
	nvcc $^ -o $@ -lm -arch=sm_60

proj.o:proj.c proto.h
	gcc $< -c -o $@

utility.o:utility.c proto.h
	gcc $< -c -o $@ 

main.o:main.cu proj.h proto.h utility.h kernel.cu main.h
	nvcc $< $(DEBUG) -c -o $@ -arch=$(GPU_ARCH)

clean:
	rm -f *.o $(BIN)
