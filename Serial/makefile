# This Makefile requires GNU make, which is called gmake on Solaris systems
#
# 'make'        : builds and runs the project
# 'make clean'  : remove build products

PROG = bin/*

OBJS = obj/*

ARCH = $(shell uname -m)
OS   = $(shell uname -s)

CC = g++

LDLIBS  = -L/usr/lib/x86_64-linux-gnu -L/usr/lib -lm -lOpenCL
INCLUDE = -IEigen -ITools -I/usr/include/CL

.PHONY: clean Prac6 run
all:    clean Prac6 run

clean:
	rm -f -r $(PROG) $(OBJS)

Prac6:
	$(CC) $(INCLUDE) -c main.cpp -o obj/main.o
	$(CC) $(INCLUDE) -c Tools/OpenCL_Wrapper.cpp -o obj/OpenCL_Wrapper.o	
	$(CC) $(INCLUDE) -c Tools/Timer.cpp -o obj/Timer.o
	$(CC) -o bin/Prac6 obj/main.o obj/OpenCL_Wrapper.o obj/Timer.o $(LDLIBS)

run:
	bin/Prac6

