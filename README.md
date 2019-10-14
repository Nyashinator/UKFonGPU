# Implementation of the Unscented Kalman Filter on a GPU
Using OpenCL and Eigen Linear algebra library for c++

Notes on Toolchain:

* Environment (IDE): Code::Blocks 17.12
*	OpenCL Compiler  : Intel® CPU runtime for OpenCL™ applications 18.1

Important:

* There is a CPU/Serial implementation of this code and an OpenCL/Parallel implementation. 
* You need to include the Eigen libraries for linear algebra for both the CPU only and the OpenCL implementaion as all matrix manipulations are done using Eigen.
* For the OpenCL/Parallel implementation, you have to include the CL header files and install a compiler for it (CUDA comes with OpenCL 1.2)
* The OpenCL version was implemented using single precision. If you device supports OpenCL double precision it is in your best interest to change all the float and float* to double and double* (this must of course be done with care and understanding.

Useful Reference sites:
OpenCL: https://www.khronos.org/opencl/

Eigen : http://eigen.tuxfamily.org/index.php?title=Main_Page#Documentation


