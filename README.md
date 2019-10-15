# Implementation of the Unscented Kalman Filter on a GPU
Using OpenCL and Eigen Linear algebra library for c++

Notes on Compilation using code::blocks

* The Eigen library is double compressed unzip it twice.
* After unzipping it change the name of the Eigen folder from "eigen-eigen-323c052e1731" to "Eigen" without the quotation marks.
* Copy the Folder and place it in your compiler's include folder
* Copy the CL folder as well and paste it in your compiler's include folder

Compilation on Linux:
* make sure you have installed the latest version of OpenCL, drivers for your devices, make and GNU GCC compiler
* Once everything is setup simple open the linux terminal from the folder where main is ans type make and press enter. If everything is setup right the program should compile and run.

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


