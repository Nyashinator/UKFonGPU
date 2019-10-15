#include <stdlib.h>
#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/SVD"
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <iostream>

//------------------------------------------------------------------------------
#include "Timer.h"
#include "OpenCL_Wrapper.h"
//------------------------------------------------------------------------------
#define n 5
#define m 2
//#define S 3
//FUNCTION PROTOTYPES
void Process_OpenCL();
//MATRICES

// Useful globals
float dt,b,b0,h0,gm0,r0, D, r, G, V,range,angle;
float c,alpha,ki,beta,lambda,Wm,Wc;
//------------------------------------------------------------------------------
// Useful globals
int    N;
size_t LocalSize[2] = {1, 1};
//------------------------------------------------------------------------------

// CPU Memory Handles
float* A;
float* B;
float* L;
float* S;
float* T;
float* W;
float* tempx;
float* tempy;
float* Output_OpenCL;
float* Output_OpenCL1;
float* Output_OpenCL2;
float* Output_OpenCL3;
float* Output_OpenCL4;
//------------------------------------------------------------------------------

// GPU Memory Handles
//--0
cl_mem A_Buffer;
cl_mem B_Buffer;
cl_mem L_Buffer;
cl_mem OutputBuffer;
//--1
cl_mem S_Buffer;
cl_mem T_Buffer;
cl_mem OutputBuffer1;
//--2
cl_mem OutputBuffer2;
//--3
cl_mem W_Buffer;
cl_mem OutputBuffer3;
//--4
cl_mem OutputBuffer4;
//------------------------------------------------------------------------------
// OpenCL event handles

