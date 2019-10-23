#include <stdlib.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <iostream>
#include <fstream>
#include "main.h"

using namespace std ;
using namespace Eigen;

//------------------------------------------------------------------------------
//GLOBAL MATRICES
//------------------------------------------------------------------------------
//Matrix <float, n, 1> x;
//Matrix <float, n, 1> s;
//------------------------------------------------------------------------------
// FUNCTIONS
//------------------------------------------------------------------------------
void csv_write(const char* filename, float* mat, int r, int c)
{
    fstream fs;
    fs.open(filename, fstream::in | fstream::out | fstream::app);

    //fs << "Step(k),x\n";
    for(int j=0; j<c; j++)
    {
        for(int i=0; i<r; i++)
        {
            fs << mat[i*c+j]<< ",";
        }
        fs << endl;
    }

    fs.close();
}

void clear_text(const char* filename)
{
    fstream fs;
    fs.open(filename, fstream::in | fstream::out | fstream::trunc);
    fs.close();
}

Matrix <float,n,1> state(Matrix<float,n,1> x, float t)
{
    Matrix <float, n, 1> s;
    Matrix <float, n, 1> vn;
    vn = Matrix <float, n, 1>::Random();

//Constants
    gm0 = 3.9860e5;
    b0 = -0.59783;
    h0 = 13.406;
    r0 = 6374;

// Variables
    b = -b0*exp(x(4,0));
    r = sqrt(pow(x(0,0),2)+pow(x(1,0),2));
    V = sqrt(pow(x(2,0),2)+pow(x(3,0),2));
    G = -(gm0)/(pow(r,3));
    D = -b*exp((r0-r)/h0)*V;


    s(0,0) = x(0,0)+x(2,0)*t;
    s(1,0) = x(1,0)+x(3,0)*t;
    s(2,0) = x(2,0)+(D*x(2,0)+G*x(0,0)+vn(0,0))*t;
    s(3,0) = x(3,0)+(D*x(3,0)+G*x(1,0)+vn(1,0))*t;
    s(4,0) = vn(2,0);
    
    return s;
}
//-----------------------------------------------------------------------------
Matrix <float,m,1> measure(Matrix<float,n,1> x)
{
    Matrix <float, m, 1> wn;
    Matrix <float, m, 1> y;
    wn = Matrix <float, m, 1>::Random();

    y(0,0) = sqrt(pow((x(0,0)-r0),2)+pow(x(1,0),2))+wn(0,0);
    y(1,0) = atan(x(1,0)/(x(0,0)-r0))+wn(1,0);

    return y;
}
//-----------------------------------------------------------------------------
Matrix<float,n,2*n+1> sgmp(Matrix<float,n,1> x, Matrix<float,n,n> B, float g)
{
    Matrix<float,n,n> E;
    Matrix<float,n,2*n+1> X;
    Matrix<float,n,n> Y;
//cholesky
    LLT<Matrix<float,n,n>> lltOfA(B*sqrt(g)); //the Cholesky decomposition of P
    E = lltOfA.matrixL(); // retrieve factor E
//make sigma matrix
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            X(j,i+1) = x(j,0)+E(j,i);
            X(j,i+n+1) = x(j,0)-E(j,i);
        }
        X(i,0) = x(i,0);
    }
    return X;
}
//-----------------------------------------------------------------------------
Matrix<float,n,2*n+1> utransformX(Matrix<float,n,2*n+1> B)
{
    Matrix<float,n,2*n+1> X_t;
    Matrix<float,n,1> tmp;

    for(int i=0; i<2*n+1; i++)
    {
        for(int j=0; j<n; j++)
        {
            tmp(j,0) = B(j,i);
        }
        tmp = state(tmp,dt);
        for(int l=0; l<n; l++)
        {
            X_t(l,i) = tmp(l,0);
        }
    }
    return X_t;
}
//------------------------------------------------------------------------------
Matrix<float,m,2*n+1> utransformY(Matrix<float,n,2*n+1> B)
{
    Matrix<float,m,2*n+1> Y_t;
    Matrix<float,n,1> tmp;
    Matrix<float,m,1> tmp2;

    for(int i=0; i<2*n+1; i++)
    {
        for(int j=0; j<n; j++)
        {
            tmp(j,0) = B(j,i);
        }
        tmp2 = measure(tmp);
        for(int l=0; l<m; l++)
        {
            Y_t(l,i) = tmp2(l,0);
        }
    }
    return Y_t;
}
//-----------------------------------------------------------------------------
// GPU reset?  See <http://stackoverflow.com/questions/12259044/
//                  limitations-of-work-item-load-in-gpu-cuda-opencl>

// <https://social.technet.microsoft.com/Forums/windows/en-US/
// 92a45329-3dd1-4c42-8a53-42dd232edd81/
// how-to-turn-off-timeout-detection-and-recovery-of-gpus>
//------------------------------------------------------------------------------
void Process_OpenCL()
{
//printf("\n");

    OpenCL_PrepareLocalSize(N, LocalSize);

    OpenCL_ConstantInt(10, N); //take N as argument
    OpenCL_ConstantDouble(11, dt);
    OpenCL_ConstantDouble(12, r0);
    OpenCL_ConstantDouble(13, gm0);
    OpenCL_ConstantDouble(14, b0);
    OpenCL_ConstantDouble(15, h0);

    OpenCL_WriteData(A_Buffer, N*sizeof(float), A);
    OpenCL_WriteData(B_Buffer, N*N*sizeof(float), B);
    OpenCL_WriteData(L_Buffer, N*N*sizeof(float), L);
    OpenCL_WriteData(T_Buffer, (2*N*N+N)*sizeof(float), T);
    OpenCL_WriteData(W_Buffer, (2*N*N+N)*sizeof(float), W);

    OpenCL_Run(N, LocalSize);

    OpenCL_ReadData(OutputBuffer, (2*N*N+N)*sizeof(float), Output_OpenCL);
    OpenCL_ReadData(OutputBuffer1, (2*N*N+N)*sizeof(float), Output_OpenCL1);
    OpenCL_ReadData(OutputBuffer2, (2*N*N+N)*sizeof(float), Output_OpenCL2);
    OpenCL_ReadData(OutputBuffer3, (2*N*N+N)*sizeof(float), Output_OpenCL3);
    OpenCL_ReadData(OutputBuffer4, (2*N*N+N)*sizeof(float), Output_OpenCL4);
}
//===============================================================================

int main()
{
    tic();
    Matrix<float, n, n> P;
    Matrix<float, n, n> Q;
    Matrix<float, m, m> R;
    Matrix<float, n, 1> x;
    Matrix<float, n, 1> s;
    Matrix<float, m, 1> y;
    Matrix<float, n, 1> x_mod;
    Matrix<float, 1, 2*n+1> Wm;
    Matrix<float, 1, 2*n+1> Wc;

    Matrix<float,n,2*n+1> X_trans;
    Matrix<float,m,2*n+1> Y_trans;
    Matrix<float,n,2*n+1> X_test;
    Matrix<float,n,2*n+1> Y_test;
//-------------------------------------
    Matrix<float,n,1> tmpx;
    Matrix<float,m,1> tmpy;
    Matrix<float,n,1> tmpx1;
    Matrix<float,m,1> tmpy1;
    Matrix<float,n,1> E_x;
    Matrix<float,m,1> E_y;
//--------------------------------------
    Matrix<float,n,n> tmpc;
    Matrix<float,n,n> P_xx;
    Matrix<float,m,m> P_yy;
    Matrix<float,n,m> P_xy;
    Matrix<float,n,m> K;
    Matrix<float,n,1> x_ukf;
    Matrix<float,n,n> P_ukf;

    Matrix<float, n, 1> x_bar;
    Matrix<float, m, 1> y_bar;
    Matrix<float, n, 2*n+1> X;
    Matrix<float, n, 2*n+1> Y;
    Matrix<float, n, 2*n+1> Z;
    Matrix<float,n,2*n+1> noiz;
    Matrix<float,20,1> spent;


//initialize variables
    P = Matrix<float,n,n>::Identity()*0.00001f;

    Q <<  2.4064e-5f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 2.4064e-5f, 0.0f, 0.0f, 0.0f,
    0.0f,       0.0f, 0.0f, 0.0f, 0.0f,
    0.0f,       0.0f, 0.0f, 0.0f, 0.0f,
    0.0f,       0.0f, 0.0f, 0.0f, 0.0f;

    R << 1.0f, 0.0f,
    0.0f,17.0f;

    x << 6500.4f,
    394.14f,
    -1.8093f,
    -6.7967f,
    0.0f;

    /**Q <<  2.4064e-5,       0.0, 0.0, 0.0, 0.0,
                0.0, 2.4064e-5, 0.0, 0.0, 0.0,
                0.0,       0.0, 0.0, 0.0, 0.0,
                0.0,       0.0, 0.0, 0.0, 0.0,
                0.0,       0.0, 0.0, 0.0, 0.0;

    R << 1.0, 0.0,
         0.0,17.0;

    x <<  6500.4,
          394.14,
         -1.8093,
         -6.7967,
             0.0;**/

//UKF parameters
//L=n
    r0 = 6374;
    gm0 = 3.9860e5;
    b0 = -0.59783;
    h0 = 13.406;
//--------------------------------------------------------------------------
    alpha = 1;
    ki = 1000;
    beta = 2.0;
    dt = 0.0051;
    lambda = pow(alpha,2)*(n+ki)-n;
    c = n+lambda;
    Wm = Matrix<float,1,2*n+1>::Constant(0.5/c);
    Wc = Wm;
    Wm(0,0) = lambda/c;
    Wc(0,0) = Wm(0,0)+(1-pow(alpha,2)+beta);
    x = x + Matrix <float, n, 1>::Random();
    x_mod = x;
    Matrix<float,n,2*n+1> W_temp;


    for(int j=0; j<n; j++)
    {
        for(int i =0; i<2*n+1; i++)
        {
            W_temp(j,i) = Wm(0,i);
        }
    }
    //OPENCL CODE
    if(
        !OpenCL_Init("NVIDIA"                ) && // nVidia
        !OpenCL_Init("Advanced Micro Devices") && // AMD
        !OpenCL_Init("Intel"                 ) && // Intel
        !OpenCL_Init(0                       )    // Default
    )
    {
        printf("Error: Cannot initialise OpenCL.\n");
        return 1;
    }

// Load a kernel
    if(!OpenCL_LoadKernel("OpenCL/Kernel.cl", "sgmp"))
        return 1;
//-------------------------------------------------------------------------
    N = 5;
    //dt = 0.1;
    size_t BufferSizeNx1 = N*sizeof(float);
//size_t BufferSizeMx1 = m*sizeof(float);
    size_t BufferSizeNxN = N*N*sizeof(float);
    size_t BufferSize2N1 = N*(2*N+1)*sizeof(float);
//size_t BufferSizeMxM = m*m*sizeof(float);
//size_t BufferSizeM2N1 = m*(2*N+1)*sizeof(float);
//size_t BufferSize2N = (2*N+1)*sizeof(float);

    A = (float*)malloc(BufferSizeNx1);
    B = (float*)malloc(BufferSizeNxN);
    L = (float*)malloc(BufferSizeNxN);
    T = (float*)malloc(BufferSize2N1);
    W = (float*)malloc(BufferSize2N1);

    Output_OpenCL  = (float*)malloc(BufferSize2N1);
    Output_OpenCL1 = (float*)malloc(BufferSize2N1);
    Output_OpenCL2 = (float*)malloc(BufferSize2N1);
    Output_OpenCL3 = (float*)malloc(BufferSize2N1);
    Output_OpenCL4 = (float*)malloc(BufferSize2N1);

    noiz = Matrix<float,n,2*n+1>::Random();
    T = noiz.data();
    W = W_temp.data();

// Allocate GPU RAM
//OpenCL_PrepareLocalSize(N, LocalSize);
    A_Buffer      = OpenCL_CreateBuffer(0, CL_MEM_READ_ONLY, BufferSizeNx1);
    B_Buffer      = OpenCL_CreateBuffer(1, CL_MEM_READ_ONLY, BufferSizeNxN);
    L_Buffer      = OpenCL_CreateBuffer(2, CL_MEM_READ_ONLY, BufferSizeNxN);
    T_Buffer      = OpenCL_CreateBuffer(3, CL_MEM_READ_ONLY, BufferSize2N1);
    W_Buffer      = OpenCL_CreateBuffer(4, CL_MEM_READ_ONLY, BufferSize2N1);
    OutputBuffer  = OpenCL_CreateBuffer(5, CL_MEM_READ_WRITE, BufferSize2N1);
    OutputBuffer1 = OpenCL_CreateBuffer(6, CL_MEM_READ_WRITE, BufferSize2N1);
    OutputBuffer2 = OpenCL_CreateBuffer(7, CL_MEM_READ_WRITE, BufferSize2N1);
    OutputBuffer3 = OpenCL_CreateBuffer(8, CL_MEM_READ_WRITE, BufferSize2N1);
    OutputBuffer4 = OpenCL_CreateBuffer(9, CL_MEM_READ_WRITE, BufferSize2N1);

    clear_text("x_estimate.csv");
    clear_text("x_model.csv");
    clear_text("y_model.csv");
    clear_text("y_estimate.csv");
//clear_text("times.csv");
//===============================================================================
// UKF STARTS
//===============================================================================

    for(int j=0; j<50; j++)
    {
        cout << "This is cycle #" << j+1 << endl << endl;
        y = measure(x_mod);
        x_mod = state(x_mod,dt);
//cholesky
        LLT<Matrix<float,n,n>> lltOfA(P*sqrt(c)); //the Cholesky decomposition of P
        tmpc = lltOfA.matrixL(); // retrieve factor L

        A = x.data();
        B = P.data();
        L = tmpc.data();


        Process_OpenCL();

//---------------------------------------------------------------------------------------------------------
//
        for(int j = 0; j < N; j++)
        {
            for(int i = 0; i < 2*N+1; i++)
            {
                X(j,i) = Output_OpenCL[N*i+j];
            }
        }

        for(int j = 0; j < N; j++)
        {
            for(int i = 0; i < 2*N+1; i++)
            {
                X_trans(j,i) = Output_OpenCL1[N*i+j];
            }
        }

        for(int j = 0; j < m; j++)
        {
            for(int i = 0; i < 2*N+1; i++)
            {
                Y_trans(j,i) = Output_OpenCL2[N*i+j];
            }
        }

//---------------------------------------------------------------------------------------------------------
        P_xx = Matrix<float,n,n>::Zero();
        P_xy = Matrix<float,n,m>::Zero();
        P_yy = Matrix<float,m,m>::Zero();
        E_x = Matrix<float,n,1>::Zero();
        E_y = Matrix<float,m,1>::Zero();

        for(int i=0; i<2*n+1; i++)
        {
            for(int j=0; j<n; j++)
            {
                tmpx(j,0) = Output_OpenCL3[N*i+j];
            }
            E_x += tmpx;
        }

        for(int i=0; i<2*n+1; i++)
        {
            for(int j=0; j<m; j++)
            {
                tmpy(j,0) = Output_OpenCL4[N*i+j];
            }
            E_y += tmpy;
        }

//-------covariances
        for(int i=0; i<2*n+1; i++)
        {
            for(int j=0; j<n; j++)
            {
                tmpx(j,0) = X_trans(j,i);
            }
            P_xx = P_xx+Wc(0,i)*(tmpx-E_x)*(tmpx-E_x).transpose();
        }
        P_xx = P_xx + Q;

        for(int i=0; i<2*n+1; i++)
        {
            for(int j=0; j<m; j++)
            {
                tmpy(j,0) = Y_trans(j,i);
            }
            P_yy = P_yy+Wc(0,i)*(tmpy-E_y)*(tmpy-E_y).transpose();
        }
        P_yy = P_yy+R;

        for(int i=0; i<2*n+1; i++)
        {
            for(int j=0; j<n; j++)
            {
                tmpx(j,0) = X_trans(j,i);
            }
            for(int k=0; k<m; k++)
            {
                tmpy(k,0) = Y_trans(k,i);
            }
            P_xy = P_xy+Wc(0,i)*(tmpx-E_x)*(tmpy-E_y).transpose();
        }
//------Kalman gain
        K = P_xy*P_yy.inverse();
//------Kalman Estimates
        x_ukf = E_x+K*(y-E_y);
        P_ukf = P_xx-K*P_yy*K.transpose();
        /**spent(j,0) = toc()/1e-3;
        printf("\n OpenCL serial segment : %lg ms\n\n", toc()/1e-3);**/
        //--------------------------------------------------
        P = P_ukf;
        x = x_ukf;

        dp = (float*)malloc(n*1);
        dp = x_ukf.data();
        csv_write("x_estimate.csv",dp,n,1);

        dp = (float*)malloc(n*1);
        dp = x_mod.data();
        csv_write("x_model.csv",dp,n,1);

        dp = (float*)malloc(m*1);
        dp = E_y.data();
        csv_write("y_estimate.csv",dp,m,1);

        dp = (float*)malloc(m*1);
        dp = y.data();
        csv_write("y_model.csv",dp,m,1);

        cout << "Sigma Points" << endl;
        cout << X << endl << endl;
        cout << "F(X) Transformed Sigma Points" << endl;
        cout << X_trans << endl << endl;
        cout << "H(F(X)) Transformed Sigma Points" << endl;
        cout << Y_trans << endl << endl;
        cout << "Weights Wm: " << endl;
        cout << Wm << endl << endl;
        cout << "Expected Value of x" << endl;
        cout << E_x << endl << endl;
        cout << "Expected Value of y" << endl;
        cout << E_y << endl << endl;

        cout << "Posterior state covariance" << endl;
        cout << P_xx << endl << endl;
        cout << "Posterior output covariance" << endl;
        cout << P_yy << endl << endl;
        cout << "Cross Covariance" << endl;
        cout << P_xy << endl << endl;
        cout << "Kalman Gain" << endl;
        cout << K << endl << endl;
        cout << "State Estimate" << endl;
        cout << x_ukf << endl << endl;
        cout << "Estimated state covariance" << endl;
        cout << P_ukf << endl << endl;
    }
    /**dp = (float*)malloc(20*1);
    dp = spent.data();
    csv_write("times.csv",dp,20,1);**/

    free(Output_OpenCL);
    free(Output_OpenCL1);
    free(Output_OpenCL2);
    free(Output_OpenCL3);
    free(Output_OpenCL4);

    OpenCL_FreeBuffer(A_Buffer    );
    OpenCL_FreeBuffer(B_Buffer    );
    OpenCL_FreeBuffer(L_Buffer    );
    OpenCL_FreeBuffer(OutputBuffer);
    OpenCL_FreeBuffer(S_Buffer    );
    OpenCL_FreeBuffer(T_Buffer    );
    OpenCL_FreeBuffer(W_Buffer    );
    OpenCL_FreeBuffer(OutputBuffer1);
    OpenCL_FreeBuffer(OutputBuffer2);
    OpenCL_FreeBuffer(OutputBuffer3);
    OpenCL_FreeBuffer(OutputBuffer4);
    OpenCL_Destroy();
    cout << "OpenCL UKF Duration: " << toc()/1e-3 << "ms" << "\n\n";
    return 0;
}
//------------------------------------------------------------------------------
