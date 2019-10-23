#include "Eigen/Dense"
#include "Eigen/Core"
#include "Eigen/SVD"
#include <stdio.h>
#include <cmath>
#include <math.h>
#include <fstream>
#include <iostream>
#include "main.h"

using namespace std ;
using namespace Eigen;

//------------------------------------------------------------------------------
//GLOBAL MATRICES
//------------------------------------------------------------------------------
 //Matrix <double, n, 1> x;
 //Matrix <double, n, 1> s;
//------------------------------------------------------------------------------
// FUNCTIONS
//------------------------------------------------------------------------------
void csv_write(const char* filename, double* mat, int r, int c){
  fstream fs;
  fs.open(filename, fstream::in | fstream::out | fstream::app);

  //fs << "Step(k),x\n";
  for(int j=0;j<c;j++){
    for(int i=0;i<r;i++){
        fs << mat[i*c+j]<< ",";
    }
    fs << endl;
  }

  fs.close();
}

void clear_text(const char* filename){
  fstream fs;
  fs.open(filename, fstream::in | fstream::out | fstream::trunc);
  fs.close();
}

Matrix <double,n,1> state(Matrix<double,n,1> x, double t){
 Matrix <double, n, 1> s;
 Matrix <double, n, 1> vn;
 vn = Matrix <double, n, 1>::Random();

 //state vector derivative
 s(0,0) = x(0,0)+x(2,0)*t;
 s(1,0) = x(1,0)+x(3,0)*t;
 s(2,0) = x(2,0)+(x(2,0)+x(0,0)+vn(0,0))*t;
 s(3,0) = x(3,0)+(x(3,0)+x(1,0)+vn(1,0))*t;
 s(4,0) = vn(2,0);
 return s;
}
//-----------------------------------------------------------------------------
Matrix <double,m,1> measure(Matrix<double,n,1> x){
 Matrix <double, m, 1> wn;
 Matrix <double, m, 1> y;
 wn = Matrix <double, m, 1>::Random();

 y(0,0) = sqrt(pow((x(0,0)-r0),2)+pow(x(1,0),2))+wn(0,0);
 y(1,0) = atan(x(1,0)/(x(0,0)-r0))+wn(1,0);

 return y;
}
//-----------------------------------------------------------------------------
Matrix<double,n,2*n+1> sgmp(Matrix<double,n,1> x, Matrix<double,n,n> B, double g){
 Matrix<double,n,n> E;
 Matrix<double,n,2*n+1> X;
 Matrix<double,n,n> Y;
 //cholesky
 LLT<Matrix<double,n,n>> lltOfA(B*sqrt(g)); //the Cholesky decomposition of P
 E = lltOfA.matrixL(); // retrieve factor E
 cout << E << "\n\n";
 //make sigma matrix
 for(int i=0; i<n; i++){
  for(int j=0; j<n; j++){
   X(j,i+1) = x(j,0)+E(j,i);
   X(j,i+n+1) = x(j,0)-E(j,i);
  }
  X(i,0) = x(i,0);
 }
 return X;
}
//-----------------------------------------------------------------------------
Matrix<double,n,2*n+1> utransformX(Matrix<double,n,2*n+1> B){
 Matrix<double,n,2*n+1> X_t;
 Matrix<double,n,1> tmp;

 for(int i=0;i<2*n+1;i++){
  for(int j=0;j<n;j++){
   tmp(j,0) = B(j,i);
  }
  tmp = state(tmp,dt);
  for(int l=0;l<n;l++){
   X_t(l,i) = tmp(l,0);
  }
 }
 return X_t;
}
//------------------------------------------------------------------------------
Matrix<double,m,2*n+1> utransformY(Matrix<double,n,2*n+1> B){
 Matrix<double,m,2*n+1> Y_t;
 Matrix<double,n,1> tmp;
 Matrix<double,m,1> tmp2;

 for(int i=0;i<2*n+1;i++){
  for(int j=0;j<n;j++){
   tmp(j,0) = B(j,i);
  }
  tmp2 = measure(tmp);
  for(int l=0;l<m;l++){
   Y_t(l,i) = tmp2(l,0);
  }
 }
 return Y_t;
}

//---------------------------------------------------------------------------
int main(){
 tic();
 N = 5;

 Matrix<double, n, n> P;
 Matrix<double, n, n> Q;
 Matrix<double, m, m> R;
 Matrix<double, n, 1> x;
 Matrix<double, m, 1> y;
 Matrix<double, n, 1> x_mod;
 Matrix<double, 1, 2*n+1> Wm;
 Matrix<double, 1, 2*n+1> Wc;
 Matrix<double, n, 2*n+1> X;
 Matrix<double,n,2*n+1> X_trans;
 Matrix<double,m,2*n+1> Y_trans;
 //-------------------------------------
 Matrix<double,n,1> tmpx;
 Matrix<double,m,1> tmpy;
 Matrix<double,n,1> E_x;
 Matrix<double,m,1> E_y;
 //--------------------------------------
 Matrix<double,n,n> P_xx;
 Matrix<double,m,m> P_yy;
 Matrix<double,n,m> P_xy;
 Matrix<double,n,m> K;
 Matrix<double,n,1> x_ukf;
 Matrix<double,n,n> P_ukf;
 Matrix<double,20,1> spent;

 //initialize variables
 P << 1e-5, 0, 0, 0, 0,
      0, 1e-5, 0, 0, 0,
      0, 0, 1e-5, 0, 0,
      0, 0, 0, 1e-5, 0,
      0, 0, 0, 0, 1e-5;

 Q <<  2.4064e-5, 0, 0, 0, 0,
       0, 2.4064e-5, 0, 0, 0,
       0,         0, 0, 0, 0,
       0,         0, 0, 0, 0,
       0,         0, 0, 0, 0;

 R << 1, 0,
      0,17;

 x << 6500.4,
      394.14,
      -1.8093,
      -6.7967,
      0;

 //UKF parameters
 //L=n
 r0 = 6374;
 alpha = 0.1;
 ki = 1000;
 beta = 2;
 dt = 0.0051;
 lambda = pow(alpha,2)*(L+ki)-L;
 c = L+lambda;
 Wm = Matrix<double,1,2*n+1>::Constant(0.5/c);
 Wc = Wm;
 Wm(0,0) = lambda/c;
 Wc(0,0) = Wm(0,0)+(1-pow(alpha,2)+beta);
 x = x + Matrix <double, n, 1>::Random();
 x_mod = x;

 clear_text("x_estimate.csv");
 clear_text("x_model.csv");
 clear_text("y_model.csv");
 clear_text("y_estimate.csv");
 clear_text("times.csv");
//===============================================================================
// UKF STARTS
//===============================================================================
 for(int j=0;j<50;j++){
  cout << "This is cycle #" << j+1 << endl << endl;
  y = measure(x_mod);
  x_mod = state(x_mod,dt);
//-------------------------------------------------------
 Matrix<double,n,n> E;
 Matrix<double,n,2*n+1> Z;
//------------------------------------------------------

  X = sgmp(x,P,c);
  X_trans = utransformX(X);
  Y_trans = utransformY(X_trans);

 //-------------means
 E_x = Matrix<double,n,1>::Zero();
 E_y = Matrix<double,m,1>::Zero();
 P_xy = Matrix<double,n,m>::Zero();
 P_xx = Matrix<double,n,n>::Zero();
 P_yy = Matrix<double,m,m>::Zero();

  for(int i=0;i<2*n+1;i++){
   for(int j=0;j<n;j++){
    tmpx(j,0) = X_trans(j,i);
   }
   E_x = E_x+Wm(0,i)*tmpx;
  }

  for(int i=0;i<2*n+1;i++){
   for(int j=0;j<m;j++){
    tmpy(j,0) = Y_trans(j,i);
   }
   E_y = E_y+Wm(0,i)*tmpy;
  }

 //-------covariances
  for(int i=0;i<2*n+1;i++){
   for(int j=0;j<n;j++){
    tmpx(j,0) = X_trans(j,i);
   }
   P_xx = P_xx+Wc(0,i)*(tmpx-E_x)*(tmpx-E_x).transpose();
  }
  P_xx = P_xx + Q;

  for(int i=0;i<2*n+1;i++){
   for(int j=0;j<m;j++){
    tmpy(j,0) = Y_trans(j,i);
   }
   P_yy = P_yy+Wc(0,i)*(tmpy-E_y)*(tmpy-E_y).transpose();
  }
  P_yy = P_yy+R;

  for(int i=0;i<2*n+1;i++){
   for(int j=0;j<n;j++){
    tmpx(j,0) = X_trans(j,i);
   }
   for(int k=0;k<m;k++){
    tmpy(k,0) = Y_trans(k,i);
   }
   P_xy = P_xy+Wc(0,i)*(tmpx-E_x)*(tmpy-E_y).transpose();
  }
 //------Kalman gain
  K = P_xy*P_yy.inverse();
 //------Kalman Estimates
 x_ukf = E_x+K*(y-E_y);
 P_ukf = P_xx-K*P_yy*K.transpose();
//cout << "Segment Timer : " << toc()/1e-3 << endl;
//spent(j,0) = toc()/1e-3;
 //--------------------------------------------------
 P = P_ukf;
 x = x_ukf;
//-----write results to text file
   dp = (double*)malloc(n*1);
  dp = x_ukf.data();
  csv_write("x_estimate.csv",dp,n,1);

  dp = (double*)malloc(n*1);
  dp = x_mod.data();
  csv_write("x_model.csv",dp,n,1);

  dp = (double*)malloc(m*1);
  dp = E_y.data();
  csv_write("y_estimate.csv",dp,m,1);

  dp = (double*)malloc(m*1);
  dp = y.data();
  csv_write("y_model.csv",dp,m,1);

  //----print results in terminal
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
   /**dp = (double*)malloc(20*1);
  dp = spent.data();
  csv_write("times.csv",dp,20,1);**/
cout << "Serial UKF Duration: " << toc()/1e-3 << "ms" << "\n\n";
 return 0;
}
//------------------------------------------------------------------------------
