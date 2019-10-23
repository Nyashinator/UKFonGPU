__kernel
void sgmp(
    __global float* A,
    __global float* B,
    __global float* L,
    __global float* T,
    __global float* W,
    __global float* Output,
    __global float* Output1,
    __global float* Output2,
    __global float* Output3,
    __global float* Output4,
    const unsigned int N,
    const float       dt,
    const float       r0,
    const float      gm0,
    const float       b0,
    const float       h0
)
{
    const int i = get_global_id(1); // Row
    const int j = get_global_id(0); // Column
//-----------------------------------------------------------------------------------------------------
// CALCULATE SIGMA POINTS FROM X
//-----------------------------------------------------------------------------------------------------
    /**if(j==0){
     Output[i] = A[i];
     Output[N+i] = A[i]+L[N*(j)+i];        //e0 = A[i]+L[N*(j-1)+i];
     Output[N*N+N+i] = A[i]-L[N*(j)+i];  //e1 = A[i]-L[N*(j-1)+i];
    }else{

     Output[N*(j+1)+i] = A[i]+L[N*(j)+i];        //e0 = A[i]+L[N*(j-1)+i];
     Output[N*N+N*(j+1)+i] = A[i]-L[N*(j)+i];  //e1 = A[i]-L[N*(j-1)+i];

     }
     if(j==4){
      Output[N*(j+1)+i] = A[i]+L[N*(j)+i];
      Output[N*N+N*(j+1)+i] = A[i]-L[N*(j)+i];
     }**/

    if(i==0)
    {
        Output[i] = A[i];
        Output[N*(j+1)+i] = A[i]+L[N*(j)+i];        //e0 = A[i]+L[N*(j-1)+i];
        Output[N*N+N*(j+1)+i] = A[i]-L[N*(j)+i];  //e1 = A[i]-L[N*(j-1)+i];
    }
    if(i==1)
    {
        Output[i] = A[i];
        Output[N*(j+1)+i] = A[i]+L[N*(j)+i];        //e0 = A[i]+L[N*(j-1)+i];
        Output[N*N+N*(j+1)+i] = A[i]-L[N*(j)+i];
    }
    if(i==2)
    {
        Output[i] = A[i];
        Output[N*(j+1)+i] = A[i]+L[N*(j)+i];        //e0 = A[i]+L[N*(j-1)+i];
        Output[N*N+N*(j+1)+i] = A[i]-L[N*(j)+i];
    }
    if(i==3)
    {
        Output[i] = A[i];
        Output[N*(j+1)+i] = A[i]+L[N*(j)+i];        //e0 = A[i]+L[N*(j-1)+i];
        Output[N*N+N*(j+1)+i] = A[i]-L[N*(j)+i];
    }
    if(i==4)
    {
        Output[i] = A[i];
        Output[N*(j+1)+i] = A[i]+L[N*(j)+i];        //e0 = A[i]+L[N*(j-1)+i];
        Output[N*N+N*(j+1)+i] = A[i]-L[N*(j)+i];
    }

//-----------------------------------------------------------------------------------------------------
// PROPAGATE SIGMA POINTS THROUGH F(X)
//-----------------------------------------------------------------------------------------------------
    float b,b1,r,r1,V,V1,G,G1,D,D1;
    b = -b0*exp(Output[N*j+4]);
    b1 = -b0*exp(Output[N*N+N*j+4]);
    r = sqrt(pow(Output[N*j],2)+pow(Output[N*j+1],2));
    r1 = sqrt(pow(Output[N*N+N*j],2)+pow(Output[N*N+N*j+1],2));
    V = sqrt(pow(Output[N*j+2],2)+pow(Output[N*j+3],2));
    V1 = sqrt(pow(Output[N*N+N*j+2],2)+pow(Output[N*N+N*j+3],2));
    G = -(gm0)/(pow(r,3));
    G1 = -(gm0)/(pow(r1,3));
    D = -b*exp((r0-r)/h0)*V;
    D1 = -b1*exp((r0-r1)/h0)*V1;

    if(i==0)
    {
        Output1[N*j+i]     = Output[N*j+i]+Output[N*j+2]*dt;
        Output1[N*N+N*j+i] = Output[N*N+N*j+i]+Output[N*N+N*j+2]*dt;
    }
    if(i==1)
    {
        Output1[N*j+i]     = Output[N*j+i]+Output[N*j+3]*dt;
        Output1[N*N+N*j+i] = Output[N*N+N*j+i]+Output[N*N+N*j+3]*dt;
    }
    if(i==2)
    {
        Output1[N*j+i]     = Output[N*j+i]+(D*Output[N*j+2]+G*Output[N*j]+T[N*j])*dt;
        Output1[N*N+N*j+i] = Output[N*N+N*j+i]+(D1*Output[N*N+N*j+2]+G1*Output[N*N+N*j]+T[N*N+N*j])*dt;
    }
    if(i==3)
    {
        Output1[N*j+i]     = Output[N*j+i]+(D*Output[N*j+3]+G*Output[N*j+1]+T[N*j+1])*dt;
        Output1[N*N+N*j+i] = Output[N*N+N*j+i]+(D1*Output[N*N+N*j+3]+G1*Output[N*N+N*j+1]+T[N*N+N*j+1])*dt;
    }
    if(i==4)
    {
        Output1[N*j+i]     = T[N*j+2];
        Output1[N*N+N*j+i] = T[N*N+N*j+2];
    }
    if(j==4)
    {
        Output1[N*N+N*(j+1)+0] = Output[N*N+N*(j+1)+0]+Output[N*N+N*(j+1)+2]*dt;
        Output1[N*N+N*(j+1)+1] = Output[N*N+N*(j+1)+1]+Output[N*N+N*(j+1)+3]*dt;
        Output1[N*N+N*(j+1)+2] = Output[N*N+N*(j+1)+2]+(D1*Output[N*N+N*(j+1)+2]+G1*Output[N*N+N*(j+1)]+T[N*N+N*(j+1)])*dt;
        Output1[N*N+N*(j+1)+3] = Output[N*N+N*(j+1)+3]+(D1*Output[N*N+N*(j+1)+3]+G1*Output[N*N+N*(j+1)+1]+T[N*N+N*(j+1)+1])*dt;
        Output1[N*N+N*(j+1)+4] = T[N*N+N*(j+1)+2];

        Output1[N*(j+1)+0] = Output[N*(j+1)+0]+Output[N*(j+1)+2]*dt;
        Output1[N*(j+1)+1] = Output[N*(j+1)+1]+Output[N*(j+1)+3]*dt;
        Output1[N*(j+1)+2] = Output[N*(j+1)+2]+(D1*Output[N*(j+1)+2]+G1*Output[N*(j+1)]+T[N*(j+1)])*dt;
        Output1[N*(j+1)+3] = Output[N*(j+1)+3]+(D1*Output[N*(j+1)+3]+G1*Output[N*(j+1)+1]+T[N*(j+1)+1])*dt;
        Output1[N*(j+1)+4] = T[N*(j+1)+2];
    }
//-----------------------------------------------------------------------------------------------------
// PROPAGATE F(X) OUTPUT FROM SIGMA POINTS THROUGH H(X)
//-----------------------------------------------------------------------------------------------------

    if(i==0)
    {
        Output2[N*j]     = sqrt(pow((Output1[N*j]-r0),2)+pow(Output1[N*j+1],2));
        Output2[N*N+N*j] = sqrt(pow((Output1[N*N+N*j]-r0),2)+pow(Output1[N*N+N*j+1],2));
    }
    if(i==1)
    {
        Output2[N*j+i]     = atan(Output1[N*j+i]/(Output1[N*j]-r0));
        Output2[N*N+N*j+i] = atan(Output1[N*N+N*j+i]/(Output1[N*N+N*j]-r0));
    }
    if(j==4)
    {
        Output2[N*N+N*(j+1)+0] = sqrt(pow((Output1[N*N+N*(j+1)]-r0),2)+pow(Output1[N*N+N*(j+1)+1],2));
        Output2[N*N+N*(j+1)+1] = atan(Output1[N*N+N*(j+1)+1]/(Output1[N*N+N*(j+1)]-r0));

        Output2[N*(j+1)+0] = sqrt(pow((Output1[N*(j+1)]-r0),2)+pow(Output1[N*(j+1)+1],2));
        Output2[N*(j+1)+1] = atan(Output1[N*(j+1)+1]/(Output1[N*(j+1)]-r0));
    }
//-----------------------------------------------------------------------------------------------------
// FIND THE EXPECTED VALUES OF X AND Y
//-----------------------------------------------------------------------------------------------------
    if(i==0)
    {
        Output3[N*j] = W[N*j]*Output1[N*j];
        Output3[N*N+N*j]  = W[N*N+N*j]*Output1[N*N+N*j];
        Output4[N*j] = W[N*j]*Output2[N*j];
        Output4[N*N+N*j] = W[N*N+N*j]*Output2[N*N+N*j];
    }
    if(i==1)
    {
        Output3[N*j+i] = W[N*j+i]*Output1[N*j+i];
        Output3[N*N+N*j+i]  = W[N*N+N*j+i]*Output1[N*N+N*j+i];
        Output4[N*j+i] = W[N*j+i]*Output2[N*j+i];
        Output4[N*N+N*j+i] = W[N*N+N*j+i]*Output2[N*N+N*j+i];
    }
    if(i==2)
    {
        Output3[N*j+i] = W[N*j+i]*Output1[N*j+i];
        Output3[N*N+N*j+i]  = W[N*N+N*j+i]*Output1[N*N+N*j+i];
    }
    if(i==3)
    {
        Output3[N*j+i] = W[N*j+i]*Output1[N*j+i];
        Output3[N*N+N*j+i]  = W[N*N+N*j+i]*Output1[N*N+N*j+i];
    }
    if(i==4)
    {
        Output3[N*j+i] = W[N*j+i]*Output1[N*j+i];
        Output3[N*N+N*j+i]  = W[N*N+N*j+i]*Output1[N*N+N*j+i];
    }
    if(j==4)
    {
        Output3[N*N+N*(j+1)+0]  = W[N*N+N*(j+1)+0]*Output1[N*N+N*(j+1)+0];
        Output3[N*N+N*(j+1)+1]  = W[N*N+N*(j+1)+1]*Output1[N*N+N*(j+1)+1];
        Output3[N*N+N*(j+1)+2]  = W[N*N+N*(j+1)+2]*Output1[N*N+N*(j+1)+2];
        Output3[N*N+N*(j+1)+3]  = W[N*N+N*(j+1)+3]*Output1[N*N+N*(j+1)+3];
        Output3[N*N+N*(j+1)+4]  = W[N*N+N*(j+1)+4]*Output1[N*N+N*(j+1)+4];

        Output4[N*N+N*(j+1)+0] = W[N*N+N*(j+1)+0]*Output2[N*N+N*(j+1)+0];
        Output4[N*N+N*(j+1)+1] = W[N*N+N*(j+1)+1]*Output2[N*N+N*(j+1)+1];

        Output3[N*(j+1)+0]  = W[N*(j+1)+0]*Output1[N*(j+1)+0];
        Output3[N*(j+1)+1]  = W[N*(j+1)+1]*Output1[N*(j+1)+1];
        Output3[N*(j+1)+2]  = W[N*(j+1)+2]*Output1[N*(j+1)+2];
        Output3[N*(j+1)+3]  = W[N*(j+1)+3]*Output1[N*(j+1)+3];
        Output3[N*(j+1)+4]  = W[N*(j+1)+4]*Output1[N*(j+1)+4];

        Output4[N*(j+1)+0] = W[N*(j+1)+0]*Output2[N*(j+1)+0];
        Output4[N*(j+1)+1] = W[N*(j+1)+1]*Output2[N*(j+1)+1];
    }
}






