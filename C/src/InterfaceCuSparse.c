#ifdef _CUDABLAS_
#include<InterfaceCuSparse.h>
#include<cusparse.h>
#include<cuda_runtime.h>

cusparseHandle_t cusparseHandle;
cusparseStatus_t cusparseStatus;
cusparseMatDescr_t descr;
cudaEvent_t start, stop;
double *devA,*devX,*devY;
int    *devIaCsr,*devJaCsr;

/********************************************************************* 
 * INITCUSPARSE : innicializacao dos parametros de biblioteca        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------*  
 * neq          -> numero de equacoes                                * 
 * ia           -> vetor CSR                                         * 
 * ja           -> vetor CSR                                         * 
 * a            -> vetor com os valores da matriz                    * 
 * timeOverHead -> indefinido                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * timeOverHead -> tempo de overhead de transferencia de dados       * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void initCuSparse(INT neq  ,INT *ia            ,INT*ja
                 ,double *a,double *timeOverHead){       
  INT nad = ia[neq];
/*... incialicando sparse*/
  cusparseStatus = cusparseCreate(&cusparseHandle);
  cusparseStatus = cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
/*  cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_SYMMETRIC);*/
  cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
/*...................................................................*/

/*... Time Event*/
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
/*...................................................................*/

/*... copiando para a GPU*/
  *timeOverHead = getTimeC() - *timeOverHead;
  cudaMalloc((void **)&devIaCsr,(neq+1)*sizeof(int));
  cudaMalloc((void **)&devJaCsr,    nad*sizeof(int));
  cudaMalloc((void **)&devX    ,    neq*sizeof(double));
  cudaMalloc((void **)&devY    ,    neq*sizeof(double));
  cudaMalloc((void **)&devA    ,    nad*sizeof(double));
/*...................................................................*/

/*... copiando para a GPU*/
  cudaMemcpy(devIaCsr,ia,    (neq+1)*sizeof(int)
            , cudaMemcpyHostToDevice);
  cudaMemcpy(devJaCsr,ja,    nad*sizeof(int)
            , cudaMemcpyHostToDevice);
  cudaMemcpy(devA    , a, nad*sizeof(double)
            , cudaMemcpyHostToDevice);
/*...................................................................*/

/*... copiando para a GPU*/
  cudaMemcpy(devIaCsr,ia,(neq+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(devJaCsr,ja,    nad*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(devA    , a, nad*sizeof(double),cudaMemcpyHostToDevice);
  *timeOverHead = getTimeC() - *timeOverHead;
/*...................................................................*/
}
/*********************************************************************/ 

/********************************************************************* 
 * CSRCUSPARSE : interface para csr da biblioteca cuSparse           * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------*  
 * neq          -> numero de equacoes                                * 
 * ia           -> vetor CSR                                         * 
 * ja           -> vetor CSR                                         * 
 * a            -> vetor com os valores da matriz                    * 
 * x            -> vetor a ser multiplicado                          * 
 * y            -> indefinido                                        * 
 * timeOverHead -> indefinido                                        * 
 * timeKernell  -> indefinido                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y            -> valor do produto matriz vetor                     * 
 * timeOverHead -> tempo de overhead de transferencia de dados       * 
 * timeKernell  -> tempo do kernell da CPU                           * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void csrCuSparse(INT neq             ,INT nad  
                ,double *x           ,double *y
                ,double *timeOverHead,double *timeKernell){       
  
  double alpha,beta;
  float  taux;

  alpha = 1.0e0;
  beta  = 0.0e0;

/*... copiando x e y para o device*/
// *timeOverHead = getTimeC() - *timeOverHead;
  cudaMemcpy(devX,x,neq*sizeof(double),cudaMemcpyHostToDevice);
//  *timeOverHead = getTimeC() - *timeOverHead;
/*...................................................................*/


/*...*/
//   cudaEventRecord(start, 0);
   cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE
                 ,neq           ,neq  
                 ,nad           ,&alpha
                 ,descr         ,devA  
                 ,devIaCsr      ,devJaCsr
                 ,devX          ,&beta
                 ,devY);
//    cudaEventRecord(stop, 0);
//    cudaEventSynchronize(stop);
//    cudaEventElapsedTime(&taux, start, stop);
//    *timeKernell += taux;
/*...................................................................*/

/*... copiando DEVICE -> HOST*/ 
//    *timeOverHead = getTimeC() - *timeOverHead;
    cudaMemcpy(y,devY,neq*sizeof(double),cudaMemcpyDeviceToHost);
//    *timeOverHead = getTimeC() - *timeOverHead;
/*...................................................................*/
}
#endif
