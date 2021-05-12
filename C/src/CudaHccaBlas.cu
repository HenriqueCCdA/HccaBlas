#include<CudaHccaBlas.h>
#define HANDLE_ERROR_CUDA(err) (HandleError( err, __FILE__, __LINE__ ))

static void HandleError(cudaError_t err,const char *file,int line){
  if (err != cudaSuccess) {
    printf( "CUDA: %s in %s at line %d\n", cudaGetErrorString( err ),
          file, line );
    exit( EXIT_FAILURE );
  }
}
/********************************************************************* 
 * addVectorDevice: kernel da soma de vetores escrito em cuda        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a -> vetor a                                                      * 
 * b -> vetor b                                                      * 
 * c -> nao definido                                                 * 
 * n -> dimensao do vetores                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * c -> resultado da soma                                            * 
 *-------------------------------------------------------------------* 
 * OBS: kernel que ira para GPU                                      * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
__global__ void addVectorDevice(DOUBLE const *a,DOUBLE const *b
                                ,DOUBLE *c     ,INT const n){
 

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  while(tid < n){
    c[tid] = a[tid] + b[tid];
    tid   += blockDim.x * gridDim.x;
  }

  
}
/*********************************************************************/ 

/********************************************************************* 
 * ZEROVECDEVICE: zerando a valores de um vetor no device            * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a -> vetor a                                                      * 
 * n -> dimensao do vetores                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * a -> vetor a zerado                                               * 
 *-------------------------------------------------------------------* 
 * OBS: kernel que ira para GPU                                      * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
__global__ void zeroVecDevice(DOUBLE *a,INT const n){
 

  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  while(tid < n){
    a[tid] = 0.0e0;
    tid   += blockDim.x * gridDim.x;
  }

  
}
/*********************************************************************/ 


/********************************************************************* 
 * matVecCsrScalarDevice: kernel do produto matriz vetor CSR (SCALAR)* 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq     -> numero de equacoes                                     * 
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> vetor x na GPU                                         * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devY -> produto da operacao de matriz vetor (GPU)                 * 
 *-------------------------------------------------------------------* 
 * OBS: kernel que ira para GPU                                      * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
__global__ void  matVecCsrScalarDevice(INT const neq     
                                ,INT const *ia  ,INT const *ja
                                ,DOUBLE const *a,DOUBLE const *x
                                ,DOUBLE *y){
  int threadId = threadIdx.x + blockIdx.x*blockDim.x;
  int gridSize = blockDim.x*gridDim.x;
  int i;
  double tmp;
  int ia1,ia2,ja1;
//  printf("thread %d block %d blockDim %d gridDim %d\n"
//        ,threadIdx.x,blockIdx.x,blockDim.x,gridDim.x);
  for( i= threadId; i < neq; i+=gridSize)  {
    tmp = 0.0e0;
    ia1 = ia[  i];
    ia2 = ia[i+1];
    for(ja1=ia1;ja1<ia2;ja1++)
      tmp += a[ja1] * x[ja[ja1]];
    y[i] = tmp;
  }    

}
/*********************************************************************/ 

/********************************************************************* 
 * matVecCsrVectorDeviceL32: kernel do produto matriz vetor CSR      *
 * (VECTOR)                                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq     -> numero de equacoes                                     * 
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> vetor x na GPU                                         * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devY -> produto da operacao de matriz vetor (GPU)                 * 
 *-------------------------------------------------------------------* 
 * OBS: kernel que ira para GPU                                      * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
__global__ void  matVecCsrVectorDeviceL32(INT const neq     
                                         ,INT const *ia  ,INT const *ja
                                         ,DOUBLE const *a,DOUBLE const *x
                                         ,DOUBLE *y){
//  __shared__ double volatile vals[128+16];
  extern __shared__ volatile double vals[];
  int threadsPerVector = 8;
  int threadsPerBlock  = blockDim.x;
  int vectorPerBlock   = threadsPerBlock/threadsPerVector;
  int threadId = threadIdx.x + blockIdx.x*threadsPerBlock; 
  int numVectors = vectorPerBlock*gridDim.x;
  int cacheIndex = threadIdx.x;
/*global warp index*/
  int lineVector =  threadId   / threadsPerVector;
/*thread index within the warp*/
  int threadLane     = threadId & (threadsPerVector-1);
  int j,ia1,ia2,ja1;
  double tmp;
  

//  for( i= warpId; i < neq; i+=numVectors)  {
  while(lineVector < neq){
    ia1 = ia[  lineVector];
    ia2 = ia[lineVector+1];
/*... compute runnig sum per thread*/
    tmp = 0.0e0;
    for(ja1=ia1+threadLane;ja1<ia2;ja1+=threadsPerVector){
      tmp += a[ja1] * x[ja[ja1]];
    }
    vals[cacheIndex] = tmp;

/*... parallel reduction in shared memory*/
    j = threadsPerVector/2; 
    while( j!= 0) {
      if(threadLane < j)
        vals[cacheIndex] += vals[cacheIndex + j];
      j/= 2;
    }

/*... firts thread writs the result*/
    if(threadLane == 0) y[lineVector] = vals[cacheIndex];
    
    lineVector+=numVectors;
    
  }    

}
/*********************************************************************/ 

/********************************************************************* 
 * matVecCsrVectorDeviceLCuspRt2008: kernel do produto matriz vetor  *
 * CSR (VECTOR)                                                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq     -> numero de equacoes                                     * 
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> vetor x na GPU                                         * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devY -> produto da operacao de matriz vetor (GPU)                 * 
 *-------------------------------------------------------------------* 
 * OBS: Retirado da biblioteca Cusp 2008                             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
template <typename indexType
         ,typename valueType
         ,unsigned int BLOCK_SIZE
         ,unsigned int WARP_SIZE> 
__global__ void  matVecCsrVectorDeviceCuspRt2008(indexType const neq     
                          ,indexType const *ia  ,indexType const *ja
                          ,valueType const *a   ,valueType const *x
                          ,valueType *y){
  __shared__ valueType vals[BLOCK_SIZE];
  __shared__ indexType ptrs[BLOCK_SIZE/WARP_SIZE][2];
  const indexType threadId    = threadIdx.x + blockIdx.x*BLOCK_SIZE; 
  const indexType threadLane  = threadId & (WARP_SIZE-1);
/*global warp index*/
  const indexType warpId   = threadId   / WARP_SIZE;
/*warp index within the CTA*/
  const indexType warpLane = threadIdx.x / WARP_SIZE;
/* total number of active warps*/
  const indexType numWarps = (BLOCK_SIZE/WARP_SIZE)*gridDim.x;
/*thread index within the warp*/
  indexType line,ia1,ia2,ja1;
  valueType tmp;


  for( line= warpId; line < neq; line+=numWarps)  {
/* use two threads to fetch ia[line] and ia[line+1]
   this is considerably faster than the more straightforward option*/
    if( threadLane < 2 )
      ptrs[warpLane][threadLane] = ia[line + threadLane];
/*same as: ia1 = Ap[row];*/
    ia1 = ptrs[warpLane][0];
/*same as: ia2 = Ap[row];*/
    ia2 = ptrs[warpLane][1];
/*... compute runnig sum per thread*/
    tmp = 0.0e0;
    for(ja1=ia1+threadLane;ja1<ia2;ja1+=WARP_SIZE){
      tmp += a[ja1] * x[ja[ja1]];
    }
    vals[threadIdx.x] = tmp;
    __syncthreads();

/*... parallel reduction in shared memory*/
    if (threadLane < 16){
      vals[threadIdx.x] += vals[threadIdx.x + 16];
      __syncthreads();
    }
    if (threadLane <  8){
      vals[threadIdx.x] += vals[threadIdx.x +  8];
      __syncthreads();
    }
    if (threadLane <  4){
      vals[threadIdx.x] += vals[threadIdx.x +  4];
      __syncthreads();
    }
    if (threadLane <  2){
      vals[threadIdx.x] += vals[threadIdx.x +  2];
      __syncthreads();
    }
    if (threadLane <  1){
      vals[threadIdx.x] += vals[threadIdx.x +  1];
      __syncthreads();
    }


/*... firts thread writs the result*/
    if(threadLane == 0) y[line] = vals[threadIdx.x];

  }    

}
/*********************************************************************/ 

/********************************************************************* 
 * matVecCsrVectorDeviceLCuspSc2009: kernel do produto matriz vetor  *
 * CSR (VECTOR)                                                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq     -> numero de equacoes                                     * 
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> vetor x na GPU                                         * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devY -> produto da operacao de matriz vetor (GPU)                 * 
 *-------------------------------------------------------------------* 
 * OBS: Retirado da biblioteca Cusp 2009                             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
template <typename indexType
         ,typename valueType
         ,unsigned int BLOCK_SIZE
         ,unsigned int WARP_SIZE> 
__global__ void  matVecCsrVectorDeviceCuspSc2009(indexType const neq     
                          ,indexType const *ia  ,indexType const *ja
                          ,valueType const *a   ,valueType const *x
                          ,valueType *y){
  __shared__ valueType vals[BLOCK_SIZE+16];
  __shared__ indexType ptrs[BLOCK_SIZE/WARP_SIZE][2];
  const indexType threadId    = threadIdx.x + blockIdx.x*BLOCK_SIZE; 
  const indexType threadLane  = threadId & (WARP_SIZE-1);
/*global warp index*/
  const indexType warpId   = threadId   / WARP_SIZE;
/*warp index within the CTA*/
  const indexType warpLane = threadIdx.x / WARP_SIZE;
/* total number of active warps*/
  const indexType numWarps = (BLOCK_SIZE/WARP_SIZE)*gridDim.x;
/*thread index within the warp*/
  indexType line,ia1,ia2,ja1;
  valueType tmp;


  for( line= warpId; line < neq; line+=numWarps)  {
/* use two threads to fetch ia[line] and ia[line+1]
   this is considerably faster than the more straightforward option*/
    if( threadLane < 2 )
      ptrs[warpLane][threadLane] = ia[line + threadLane];
/*same as: ia1 = Ap[row];*/
    ia1 = ptrs[warpLane][0];
/*same as: ia2 = Ap[row];*/
    ia2 = ptrs[warpLane][1];
/*... compute runnig sum per thread*/
    tmp = 0.0e0;
    for(ja1=ia1+threadLane;ja1<ia2;ja1+=WARP_SIZE){
      tmp += a[ja1] * x[ja[ja1]];
    }
    
    vals[threadIdx.x] = tmp;
    __syncthreads();
    vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x + 16];
    __syncthreads();
    vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x +  8];
     __syncthreads();
    vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x +  4];
    __syncthreads();
    vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x +  2];
    __syncthreads();
    vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x +  1];
     __syncthreads();


/*... firts thread writs the result*/
    if(threadLane == 0) y[line] = vals[threadIdx.x];

  }    

}
/*********************************************************************/ 

/********************************************************************* 
 * matVecCsrVectorDeviceLCuspV1: kernel do produto matriz vetor      *
 * CSR (VECTOR)                                                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq     -> numero de equacoes                                     * 
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> vetor x na GPU                                         * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devY -> produto da operacao de matriz vetor (GPU)                 * 
 *-------------------------------------------------------------------* 
 * OBS: Retirado da biblioteca Cusp V1                               * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
template <typename indexType   
         ,typename valueType
         ,unsigned int VECTORS_PER_BLOCK
         ,unsigned int THREADS_PER_VECTOR> 
__global__ void  matVecCsrVectorDeviceCuspV1(indexType const neq     
                            ,indexType const *ia  ,indexType const *ja
                            ,valueType const *a   ,valueType const *x
                            ,valueType *y){
  __shared__ volatile 
  valueType vals[VECTORS_PER_BLOCK*THREADS_PER_VECTOR+THREADS_PER_VECTOR/2];
  __shared__ volatile indexType ptrs[VECTORS_PER_BLOCK][2];
  const indexType THREADS_PER_BLOCK = VECTORS_PER_BLOCK * THREADS_PER_VECTOR;
  const indexType threadId    = THREADS_PER_BLOCK * blockIdx.x + threadIdx.x;    
  const indexType threadLane  = threadIdx.x & (THREADS_PER_VECTOR - 1);    
/*global warp index*/
  const indexType vectorId     = threadId   / THREADS_PER_VECTOR;
/*warp index within the CTA*/
  const indexType vectorLane   = threadIdx.x / THREADS_PER_VECTOR;
/* total number of active warps*/
  const indexType numVectors = VECTORS_PER_BLOCK * gridDim.x;  
/*thread index within the warp*/
  indexType line,ia1,ia2,ja1;
  valueType tmp;


  for( line = vectorId; line < neq; line+=numVectors)  {
/* use two threads to fetch ia[line] and ia[line+1]
   this is considerably faster than the more straightforward option*/
    if( threadLane < 2 )
      ptrs[vectorLane][threadLane] = ia[line + threadLane];
/*same as: ia1 = Ap[row];*/
    ia1 = ptrs[vectorLane][0];
/*same as: ia2 = Ap[row];*/
    ia2 = ptrs[vectorLane][1];
/*... compute runnig sum per thread*/
    tmp = 0.0e0;

//    if(THREADS_PER_VECTOR == 32 && ia2 - ia1 > 32){
/* ensure aligned memory access to ja and a*/
//      ja1 = ia1 - (ia1 & (THREADS_PER_VECTOR - 1)) + threadLane;

/* accumulate local sums*/
//      if( ja1 >= ia1 && ja1 < ia2)
//        tmp += a[ja1] * x[ja[ja1]];

/* accumulate local sums*/
//      for(ja1+=THREADS_PER_VECTOR; ja1<ia2 ;ja1+=THREADS_PER_VECTOR)
//        tmp += a[ja1] * x[ja[ja1]];
//    }
//    else{
/* accumulate local sums*/
      for(ja1=ia1+threadLane; ja1<ia2 ;ja1+=THREADS_PER_VECTOR)
        tmp += a[ja1] * x[ja[ja1]];
//    }
/* store local sum in shared memory*/
    vals[threadIdx.x] = tmp;
    __syncthreads();

/* reduce local sums to row sum*/
    if (THREADS_PER_VECTOR > 16) 
      vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x + 16];
    __syncthreads();
    if (THREADS_PER_VECTOR >  8) 
      vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x +  8];
    __syncthreads();
    if (THREADS_PER_VECTOR >  4) 
      vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x +  4];
    __syncthreads();
    if (THREADS_PER_VECTOR >  2) 
      vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x +  2];
    __syncthreads();
    if (THREADS_PER_VECTOR >  1) 
      vals[threadIdx.x] = tmp = tmp + vals[threadIdx.x +  1];
    __syncthreads();


/*... firts thread writs the result*/
    if(threadLane == 0) y[line] = vals[threadIdx.x];

  }    

}
/*********************************************************************/ 

/********************************************************************* 
 * ADDVECTORGPUCUDA: interface para chamar add de dois vetores em GPU* 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a       -> vetor a                                                * 
 * b       -> vetor b                                                * 
 * c       -> nao definido                                           * 
 * nDim    -> dimensao do vetores                                    * 
 * nBlocks -> numero de blocos                                       * 
 * nThreads-> numero de threads por blocos                           * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devY -> produto da operacao de matriz vetor (GPU)                 * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void addVectorGpuCuda(DOUBLE *a,DOUBLE *b,DOUBLE *c, int const nDim
                     ,int const nBlocks  ,int const nThreads){   

  DOUBLE *devA,*devB,*devC;

/*... alocacao de memoria na GPU*/
  HANDLE_ERROR_CUDA(cudaMalloc((void **)&devA,nDim*sizeof(DOUBLE)));
  HANDLE_ERROR_CUDA(cudaMalloc((void **)&devB,nDim*sizeof(DOUBLE)));
  HANDLE_ERROR_CUDA(cudaMalloc((void **)&devC,nDim*sizeof(DOUBLE)));
/*...................................................................*/

/*... copia dos vetores a e b para GPU*/
  HANDLE_ERROR_CUDA(cudaMemcpy(devA,a,nDim*sizeof(DOUBLE)
                   ,cudaMemcpyHostToDevice));
  HANDLE_ERROR_CUDA(cudaMemcpy(devB,b,nDim*sizeof(DOUBLE)
                   ,cudaMemcpyHostToDevice));
/*...................................................................*/

/*... chamada do kernell*/
  addVectorDevice<<<nBlocks,nThreads>>>(devA,devB,devC,nDim);
/*...................................................................*/

/*... copia dos vetores a e b para GPU*/
  HANDLE_ERROR_CUDA(cudaMemcpy(c,devC,nDim*sizeof(DOUBLE)
                   ,cudaMemcpyDeviceToHost));
/*...................................................................*/

/*... liberando a memoria da GPU*/
  cudaFree( devA );
  cudaFree( devB );
  cudaFree( devC );
/*...................................................................*/

}
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRGPUCUDASCALAR: interface para chamar o produto matriz    *
 * vetor no formato CSR (versao escalar)                             *
 * ------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq     -> numero de equacoes                                     * 
 * x       -> vetor a ser multiplicado                               * 
 * y       -> indefinido                                             * 
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> apenas a alocacao do vetor x na GPU                    * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 * nBlocks -> numero de blocos                                       * 
 * nThreads-> numero de threads por blocos                           * 
 * fUpX    -> atualizaca do vetor x (CPU->GPU) [0|1]                 * 
 * fUpY    -> atualizaca do vetor y (GPU->CPU) [0|1]                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devY -> produto da operacao de matriz vetor (GPU)                 * 
 * y    -> produto da operacao de matriz vetor (CPU) para fUpY = 1   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrGpuCudaScalar(INT neq     ,DOUBLE *x, DOUBLE *y
                     ,INT   **devIa,INT **devJa
                     ,DOUBLE **devA,DOUBLE **devX,DOUBLE **devY
                     ,int  nBlocks,int nThreads
                     ,char fUpX   ,char fUpY         ){   
  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  
  if( (deviceProp.maxThreadsPerBlock < nThreads) && 
    (MAX_THREADS_PER_BLOCK< nThreads) ){
    fprintf(stderr,"Numero de threads por blocos excedidos.\n");
    fprintf(stderr,"Numero Maximo de threads por blocos %d.\n"
           ,deviceProp.maxThreadsPerBlock);
    exit(EXIT_FAILURE);
  } 
    

/*... copia o vetor x-> devX (GPU) */
  if(fUpX)
    HANDLE_ERROR_CUDA(cudaMemcpy(*devX,x,neq*sizeof(DOUBLE)
                     ,cudaMemcpyHostToDevice));
/*...................................................................*/

/*... zero vetor devY no device*/
  zeroVecDevice<<<nBlocks,nThreads>>>(*devY,neq);
/*...................................................................*/

/*... chamada do kernell*/
  matVecCsrScalarDevice<<<nBlocks,nThreads>>>(neq,*devIa,*devJa
                                             ,*devA,*devX,*devY);
/*...................................................................*/

/*... copia o vetor devY-> y (CPU)*/
  if(fUpY)
    HANDLE_ERROR_CUDA(cudaMemcpy(y,*devY,neq*sizeof(DOUBLE)
                     ,cudaMemcpyDeviceToHost));
/*...................................................................*/

}
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRGPUCUDAVECTOR: interface para chamar o produto matriz    *
 * vetor no formato CSR (versao vetorial)                            *
 * ------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq     -> numero de equacoes                                     * 
 * x       -> vetor a ser multiplicado                               * 
 * y       -> indefinido                                             * 
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> apenas a alocacao do vetor x na GPU                    * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 * nBlocks -> numero de blocos                                       * 
 * nThreads-> numero de threads por blocos                           * 
 * fUpX    -> atualizaca do vetor x (CPU->GPU) [0|1]                 * 
 * fUpY    -> atualizaca do vetor y (GPU->CPU) [0|1]                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devY -> produto da operacao de matriz vetor (GPU)                 * 
 * y    -> produto da operacao de matriz vetor (CPU) para fUpY = 1   * 
 *-------------------------------------------------------------------* 
 * OBS: nThreads (2,4,8,16,32,64,128,256,512,1014)                   * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrGpuCudaVector(INT neq     ,DOUBLE *x, DOUBLE *y
                     ,INT   **devIa,INT **devJa
                     ,DOUBLE **devA,DOUBLE **devX,DOUBLE **devY
                     ,int  nBlocks,int nThreads
                     ,char fUpX   ,char fUpY         ){   
  
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);
  const int threadPerBlock = 128;
  if( (deviceProp.maxThreadsPerBlock < nThreads) && 
    (MAX_THREADS_PER_BLOCK< nThreads) ){
    fprintf(stderr,"Numero de threads por blocos excedidos.\n");
    fprintf(stderr,"Numero Maximo de threads por blocos %d.\n"
           ,deviceProp.maxThreadsPerBlock);
    exit(EXIT_FAILURE);
  } 
    
/*... copia o vetor x-> devX (GPU) */
  if(fUpX)
    HANDLE_ERROR_CUDA(cudaMemcpy(*devX,x,neq*sizeof(DOUBLE)
                     ,cudaMemcpyHostToDevice));
/*...................................................................*/

/*... zero vetor devY no device*/
//  zeroVecDevice<<<nBlocks,nThreads>>>(*devY,neq);
/*...................................................................*/

/*... chamada do kernell*/
  matVecCsrVectorDeviceL32
  <<<nBlocks,threadPerBlock,threadPerBlock*8>>>
  (neq,*devIa,*devJa,*devA,*devX,*devY);
/*...................................................................*/

/*... copia o vetor devY-> y (CPU)*/
  if(fUpY)
    HANDLE_ERROR_CUDA(cudaMemcpy(y,*devY,neq*sizeof(DOUBLE)
                     ,cudaMemcpyDeviceToHost));
/*...................................................................*/

}
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRGPUCUDAVECTORCUSP: interface para chamar o produto matriz*
 * vetor no formato CSR (versao vetorial)                            *
 * ------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq     -> numero de equacoes                                     * 
 * x       -> vetor a ser multiplicado                               * 
 * y       -> indefinido                                             * 
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> apenas a alocacao do vetor x na GPU                    * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 * nBlocks -> numero de blocos                                       * 
 * fUpX    -> atualizaca do vetor x (CPU->GPU) [0|1]                 * 
 * fUpY    -> atualizaca do vetor y (GPU->CPU) [0|1]                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devY -> produto da operacao de matriz vetor (GPU)                 * 
 * y    -> produto da operacao de matriz vetor (CPU) para fUpY = 1   * 
 *-------------------------------------------------------------------* 
 * OBS: nThreads (2,4,8,16,32,64,128,256,512,1014)                   * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrGpuCudaVectorCusp(INT neq     
                     ,DOUBLE *x    , DOUBLE *y
                     ,INT   **devIa,INT **devJa
                     ,DOUBLE **devA,DOUBLE **devX,DOUBLE **devY
                     ,int  nBlocks ,char fUpX    ,char fUpY         
                     ,int  code ){   
  
  const unsigned int THREADS_PER_BLOCK  = 128;
  const unsigned int VECTORS_PER_BLOCK  = THREADS_PER_BLOCK / 32;

/*... copia o vetor x-> devX (GPU) */
  if(fUpX)
    HANDLE_ERROR_CUDA(cudaMemcpy(*devX,x,neq*sizeof(DOUBLE)
                     ,cudaMemcpyHostToDevice));
/*...................................................................*/

/*... zero vetor devY no device*/
    zeroVecDevice<<<nBlocks,256>>>(*devY,neq);
/*...................................................................*/

/*... chamada do kernell*/
   if(code == 0) 
       matVecCsrVectorDeviceCuspRt2008
       <INT,DOUBLE,THREADS_PER_BLOCK,32>
       <<<nBlocks,THREADS_PER_BLOCK>>>
       (neq,*devIa,*devJa,*devA,*devX,*devY);
   else if(code == 1) 
       matVecCsrVectorDeviceCuspSc2009
       <INT,DOUBLE,THREADS_PER_BLOCK,32>
       <<<nBlocks,THREADS_PER_BLOCK>>>
       (neq,*devIa,*devJa,*devA,*devX,*devY);
   else if(code == 2) 
       matVecCsrVectorDeviceCuspSc2009
       <INT,DOUBLE,THREADS_PER_BLOCK,VECTORS_PER_BLOCK>
       <<<nBlocks,THREADS_PER_BLOCK>>>
       (neq,*devIa,*devJa,*devA,*devX,*devY);
/*...................................................................*/

/*... copia o vetor devY-> y (CPU)*/
  if(fUpY)
    HANDLE_ERROR_CUDA(cudaMemcpy(y,*devY,neq*sizeof(DOUBLE)
                     ,cudaMemcpyDeviceToHost));
/*...................................................................*/

}
/*********************************************************************/ 

/********************************************************************* 
 * INITCSRGPUCUDA: interface para chamar incializar a estrutura CSR  *
 * na CPU                                                            *
 * ------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------*
 * neq     -> numero de equacoes                                     * 
 * ia      -> vetor csr                                              * 
 * ja      -> vetor csr                                              * 
 * a       -> vetor com os valores da matriz                         * 
 * x       -> vetor a ser multiplicado                               * 
 * y       -> indefinido                                             * 
 * devIa   -> nao definido                                           * 
 * devJa   -> nao definido                                           * 
 * devA    -> nao definido                                           * 
 * devX    -> nao definido                                           * 
 * devY    -> nao definido                                           * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> apenas a alocacao do vetor x na GPU                    * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void initCsrGpuCuda(INT neq     
                   ,INT *ia            ,INT *ja
                   ,DOUBLE *a        
                   ,DOUBLE *x          ,DOUBLE *y
                   ,INT   **devIa      ,INT **devJa
                   ,DOUBLE **devA      ,DOUBLE **devX
                   ,DOUBLE **devY){

  INT nad = ia[neq];

/*...*/  
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
//  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//  cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//  cudaDeviceSetCacheConfig(cudaFuncCachePreferEqual);
/*...................................................................*/

/*... alocacao de memoria na GPU (devIa,devJa,devA,devX,devY)*/
  HANDLE_ERROR_CUDA(cudaMalloc((void **)devIa,(neq+1)*sizeof(INT)));
  HANDLE_ERROR_CUDA(cudaMalloc((void **)devJa,    nad*sizeof(INT)));
  HANDLE_ERROR_CUDA(cudaMalloc((void **)devA , nad*sizeof(DOUBLE)));
  HANDLE_ERROR_CUDA(cudaMalloc((void **)devX , neq*sizeof(DOUBLE)));
  HANDLE_ERROR_CUDA(cudaMalloc((void **)devY , neq*sizeof(DOUBLE)));
/*...................................................................*/

/*... copia CPU -> GPU (devIa,devJa,devA)*/
  HANDLE_ERROR_CUDA(cudaMemcpy(*devIa,ia,(neq+1)*sizeof(INT)
                   ,cudaMemcpyHostToDevice));
  HANDLE_ERROR_CUDA(cudaMemcpy(*devJa,ja,nad*sizeof(INT)
                   ,cudaMemcpyHostToDevice));
  HANDLE_ERROR_CUDA(cudaMemcpy(*devA,a,nad*sizeof(DOUBLE)
                   ,cudaMemcpyHostToDevice));
/*...................................................................*/
}
/*********************************************************************/ 

/********************************************************************* 
 * FINALIZECSRGPUCUDA: interface para chamar finalizar a estrutura   *
 * CSR na GPU                                                        *
 * ------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------*
 * devIa   -> vetor ia na GPU                                        * 
 * devJa   -> vetor ja na GPU                                        * 
 * devA    -> vetor a na GPU                                         * 
 * devX    -> apenas a alocacao do vetor x na GPU                    * 
 * devY    -> apenas a alocacao do vetor y na GPU                    * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void finalizeCsrGpuCuda(INT   **devIa       ,INT **devJa
                       ,DOUBLE **devA       ,DOUBLE **devX
                       ,DOUBLE **devY){


/*... liberacao da memoria da GPU (devIa,devJa,devA,devX,devY)*/
  cudaFree( *devIa );
  cudaFree( *devJa );
  cudaFree( *devA  );
  cudaFree( *devX  );
  cudaFree( *devY  );
/*...................................................................*/

}
/*********************************************************************/ 



