#ifndef _CUDA_ 
  #include<stdio.h>
  #include<Define.h>
  #ifdef __cplusplus
    extern "C" {
  #endif
/*soma de vetores (level-1)*/
      void addVectorGpuCuda(DOUBLE *a,DOUBLE *b,DOUBLE *c
                           ,int const nDim,int const nBlocks  
                           ,int const nThreads); 
/*soma de vetores (level-2)*/
/*...CSR*/
/*inicializa da estrutura do CSR na GPU*/
      void initCsrGpuCuda(INT neq     
                         ,INT *ia            ,INT *ja
                         ,DOUBLE *a       
                         ,DOUBLE *x          ,DOUBLE *y
                         ,INT   **devIa      ,INT **devJa
                         ,DOUBLE **devA      ,DOUBLE **devX
                         ,DOUBLE **devY);
/*finalizacap da estrutura do CSR na GPU*/
      void finalizeCsrGpuCuda(INT   **devIa,INT **devJa
                             ,DOUBLE **devA,DOUBLE **devX
                             ,DOUBLE **devY);        
/*... produto matriz vetor*/
      void matVecCsrGpuCudaScalar(INT neq 
                           ,DOUBLE *x    , DOUBLE *y
                           ,INT   **devIa,INT **devJa
                           ,DOUBLE **devA,DOUBLE **devX,DOUBLE **devY
                           ,int nBlocks  ,int nThreads
                           ,char fUpX    ,char fUpY );  
      void matVecCsrGpuCudaVector(INT neq     
                           ,DOUBLE *x    , DOUBLE *y
                           ,INT   **devIa,INT **devJa
                           ,DOUBLE **devA,DOUBLE **devX,DOUBLE **devY
                           ,int nBlocks  ,int nThreads
                           ,char fUpX    ,char fUpY );  
      void matVecCsrGpuCudaVectorCusp(INT neq     
                           ,DOUBLE *x    , DOUBLE *y
                           ,INT   **devIa,INT **devJa
                           ,DOUBLE **devA,DOUBLE **devX,DOUBLE **devY
                           ,int nBlocks  ,char fUpX    ,char fUpY 
                           ,int  code);       
/*...................................................................*/
  #ifdef __cplusplus
    }
  #endif
  
#endif
