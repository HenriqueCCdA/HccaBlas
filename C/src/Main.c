#include<stdlib.h>
#include<HccaTime.h>
#include<stdio.h>
#include<Coo.h>
#include<Openmp.h>
#include<HccaRandom.h>
#include<HccaBlas.h>
#include<File.h>
#include<Memoria.h>
#include<Csr.h>

/*... OpenBlAS*/
#ifdef _OPENBLAS_
  #include<cblas.h>
#endif

/*... Atlas*/
#ifdef _ATLAS_
  #include<cblas.h>
#endif
/*...................................................................*/

/*... GNU BlAS*/
#ifdef _GSL_BLAS_
  #include<gsl/gsl_blas.h>
#endif
/*...................................................................*/

/*... MKL*/
#ifdef _MKL_
  #include<mkl.h>
#endif
/*...................................................................*/

#define NSAMPLES      200
#define  BLOCKS         6
#define IBLOCKS       128
#define NARGS           4
#define WORDSIZE      100
#define MATVECZERO    10e-9
#define MAX_NUM_THREADS 24

int main(int argc,char **argv){
 
#ifdef _MMIO_ 
  MM_typecode matCode;
#endif
  Memoria m;
  double alpha,beta;
  double *a=NULL,*x1=NULL,*y1=NULL,*y2=NULL;
  int *iLin=NULL,*iCol=NULL;
  double *val=NULL;
  double gFlop;
  double dDot1,dDot2;
  int nl,ncl,nnz,i,j;
/*... Csr*/
  typeCsr tyCsr;
  int     nadCsr,nadCsrD,nadCsrC;
  int    *iaCsr,*jaCsr;
  int    *iaCsrD,*jaCsrD;
  int    *iaCsrC,*jaCsrC;  
  double *aCsr;
  double *adCsrD,*aCsrD;
  double *adCsrC,*auCsrC,*alCsrC;  
  int    *aux;
  double  gFlopCsr,gFlopCsrD,gFlopCsrC;
  long    bandMax,bandMin,bandMed;
  long    nlMax,nlMin,nlMed;
#ifdef _MKL_
  MKL_INT    *iaMkl,*jaMkl;
  char trans;
#endif
  double timeMklCsr = 0.0;
/*...................................................................*/

/*...Time*/
  double timeBlas           = 0.0;
  double timeMyBlasFull     = 0.0;
  double timeMyBlasFullO2   = 0.0;
  double timeMyBlasFullO4   = 0.0;
  double timeMyBlasFullO2I2 = 0.0;
  double timeMyBlasFullO4I2 = 0.0;
  double timeMyBlasFullO4I4 = 0.0;
  double timeMyCsr          = 0.0;
  double timeMyCsrO2        = 0.0;
  double timeMyCsrO4        = 0.0;
  double timeMyCsrO6        = 0.0;
  double timeMyCsrI2        = 0.0;
  double timeMyCsrI4        = 0.0;
  double timeMyCsrI6        = 0.0;
  double timeMyCsrO2I2      = 0.0;
  double timeMyCsrD         = 0.0;
  double timeMyCsrDI2       = 0.0;
  double timeMyCsrDI4       = 0.0;
  double timeMyCsrDI6       = 0.0;
  double timeMyCsrDO2       = 0.0;
  double timeMyCsrDO4       = 0.0;
  double timeMyCsrDO6       = 0.0;
  double timeMyCsrDO2I2     = 0.0;
  double timeMyCsrC         = 0.0;
  double timeMyCsrCI2       = 0.0;
  double timeMyCsrCI4       = 0.0;
  double timeMyCsrCI6       = 0.0;
  double timeMyCsrCO2       = 0.0;
  double timeMyCsrCO4       = 0.0;
  double timeMyCsrCO6       = 0.0;
  double timeMyCsrCO2I2     = 0.0;
  double timeMyDotL2        = 0.0;
  double timeMyDotL4        = 0.0;
  double timeMyDotL6        = 0.0;
  double timeMyDotL8        = 0.0;
  double timeMyDotO2        = 0.0;
  double timeMyDotO4        = 0.0;
  double timeMyDotO6        = 0.0;
  double timeMyDotO8        = 0.0;
  double timeMyDotO2L2      = 0.0;
  double timeMyDot          = 0.0;
  double timeDot            = 0.0;
  double timeMyCsrBest      = 0.0;
  double timeMyCsrDBest     = 0.0;
  double timeMyCsrCBest     = 0.0;
  char   nameCsr[30];
  char   nameCsrD[30];
  char   nameCsrC[30];

#ifdef _OPENMP
  double timeMyDotOmp       = 0.0;
  double timeMyDotOmpO2     = 0.0;
  double timeMyDotOmpO4     = 0.0;
  double timeMyDotOmpO6     = 0.0;
  double timeMyDotOmpO8     = 0.0;
  double timeMyDotOmpL2     = 0.0;
  double timeMyDotOmpL4     = 0.0;
  double timeMyDotOmpL6     = 0.0;
  double timeMyDotOmpL8     = 0.0;
  double timeMyDotOmpO2L2   = 0.0;
  double timeMyDotOmpO2L4   = 0.0;
  double      timeMyCsrOmpBest[MAX_NUM_THREADS]; 
  double     timeMyCsrDOmpBest[MAX_NUM_THREADS];
  double  timeMyCsrDBalOmpBest[MAX_NUM_THREADS];
  double     timeMyCsrCOmpBest[MAX_NUM_THREADS];
  double         timeMyCsrDOmp[MAX_NUM_THREADS];
  double       timeMyCsrDOmpI2[MAX_NUM_THREADS];
  double       timeMyCsrDOmpI4[MAX_NUM_THREADS];
  double       timeMyCsrDOmpI6[MAX_NUM_THREADS];
  double       timeMyCsrDOmpO2[MAX_NUM_THREADS];
  double       timeMyCsrDOmpO4[MAX_NUM_THREADS];
  double       timeMyCsrDOmpO6[MAX_NUM_THREADS];
  double     timeMyCsrDOmpO2I2[MAX_NUM_THREADS];
  double         timeMyCsrCOmp[MAX_NUM_THREADS];
  double       timeMyCsrCOmpI2[MAX_NUM_THREADS];
  double       timeMyCsrCOmpI4[MAX_NUM_THREADS];
  double       timeMyCsrCOmpI6[MAX_NUM_THREADS];
  double       timeMyCsrCOmpO2[MAX_NUM_THREADS];
  double       timeMyCsrCOmpO4[MAX_NUM_THREADS];
  double       timeMyCsrCOmpO6[MAX_NUM_THREADS];
  double     timeMyCsrCOmpO2I2[MAX_NUM_THREADS];
  double      timeMyCsrDOmpBal[MAX_NUM_THREADS];
  double    timeMyCsrDOmpBalI2[MAX_NUM_THREADS];
  double    timeMyCsrDOmpBalI4[MAX_NUM_THREADS];
  double    timeMyCsrDOmpBalI6[MAX_NUM_THREADS];
  double    timeMyCsrDOmpBalO2[MAX_NUM_THREADS];
  double    timeMyCsrDOmpBalO4[MAX_NUM_THREADS];
  double    timeMyCsrDOmpBalO6[MAX_NUM_THREADS];
  double  timeMyCsrDOmpBalO2I2[MAX_NUM_THREADS];
  double          timeMyCsrOmp[MAX_NUM_THREADS];
  double        timeMyCsrOmpI2[MAX_NUM_THREADS];
  double        timeMyCsrOmpI4[MAX_NUM_THREADS];
  double        timeMyCsrOmpI6[MAX_NUM_THREADS];
  double        timeMyCsrOmpO2[MAX_NUM_THREADS];
  double        timeMyCsrOmpO4[MAX_NUM_THREADS];
  double        timeMyCsrOmpO6[MAX_NUM_THREADS];
  double      timeMyCsrOmpO2I2[MAX_NUM_THREADS];
  double         timeMklCsrOmp[MAX_NUM_THREADS];
  double       timeOverHeadBal[MAX_NUM_THREADS];
  double       timeOverHeadCsrC[MAX_NUM_THREADS];
  char       nameCsrOmp[MAX_NUM_THREADS][30];
  char      nameCsrDOmp[MAX_NUM_THREADS][30];
  char   nameCsrDBalOmp[MAX_NUM_THREADS][30];
  char      nameCsrCOmp[MAX_NUM_THREADS][30];
  double       *bufferThY;
  int    nThreads,numTotalThreads;
/*int    nThBeginCsr[MAX_NUM_THREADS], nThEndCsr[MAX_NUM_THREADS];
  int     nThSizeCsr[MAX_NUM_THREADS];*/
  int   nThBeginCsrD[MAX_NUM_THREADS],   nThEndCsrD[MAX_NUM_THREADS];
  int    nThSizeCsrD[MAX_NUM_THREADS],nThHeightCsrD[MAX_NUM_THREADS];
  int   nThBeginCsrC[MAX_NUM_THREADS],   nThEndCsrC[MAX_NUM_THREADS];
  int    nThSizeCsrC[MAX_NUM_THREADS],nThHeightCsrC[MAX_NUM_THREADS];
#endif

#ifdef _CUDABLAS_
  #include<CudaHccaBlas.h>
  #include<InterfaceCuSparse.h>
  double timeCudaCsr           = 0.0;
  double timeCudaCsrEvent      = 0.0;
  double timeOverHeadCuda      = 0.0;
  double     timeMyCudaCsrScalar  = 0.0;
  double   timeMyCudaCsrScalar16[BLOCKS];
  double   timeMyCudaCsrScalar32[BLOCKS];
  double   timeMyCudaCsrScalar64[BLOCKS];
  double  timeMyCudaCsrScalar128[BLOCKS];
  double  timeMyCudaCsrScalar256[BLOCKS];
  double  timeMyCudaCsrVectorL2[BLOCKS];
  double  timeMyCudaCsrVectorL4[BLOCKS];
  double  timeMyCudaCsrVectorL8[BLOCKS];
  double timeMyCudaCsrVectorL16[BLOCKS];
  double timeMyCudaCsrVectorL32[BLOCKS];
//  double timeMyCudaCsrVectorL64[BLOCKS];
  double timeMyCudaCsrVectorCuspRt2008[BLOCKS];
  double timeMyCudaCsrVectorCuspSc2009[BLOCKS];
  double     timeMyCudaCsrVectorCuspV1[BLOCKS];
  double *devA=NULL,*devX=NULL,*devY=NULL;
  INT    *devIa=NULL,*devJa=NULL;
  int    nBlock;
#endif

  char flagRandom=0;
  char nameOut1[40],nameOut2[40],nameOut0[30],prename[30];
  FILE *fileIn,*fileOut;
  char nameArgs1[][WORDSIZE] = 
  {"-matVecFull","-dot","-matVecSparseCsr","-matVecSparseCsrCuda"};
  char nameArgs2[][WORDSIZE]
  = {"-matVecFull fileIn fileOut randomVectorDenseX[true|false]"
    ,"-dot nDim fileOut randomVectorDenseX[true|false]"
    ,"-matVecSparseCsr     fileIn fileOut randomVectorDenseX[true|false]"
    ,"-matVecSparseCsrCuda fileIn fileOut randomVectorDenseX[true|false]"};
/*... matvec*/
  void (*myMatVecFull)(double *restrict,double *restrict
                      ,double *restrict,INT,INT);
  void (*myMatVecCsr)(INT              
                     ,INT *restrict    ,INT *restrict
                     ,double *restrict ,double *restrict
                     ,double *restrict);
  void (*myMatVecCsrD)(INT              
                        ,INT *restrict    ,INT *restrict
                       ,double *restrict ,double *restrict
                       ,double *restrict ,double *restrict);
  void (*myMatVecCsrC  )(INT              
                        ,INT *restrict    ,INT *restrict
                       ,double *restrict ,double *restrict
                       ,double *restrict                  
                       ,double *restrict ,double *restrict);
  void (*myMatVecCsrDOmpBal)(INT              
                            ,INT *restrict    ,INT *restrict
                            ,double *restrict ,double *restrict
                            ,double *restrict ,double *restrict
                            ,int*            ,int*             );
  void (*myMatVecCsrCOmp)(INT              
                         ,INT *restrict    ,INT *restrict
                         ,double *restrict ,double *restrict 
                         ,double *restrict 
                         ,double *restrict ,double *restrict 
                         ,int *            ,int *  
                         ,int *    
                         ,double *restrict ,int );       
/*... dot*/
  double (*myDot)(double *restrict,double *restrict,INT); 
   
/*... */ 
  if(argc < 5){
    ERRO_ARGS(argv,NARGS,nameArgs2);
  }  
/*.....................................................................*/

/*... alocacao de memoria*/
  nmax=9000000000;
  initMem(&m,nmax,true);
/*.....................................................................*/

/* matriz cheia*/  
  if(!strcmp(argv[1],nameArgs1[0])){ 
/*...*/
    fileIn = openFile(argv[2],"r");
/*.....................................................................*/

/*... leitura de arquivo mtx*/  
#ifdef _MMIO_ 
    mm_read_banner(fileIn,&matCode);
    mm_read_mtx_crd_size(fileIn,&nl,&ncl,&nnz);
/*.....................................................................*/

/*... alocacao COO*/  
    hccaAlloc(double,&m,val ,nnz,"val" ,false);
    hccaAlloc(   int,&m,iCol,nnz,"iCol" ,false);
    hccaAlloc(   int,&m,iLin,nnz,"iLin" ,false);
  
    mm_read_mtx_crd_data(fileIn,nl,ncl,nnz,iLin,iCol,val,matCode);
#endif
/*.....................................................................*/

/*... alocacao A,x e y*/  
    hccaAlloc(double,&m,a ,(nl*nl),"a" ,false);
    hccaAlloc(double,&m,y1,nl     ,"y1" ,false);
    hccaAlloc(double,&m,y2,nl     ,"y2" ,false);
    hccaAlloc(double,&m,x1,nl     ,"x" ,false);
/*.....................................................................*/

/*... convertendo do COO para FULL matrix*/  
    cooToFull(iLin,iCol,val,a,nl,nnz);
/*.....................................................................*/
  
/*... gFlop*/
    gFlop = flopMatVecFull(nl,ncl)/1000000000.0;
/*.....................................................................*/

/*... geracao randomica do vetor denso x*/
    if(!strcmp(argv[4],"true"))
      flagRandom = 1;
    
    if(flagRandom)
      randomMatrix(x1,nl,0);
    else
      for(i = 0;i < nl;i++)
        x1[i] = 1.0;
/*.....................................................................*/

/*... openBlas*/
#ifdef _OPENBLAS_
    alpha = 1.0;
    beta  = 0.0;
    for(i = 0;i < NSAMPLES;i++){
      timeBlas  = getTimeC() - timeBlas;
      cblas_dgemv(CblasRowMajor
                  ,CblasNoTrans,nl,nl,alpha,a,nl,x1,1,beta,y1,1);
      timeBlas  = getTimeC() - timeBlas;
    }
#endif
/*...................................................................*/

/*... GNU BLAS*/
#ifdef _GSL_BLAS_
    alpha = 1.0;
    beta  = 0.0;
    for(i = 0;i < NSAMPLES;i++){
      timeBlas  = getTimeC() - timeBlas;
      cblas_dgemv(CblasRowMajor
                  ,CblasNoTrans,nl,nl,alpha,a,nl,x1,1,beta,y1,1);
      timeBlas  = getTimeC() - timeBlas;
    }
#endif
/*...................................................................*/

/*... MKL*/
#ifdef _MKL_
    alpha = 1.0;
    beta  = 0.0;
    for(i = 0;i < NSAMPLES;i++){
      timeBlas  = getTimeC() - timeBlas;
      cblas_dgemv(CblasRowMajor
                  ,CblasNoTrans,nl,nl,alpha,a,nl,x1,1,beta,y1,1);
      timeBlas  = getTimeC() - timeBlas;
    }
#endif
/*...................................................................*/

/*... ATLAS*/
#ifdef _ATLAS_
    alpha = 1.0;
    beta  = 0.0;
    for(i = 0;i < NSAMPLES;i++){
      timeBlas  = getTimeC() - timeBlas;
      cblas_dgemv(CblasRowMajor
                  ,CblasNoTrans,nl,nl,alpha,a,nl,x1,1,beta,y1,1);
      timeBlas  = getTimeC() - timeBlas;
    }
#endif
/*...................................................................*/
  
/* ... MyBlas*/
    zero(y2,nl,"double");
    myMatVecFull = &matVecFull;
    for(i = 0;i < NSAMPLES;i++){
      timeMyBlasFull   = getTimeC() - timeMyBlasFull;
      myMatVecFull(a,x1,y2,nl,nl);
      timeMyBlasFull   = getTimeC() - timeMyBlasFull;
      if(xDiffY(y2,y1,1.0e-8,nl)){
        printf("MateVecFull: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        for(i=0;i<nl;i++)
          fprintf(fileOut,"%d %20.8lf %20.8lf\n",i+1,y1[i],y2[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/
    
/* ... MyBlas O2*/
    zero(y2,nl,"double");
    myMatVecFull = &matVecFullO2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyBlasFullO2   = getTimeC() - timeMyBlasFullO2;
      myMatVecFull(a,x1,y2,nl,nl);
      timeMyBlasFullO2   = getTimeC() - timeMyBlasFullO2;
      if(xDiffY(y2,y1,1.0e-8,nl)){
        printf("MateVecFullO2: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        for(i=0;i<nl;i++)
          fprintf(fileOut,"%d %20.8lf %20.8lf\n",i+1,y1[i],y2[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/
    
/* ... MyBlas O4*/
    zero(y2,nl,"double");
    myMatVecFull = &matVecFullO4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyBlasFullO4   = getTimeC() - timeMyBlasFullO4;
      myMatVecFull(a,x1,y2,nl,nl);
      timeMyBlasFullO4   = getTimeC() - timeMyBlasFullO4;
      if(xDiffY(y2,y1,1.0e-8,nl)){
        printf("MateVecFullO4: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        for(i=0;i<nl;i++)
          fprintf(fileOut,"%d %20.8lf %20.8lf\n",i+1,y1[i],y2[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyBlas O2I2*/
    zero(y2,nl,"double");
    myMatVecFull = &matVecFullO2I2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyBlasFullO2I2 = getTimeC() - timeMyBlasFullO2I2;
      myMatVecFull(a,x1,y2,nl,nl);
      timeMyBlasFullO2I2 = getTimeC() - timeMyBlasFullO2I2;
      if(xDiffY(y2,y1,1.0e-8,nl)){
        printf("MateVecFullO2I2: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        for(i=0;i<nl;i++)
          fprintf(fileOut,"%d %20.8lf %20.8lf\n",i+1,y1[i],y2[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/
    
/* ... MyBlas O4I2*/
    zero(y2,nl,"double");
    myMatVecFull = &matVecFullO4I2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyBlasFullO4I2 = getTimeC() - timeMyBlasFullO4I2;
      myMatVecFull(a,x1,y2,nl,nl);
      timeMyBlasFullO4I2 = getTimeC() - timeMyBlasFullO4I2;
      if(xDiffY(y2,y1,1.0e-8,nl)){
        printf("MateVecFullO4I2: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        for(i=0;i<nl;i++)
          fprintf(fileOut,"%d %20.8lf %20.8lf\n",i+1,y1[i],y2[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/
    
/* ... MyBlas O4I4*/
    zero(y2,nl,"double");
    myMatVecFull = &matVecFullO4I4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyBlasFullO4I4 = getTimeC() - timeMyBlasFullO4I4;
      myMatVecFull(a,x1,y2,nl,nl);
      timeMyBlasFullO4I4 = getTimeC() - timeMyBlasFullO4I4;
      if(xDiffY(y2,y1,1.0e-8,nl)){
        printf("MateVecFullO4I4: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        for(i=0;i<nl;i++)
          fprintf(fileOut,"%d %20.8lf %20.8lf\n",i+1,y1[i],y2[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*...*/
    printf("  Blas      time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeBlas,(NSAMPLES*gFlop)/timeBlas);
    printf("MyBlas      time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyBlasFull  ,(NSAMPLES*gFlop)/timeMyBlasFull);
    printf("MyBlas O2   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyBlasFullO2,(NSAMPLES*gFlop)/timeMyBlasFullO2);
    printf("MyBlas O4   time = %16.8lf GFLOPS = %16.8lf\n"
         ,timeMyBlasFullO4,(NSAMPLES*gFlop)/(timeMyBlasFullO4));
    printf("MyBlas O2I2 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyBlasFullO2I2,(NSAMPLES*gFlop)/timeMyBlasFullO2I2);
    printf("MyBlas O4I2 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyBlasFullO4I2,(NSAMPLES*gFlop)/timeMyBlasFullO4I2);
    printf("MyBlas O4I4 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyBlasFullO4I4,(NSAMPLES*gFlop)/timeMyBlasFullO4I4);
/*...................................................................*/

/*... Produto da operacao y=Ax*/  
    fileOut = fopen(argv[3],"w");
    for(i=0;i<nl;i++)
      fprintf(fileOut,"%d %20.8lf %20.8lf\n",i+1,y1[i],y2[i]);
/*...................................................................*/

/*...*/
    hccaDealloc(&m,iCol,"iCol",false);
    hccaDealloc(&m,iLin,"iLin",false);
    hccaDealloc(&m,y1  ,"y1",false);
    hccaDealloc(&m,y2  ,"y2",false);
    hccaDealloc(&m,x1  ,"x1",false);
    hccaDealloc(&m,a   ,"a ",false);
/*...................................................................*/
    
/*...*/
    fclose(fileOut);
    fclose(fileIn);
/*...................................................................*/
  }
/*...................................................................*/

/*... produto interno*/
  else if(!strcmp(argv[1],nameArgs1[1])){
/*...*/
    nl = atol(argv[2]);
/*...................................................................*/

/*... gFlop*/
    gFlop = flopDot(nl)/1000000000.0;
/*.....................................................................*/

/*...*/
    hccaAlloc(double,&m,y1,nnz,"y1" ,false);
/*...................................................................*/

/*...*/
    hccaAlloc(double,&m,y2,nnz,"y2" ,false);
/*...................................................................*/

/*... geracao randomica do vetor denso x*/
    if(!strcmp(argv[4],"true"))
      flagRandom = 1;
    
    if(flagRandom){
      randomMatrix(y1,nl,0);
      randomMatrix(y2,nl,0);
    }
    else
      for(i = 0;i < nl;i++){
        y1[i] = 1.0;
        y2[i] = 1.0;
      }
/*.....................................................................*/

/*... produto interno da biblioteca BLAS*/
#ifdef _MKL_
    alpha = 1.0;
    for(i = 0;i < NSAMPLES;i++){
      timeDot   = getTimeC() - timeDot; 
      dDot1 = cblas_ddot(nl,y1,1,y2,1);
      timeDot   = getTimeC() - timeDot; 
    }
#endif
/*.....................................................................*/

/*... produto interno da biblioteca BLAS*/
#ifdef _OPENBLAS_
    alpha = 1.0;
    for(i = 0;i < NSAMPLES;i++){
      timeDot   = getTimeC() - timeDot; 
      dDot1 = cblas_ddot(nl,y1,1,y2,1);
      timeDot   = getTimeC() - timeDot; 
    }
#endif
/*.....................................................................*/

/*... produto interno*/
    myDot        = &dot;            
    for(i = 0;i < NSAMPLES;i++){
      timeMyDot = getTimeC() - timeMyDot;
      dDot2 = myDot(y1,y2,nl);
      timeMyDot = getTimeC() - timeMyDot;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dot: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno O2*/
    myDot        = &dotO2;            
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotO2 = getTimeC() - timeMyDotO2;
      dDot2 = myDot(y1,y2,nl);
      timeMyDotO2 = getTimeC() - timeMyDotO2;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotO2: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno O4*/
    myDot        = &dotO4;            
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotO4 = getTimeC() - timeMyDotO4;
      dDot2 = myDot(y1,y2,nl);
      timeMyDotO4 = getTimeC() - timeMyDotO4;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotO4: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno O6*/
    myDot        = &dotO6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotO6 = getTimeC() - timeMyDotO6;
      dDot2 = myDot(y1,y2,nl);
      timeMyDotO6 = getTimeC() - timeMyDotO6;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotO6: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno O8*/
    myDot        = &dotO8;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotO8 = getTimeC() - timeMyDotO8;
      dDot2 = myDot(y1,y2,nl);
      timeMyDotO8 = getTimeC() - timeMyDotO8;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotO8: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno L2*/
    myDot        = &dotL2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotL2 = getTimeC() - timeMyDotL2;
      dDot2 = myDot(y1,y2,nl);
      timeMyDotL2 = getTimeC() - timeMyDotL2;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotL2: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno L4*/
    myDot        = &dotL4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotL4 = getTimeC() - timeMyDotL4;
      dDot2 = myDot(y1,y2,nl);
      timeMyDotL4 = getTimeC() - timeMyDotL4;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotL4: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno L6*/
    myDot        = &dotL6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotL6 = getTimeC() - timeMyDotL6;
      dDot2 = myDot(y1,y2,nl);
      timeMyDotL6 = getTimeC() - timeMyDotL6;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotL6: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno L8*/
    myDot        = &dotL8;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotL8 = getTimeC() - timeMyDotL8;
      dDot2 = myDot(y1,y2,nl);
      timeMyDotL8 = getTimeC() - timeMyDotL8;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotL8: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno O2L2*/
    myDot        = &dotO2L2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotO2L2 = getTimeC() - timeMyDotO2L2;
      dDot2 = myDot(y1,y2,nl);
      timeMyDotO2L2 = getTimeC() - timeMyDotO2L2;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotO2L2: Vetores diferentes\n"); 
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

#ifdef _OPENMP
/*... produto interno OMP*/
    myDot        = &dotOmp;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmp  = getTimeC() - timeMyDotOmp; 
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmp  = getTimeC() - timeMyDotOmp;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmp: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/
    
/*... produto interno OMPO2*/
    myDot        = &dotOmpO2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpO2= getTimeC() - timeMyDotOmpO2;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpO2 = getTimeC() - timeMyDotOmpO2;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpO2: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno OMPO4*/
    myDot        = &dotOmpO4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpO4= getTimeC() - timeMyDotOmpO4;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpO4 = getTimeC() - timeMyDotOmpO4;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpO4: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno OMPO6*/
    myDot        = &dotOmpO6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpO6= getTimeC() - timeMyDotOmpO6;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpO6 = getTimeC() - timeMyDotOmpO6;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpO6: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno OMPO8*/
    myDot        = &dotOmpO8;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpO8= getTimeC() - timeMyDotOmpO8;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpO8 = getTimeC() - timeMyDotOmpO8;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpO8: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno OMPL2*/
    myDot        = &dotOmpL2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpL2= getTimeC() - timeMyDotOmpL2;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpL2 = getTimeC() - timeMyDotOmpL2;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpL2: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno OMPL4*/
    myDot        = &dotOmpL4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpL4= getTimeC() - timeMyDotOmpL4;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpL4 = getTimeC() - timeMyDotOmpL4;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpL4: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno OMPL6*/
    myDot        = &dotOmpL6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpL6= getTimeC() - timeMyDotOmpL6;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpL6 = getTimeC() - timeMyDotOmpL6;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpL6: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno OMPL8*/
    myDot        = &dotOmpL8;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpL8= getTimeC() - timeMyDotOmpL8;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpL8 = getTimeC() - timeMyDotOmpL8;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpL8: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno OMPO2L2*/
    myDot        = &dotOmpO2L2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpO2L2= getTimeC() - timeMyDotOmpO2L2;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpO2L2 = getTimeC() - timeMyDotOmpO2L2;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpL8: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/*... produto interno OMPO2L4*/
    myDot        = &dotOmpO2L4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyDotOmpO2L4= getTimeC() - timeMyDotOmpO2L4;
      #pragma omp parallel shared(y1,y2,nl) 
      dDot2 = myDot(y1,y2,nl);
      timeMyDotOmpO2L4 = getTimeC() - timeMyDotOmpO2L4;
      if(xDiffY(&dDot1,&dDot2,1.0e-4,1)){
        printf("dotOmpL8: Vetores diferentes\n");
        fileOut = fopen(argv[3],"w");
/*...*/
        fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

#endif

/*...*/
    printf("  Blas      time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeDot         ,(NSAMPLES*gFlop)/timeDot);
    printf("MyBlas      time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDot       ,(NSAMPLES*gFlop)/timeMyDot);
    printf("MyBlas O2   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotO2     ,(NSAMPLES*gFlop)/timeMyDotO2);
    printf("MyBlas O4   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotO4     ,(NSAMPLES*gFlop)/timeMyDotO4);
    printf("MyBlas O6   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotO6     ,(NSAMPLES*gFlop)/timeMyDotO6);
    printf("MyBlas O8   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotO8     ,(NSAMPLES*gFlop)/timeMyDotO8);
    printf("MyBlas L2   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotL2     ,(NSAMPLES*gFlop)/timeMyDotL2);
    printf("MyBlas L4   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotL4     ,(NSAMPLES*gFlop)/timeMyDotL4);
    printf("MyBlas L6   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotL6     ,(NSAMPLES*gFlop)/timeMyDotL6);
    printf("MyBlas L8   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotL8     ,(NSAMPLES*gFlop)/timeMyDotL8);
    printf("MyBlas O2L2 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotO2L2   ,(NSAMPLES*gFlop)/timeMyDotO2L2);
#ifdef _OPENMP
    printf("MyBlas   Omp   time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmp    ,(NSAMPLES*gFlop)/timeMyDotOmp );
    printf("MyBlas   OmpO2 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpO2  ,(NSAMPLES*gFlop)/timeMyDotOmpO2);
    printf("MyBlas   OmpO4 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpO4  ,(NSAMPLES*gFlop)/timeMyDotOmpO4);
    printf("MyBlas   OmpO6 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpO6  ,(NSAMPLES*gFlop)/timeMyDotOmpO6);
    printf("MyBlas   OmpO8 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpO8  ,(NSAMPLES*gFlop)/timeMyDotOmpO8);
    printf("MyBlas   OmpL2 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpL2  ,(NSAMPLES*gFlop)/timeMyDotOmpL2);
    printf("MyBlas   OmpL4 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpL4  ,(NSAMPLES*gFlop)/timeMyDotOmpL4);
    printf("MyBlas   OmpL6 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpL6  ,(NSAMPLES*gFlop)/timeMyDotOmpL6);
    printf("MyBlas   OmpL8 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpL8  ,(NSAMPLES*gFlop)/timeMyDotOmpL8);
    printf("MyBlas OmpO2L2 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpO2L2  ,(NSAMPLES*gFlop)/timeMyDotOmpO2L2);
    printf("MyBlas OmpO2L4 time = %16.8lf GFLOPS = %16.8lf\n"
          ,timeMyDotOmpO2L4  ,(NSAMPLES*gFlop)/timeMyDotOmpO2L4);
#endif
/*...................................................................*/

/*... Produto da operacao y=x1*x1*/  
    fileOut = fopen(argv[3],"w");
    fprintf(fileOut,"%20.8lf %20.8lf\n",dDot1,dDot2);
/*...................................................................*/

/*...*/
    hccaDealloc(&m,y1,"y1",false);
    hccaDealloc(&m,y2,"y2",false);
/*...................................................................*/

  } 
/*...................................................................*/

/* matriz sparsa csr*/  
  else if(!strcmp(argv[1],nameArgs1[2])){ 
/*...*/
    strcpy(prename,argv[3]);
    strcpy(nameOut0,prename);
    strcpy(nameOut1,prename);
    strcpy(nameOut2,prename);
    strcat(nameOut0,"_bestTime.txt");
    strcat(nameOut1,"_matvec.txt");
    strcat(nameOut2,"_time.txt");
/*.....................................................................*/

/*...*/
    fileIn = openFile(argv[2],"r");
/*.....................................................................*/

/*... leitura de arquivo mtx*/  
    fprintf(stderr,"Lendo o arquivo.\n");
#ifdef _MMIO_
    mm_read_banner(fileIn,&matCode);
    mm_read_mtx_crd_size(fileIn,&nl,&ncl,&nnz);
/*.....................................................................*/

/*... alocacao COO*/  
    hccaAlloc(double,&m,val,nnz,"val",false);
    hccaAlloc(int,&m,iLin,nnz,"iLin" ,false);
    hccaAlloc(int,&m,iCol,nnz,"iCol" ,false);
    zero(val ,nnz,"double");
    zero(iLin,nnz,"int");
    zero(iCol,nnz,"int");
  
    mm_read_mtx_crd_data(fileIn,nl,ncl,nnz,iLin,iCol,val,matCode);
    fprintf(stderr,"Arquivo lido.\n");
#endif
/*.....................................................................*/

/*...*/
/*  matrixCheck(val,iLin,iCol,nl,nnz);*/
/*.....................................................................*/

/*...*/
    nadCsr    = nnz; 
    nadCsrD = nnz - nl; 
    nadCsrC   = (nnz - nl)/2; 
/*.....................................................................*/

/*... alocacao ACsr, iaCSr, jaCsr*/ 
    hccaAlloc(double,&m,aCsr , nadCsr,"aCsr"  ,false);
    hccaAlloc(int   ,&m,iaCsr,(nl+1) ,"iaCsr" ,false);
    hccaAlloc(int   ,&m,jaCsr ,nadCsr,"jaCsr" ,false);
    zero(aCsr    ,nadCsr,"double");
    zero(iaCsr   ,(nl+1),"int");
    zero(jaCsr   ,nadCsr,"int");

/*... alocacao ACsrD, iaCsrD, jaCsrD*/ 
    hccaAlloc(double,&m,aCsrD ,nadCsrD   ,"aCsrD"  ,false);
    hccaAlloc(double,&m,adCsrD,nl        ,"adCsrD" ,false);
    hccaAlloc(int   ,&m,iaCsrD,(nl+1)    ,"iaCsrD" ,false);
    hccaAlloc(int   ,&m,jaCsrD,nadCsrD   ,"jaCsrD" ,false);
    zero(aCsrD ,nadCsrD,"double");
    zero(adCsrD,       nl,"double");
    zero(iaCsrD,   (nl+1),"int"   );
    zero(jaCsrD,nadCsrD,"int"   );

/*... alocacao ACsrC, iaCsrC, jaCsrC*/ 
    hccaAlloc(double,&m,auCsrC   ,nadCsrC    ,"auCsrC"   ,false);
    hccaAlloc(double,&m,alCsrC   ,nadCsrC    ,"alCsrC"   ,false);
    hccaAlloc(double,&m,adCsrC  ,nl          ,"adCsrC"   ,false);
    hccaAlloc(int   ,&m,iaCsrC  ,(nl+1)      ,"iaCsrC"   ,false);
    hccaAlloc(int   ,&m,jaCsrC  ,nadCsrC     ,"jaCsrC"   ,false);
    zero(auCsrC,nadCsrC,"double");
    zero(alCsrC,nadCsrC,"double");
    zero(adCsrC,     nl,"double");
    zero(iaCsrC, (nl+1),"int"   );
    zero(jaCsrC,nadCsrC,"int"   );


/*... alocacao iaMkl, jaMkl*/ 
#ifdef _MKL_ 
    hccaAlloc(MKL_INT,&m,iaMkl    ,(nl+1),"iaMkl"  ,false);
    hccaAlloc(MKL_INT,&m,jaMkl    ,nadCsr,"jaMkl"  ,false);
#endif

/*... x, y1 e y2*/
    hccaAlloc(double,&m,x1   ,nl    ,"x1" ,false);
    hccaAlloc(double,&m,y1   ,nl    ,"y1" ,false);
    hccaAlloc(double,&m,y2   ,nl    ,"y2" ,false);
//  mapVector(&m);
/*.....................................................................*/

/*... convertendo do COO para CSR*/  
    fprintf(stderr,"Coo -> Csr\n");
    hccaAlloc(int,&m,aux  ,nl    ,"aux",false);
/*... CSR*/
    tyCsr = csr;
    cooToCsr(iLin  ,iCol  ,val
            ,iaCsr ,jaCsr
            ,aCsr  ,aCsr  ,aCsr 
            ,nl    ,nnz   ,tyCsr
            ,aux   ,true  ,true  ,true);
/*... CSRD*/
    tyCsr = csrD;  
    cooToCsr(iLin     ,iCol  ,val
            ,iaCsrD ,jaCsrD
            ,aCsrD  ,adCsrD  ,aCsrD 
            ,nl       ,nnz       ,tyCsr
            ,aux   ,true  ,false  ,true);
/*... CSRC  */
    tyCsr = csrC;   
    cooToCsr(iLin     ,iCol  ,val
            ,iaCsrC   ,jaCsrC  
            ,auCsrC   ,adCsrC    ,alCsrC   
            ,nl       ,nnz       ,tyCsr
            ,aux      ,false ,false  ,true);
/*.....................................................................*/
    hccaDealloc(&m,aux,"aux",false);
  
/*... gFlop*/
    tyCsr = csr;
    gFlopCsr    = flopMatVecCsr(nl,nadCsr ,tyCsr)/1000000000.0;
    tyCsr = csrD;  
    gFlopCsrD   = flopMatVecCsr(nl,nadCsrD,tyCsr)/1000000000.0;
    tyCsr = csrC;
    gFlopCsrC   = flopMatVecCsr(nl,nadCsrC,tyCsr)/1000000000.0;
/*.....................................................................*/

/*... banda da mariz*/
     bandMax   = bandCsr(iaCsrD,jaCsrD,nl,1);
     bandMed   = bandCsr(iaCsrD,jaCsrD,nl,2);
     bandMin   = bandCsr(iaCsrD,jaCsrD,nl,3);
     printf("banda Max   = %ld\n",bandMax);
     printf("banda Media = %ld\n",bandMed);
     printf("banda Min   = %ld\n",bandMin);
/*.....................................................................*/

/*... numero de elementos nao nulos por linha da mariz*/
     nlMax = nlCsr(iaCsrD,nl,1);
     nlMed = nlCsr(iaCsrD,nl,2);
     nlMin = nlCsr(iaCsrD,nl,3);
     printf("nl Max   = %ld\n",nlMax);
     printf("nl Med   = %ld\n",nlMed);
     printf("nl Min   = %ld\n",nlMin);
/*.....................................................................*/

/*... geracao randomica do vetor denso x*/
    if(!strcmp(argv[4],"true"))
      flagRandom = 1;
    if(flagRandom)
      randomMatrix(x1,nl,0);
    else
      for(i = 0;i < nl;i++){
        if(i%2)
          x1[i] =  fmod(i+1,10.0) + 0.2;
        else
          x1[i] = -fmod(i+1,5)    - 0.1;
      }
/*.....................................................................*/

/*...*/
#ifdef _OPENMP
    numTotalThreads = omp_get_max_threads();
    omp_set_num_threads(1);
#ifdef _MKL_
    for(i = 0;i < nl+1;i++){
      iaMkl[i] = iaCsr[i] + 1;
    }
    for(i = 0;i < nadCsr;i++){
      jaMkl[i] = jaCsr[i] + 1;
    } 
    trans = 'N';
    for(nThreads=1;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      fprintf(stderr,"MklCsr %d\n",omp_get_max_threads());
      zero(y1,nl,"double");
      timeMklCsrOmp[nThreads-1]=0.0;
      for(i = 0;i < NSAMPLES;i++){
        timeMklCsrOmp[nThreads-1]=getTimeC()-timeMklCsrOmp[nThreads-1];
        mkl_dcsrgemv (&trans,&nl,aCsr,iaMkl,jaMkl,x1,y1);
        timeMklCsrOmp[nThreads-1]=getTimeC()-timeMklCsrOmp[nThreads-1];
      }
    }
    timeMklCsr = timeMklCsrOmp[0];
#endif
    for(i = 0;i < nl+1;i++){
      iaMkl[i] = iaCsr[i] + 1;
    }
    for(i = 0;i < nadCsr;i++){
      jaMkl[i] = jaCsr[i] + 1;
    } 
    trans = 'N';
    zero(y1,nl,"double");
    timeMklCsr = 0.0;
    for(i = 0;i < NSAMPLES;i++){
      timeMklCsr=getTimeC()-timeMklCsr;
      mkl_dcsrgemv (&trans,&nl,aCsr,iaMkl,jaMkl,x1,y1);
      timeMklCsr=getTimeC()-timeMklCsr;
    }
#endif

#ifdef _CUDABLAS_

/*...*/
    printf("cudaCsr\n");

/*...*/
    initCuSparse(nl  ,iaCsr            ,jaCsr 
               ,aCsr ,&timeOverHeadCuda); 
/*...................................................................*/

    for(i = 0;i < NSAMPLES;i++){
/*...*/ 
      timeCudaCsr = getTimeC() - timeCudaCsr;
/*...................................................................*/
     
/*...*/
      csrCuSparse(nl               ,nadCsr
                 ,x1               ,y2   
                 ,&timeOverHeadCuda,&timeCudaCsrEvent); 
/*...................................................................*/

/*...*/ 
      timeCudaCsr = getTimeC() - timeCudaCsr;
/*...................................................................*/

      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("cuSparse: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

#endif
/* ... MyCsr*/
    printf("My Csr\n");
    zero(y2,nl,"double");
    myMatVecCsr = &matVecCsr;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsr   = getTimeC() - timeMyCsr;
      myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
      timeMyCsr   = getTimeC() - timeMyCsr;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsr: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrO2*/
    printf("My CsrO2\n");
    zero(y2,nl,"double");
    myMatVecCsr = &matVecCsrO2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrO2 = getTimeC() - timeMyCsrO2;
      myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
      timeMyCsrO2 = getTimeC() - timeMyCsrO2;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrO2: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrO4*/
    printf("My CsrO4\n");
    zero(y2,nl,"double");
    myMatVecCsr = &matVecCsrO4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrO4 = getTimeC() - timeMyCsrO4;
      myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
      timeMyCsrO4 = getTimeC() - timeMyCsrO4;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrO4: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrO6*/
    printf("My CsrO6\n");
    zero(y2,nl,"double");
    myMatVecCsr = &matVecCsrO6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrO6 = getTimeC() - timeMyCsrO6;
      myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
      timeMyCsrO6 = getTimeC() - timeMyCsrO6;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrO6: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrI2*/
    printf("My CsrI2\n");
    zero(y2,nl,"double");
    myMatVecCsr = &matVecCsrI2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrI2 = getTimeC() - timeMyCsrI2;
      myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
      timeMyCsrI2 = getTimeC() - timeMyCsrI2;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrI2: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrI4*/
    printf("My CsrI4\n");
    zero(y2,nl,"double");
    myMatVecCsr = &matVecCsrI4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrI4 = getTimeC() - timeMyCsrI4;
      myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
      timeMyCsrI4 = getTimeC() - timeMyCsrI4;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrI4: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrI6*/
    printf("My CsrI6\n");
    zero(y2,nl,"double");
    myMatVecCsr = &matVecCsrI6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrI6 = getTimeC() - timeMyCsrI6;
      myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
      timeMyCsrI6 = getTimeC() - timeMyCsrI6;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrI6: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrO2I2*/
    printf("My CsrO2I2\n");
    zero(y2,nl,"double");
    myMatVecCsr = &matVecCsrO2I2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrO2I2 = getTimeC() - timeMyCsrO2I2;
      myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
      timeMyCsrO2I2 = getTimeC() - timeMyCsrO2I2;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrO2I2: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrD */
    printf("My CsrD\n");
    zero(y2,nl,"double");
    myMatVecCsrD = &matVecCsrD;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrD   = getTimeC() - timeMyCsrD;
      myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
      timeMyCsrD   = getTimeC() - timeMyCsrD;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrD: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrDI2 */
    printf("My CsrDI2\n");
    zero(y2,nl,"double");
    myMatVecCsrD = &matVecCsrDI2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrDI2   = getTimeC() - timeMyCsrDI2;
      myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
      timeMyCsrDI2   = getTimeC() - timeMyCsrDI2;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrDI2: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrDI4 */
    printf("My CsrDI4\n");
    zero(y2,nl,"double");
    myMatVecCsrD = &matVecCsrDI4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrDI4   = getTimeC() - timeMyCsrDI4;
      myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
      timeMyCsrDI4   = getTimeC() - timeMyCsrDI4;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrDI4: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrDI6 */
    printf("My CsrDI6\n");
    zero(y2,nl,"double");
    myMatVecCsrD = &matVecCsrDI6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrDI6   = getTimeC() - timeMyCsrDI6;
      myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
      timeMyCsrDI6   = getTimeC() - timeMyCsrDI6;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrDI6: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrDO2 */
    printf("My CsrDO2\n");
    zero(y2,nl,"double");
    myMatVecCsrD = &matVecCsrDO2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrDO2   = getTimeC() - timeMyCsrDO2;
      myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
      timeMyCsrDO2   = getTimeC() - timeMyCsrDO2;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrDO2: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrDO4 */
    printf("My CsrDO4\n");
    zero(y2,nl,"double");
    myMatVecCsrD = &matVecCsrDO4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrDO4   = getTimeC() - timeMyCsrDO4;
      myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
      timeMyCsrDO4   = getTimeC() - timeMyCsrDO4;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrDO4: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrDO6 */
    printf("My CsrDO6\n");
    zero(y2,nl,"double");
    myMatVecCsrD = &matVecCsrDO6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrDO6   = getTimeC() - timeMyCsrDO6;
      myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
      timeMyCsrDO6   = getTimeC() - timeMyCsrDO6;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrDO6: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrDO2I2 */
    printf("My CsrDO2I2\n");
    zero(y2,nl,"double");
    myMatVecCsrD = &matVecCsrDO2I2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrDO2I2   = getTimeC() - timeMyCsrDO2I2;
      myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
      timeMyCsrDO2I2   = getTimeC() - timeMyCsrDO2I2;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrDO2I2: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrC */
    printf("My CsrC\n");
    zero(y2,nl,"double");
    myMatVecCsrC = &matVecCsrC;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrC   = getTimeC() - timeMyCsrC;
      myMatVecCsrC(nl,iaCsrC,jaCsrC,auCsrC,adCsrC,alCsrC,x1,y2);
      timeMyCsrC   = getTimeC() - timeMyCsrC;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrC: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrCI2 */
    printf("My CsrCI2\n");
    zero(y2,nl,"double");
    myMatVecCsrC = &matVecCsrCI2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrCI2   = getTimeC() - timeMyCsrCI2;
      myMatVecCsrC(nl,iaCsrC,jaCsrC,auCsrC,adCsrC,alCsrC,x1,y2);
      timeMyCsrCI2   = getTimeC() - timeMyCsrCI2;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrCI2: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrCI4 */
    printf("My CsrCI4\n");
    zero(y2,nl,"double");
    myMatVecCsrC = &matVecCsrCI2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrCI4   = getTimeC() - timeMyCsrCI4;
      myMatVecCsrC(nl,iaCsrC,jaCsrC,auCsrC,adCsrC,alCsrC,x1,y2);
      timeMyCsrCI4   = getTimeC() - timeMyCsrCI4;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrCI4: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrCI6 */
    printf("My CsrCI6\n");
    zero(y2,nl,"double");
    myMatVecCsrC = &matVecCsrCI6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrCI6 = getTimeC() - timeMyCsrCI6;
      myMatVecCsrC(nl,iaCsrC,jaCsrC,auCsrC,adCsrC,alCsrC,x1,y2);
      timeMyCsrCI6 = getTimeC() - timeMyCsrCI6;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrCI6: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrCO2 */
    printf("My CsrCO2\n");
    zero(y2,nl,"double");
    myMatVecCsrC = &matVecCsrCO2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrCO2 = getTimeC() - timeMyCsrCO2;
      myMatVecCsrC(nl,iaCsrC,jaCsrC,auCsrC,adCsrC,alCsrC,x1,y2);
      timeMyCsrCO2 = getTimeC() - timeMyCsrCO2;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrCO2: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrCO4 */
    printf("My CsrCO4\n");
    zero(y2,nl,"double");
    myMatVecCsrC = &matVecCsrCO4;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrCO4 = getTimeC() - timeMyCsrCO4;
      myMatVecCsrC(nl,iaCsrC,jaCsrC,auCsrC,adCsrC,alCsrC,x1,y2);
      timeMyCsrCO4 = getTimeC() - timeMyCsrCO4;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrCO4: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrCO6 */
    printf("My CsrCO6\n");
    zero(y2,nl,"double");
    myMatVecCsrC = &matVecCsrCO6;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrCO6 = getTimeC() - timeMyCsrCO6;
      myMatVecCsrC(nl,iaCsrC,jaCsrC,auCsrC,adCsrC,alCsrC,x1,y2);
      timeMyCsrCO6 = getTimeC() - timeMyCsrCO6;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrCO6: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

/* ... MyCsrCO2I2 */
    printf("My CsrCO2I2\n");
    zero(y2,nl,"double");
    myMatVecCsrC = &matVecCsrCO2I2;
    for(i = 0;i < NSAMPLES;i++){
      timeMyCsrCO2I2 = getTimeC() - timeMyCsrCO2I2;
      myMatVecCsrC(nl,iaCsrC,jaCsrC,auCsrC,adCsrC,alCsrC,x1,y2);
      timeMyCsrCO2I2 = getTimeC() - timeMyCsrCO2I2;
      if(xDiffY(y2,y1,MATVECZERO,nl)){
        printf("MateVecCsrCO2I2: Vetores diferentes\n");
        fileOut = fopen(nameOut1,"w");
/*...*/
        fprintf(fileOut,"%6s %24s %24s %24s\n"
               ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
        for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
                  ,i+1,y1[i],y2[i],x1[i]);
        fclose(fileOut);
/*...................................................................*/
        exit(EXIT_FAILURE);
      }
    }
/*...................................................................*/

#ifdef _OPENMP
/* ... MyCsrOmp */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrOmp[nThreads-1] = 0.0;  
      printf("My CsrOmp %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsr = &matVecCsrOmp;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrOmp[nThreads-1]=getTimeC()-timeMyCsrOmp[nThreads-1];
#pragma omp parallel
        myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
        timeMyCsrOmp[nThreads-1]=getTimeC()-timeMyCsrOmp[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrOmp: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrOmpI2 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrOmpI2[nThreads-1] = 0.0;  
      printf("My CsrOmpI2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsr = &matVecCsrOmpI2;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrOmpI2[nThreads-1]
        = getTimeC()-timeMyCsrOmpI2[nThreads-1];
#pragma omp parallel
        myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
        timeMyCsrOmpI2[nThreads-1]
        =getTimeC()-timeMyCsrOmpI2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrOmpI2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrOmpI4 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrOmpI4[nThreads-1] = 0.0;  
      printf("My CsrOmpI4 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsr = &matVecCsrOmpI4;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrOmpI4[nThreads-1]
        = getTimeC()-timeMyCsrOmpI4[nThreads-1];
#pragma omp parallel
        myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
        timeMyCsrOmpI4[nThreads-1]
        =getTimeC()-timeMyCsrOmpI4[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrOmpI4: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrOmpI6 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrOmpI6[nThreads-1] = 0.0;  
      printf("My CsrOmpI6 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsr = &matVecCsrOmpI6;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrOmpI6[nThreads-1]
        = getTimeC()-timeMyCsrOmpI6[nThreads-1];
#pragma omp parallel
        myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
        timeMyCsrOmpI6[nThreads-1]
        =getTimeC()-timeMyCsrOmpI6[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrOmpI6: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrOmpO2 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrOmpO2[nThreads-1] = 0.0;  
      printf("My CsrOmpO2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsr = &matVecCsrOmpO2;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrOmpO2[nThreads-1]
        = getTimeC()-timeMyCsrOmpO2[nThreads-1];
#pragma omp parallel shared(nl,iaCsr,jaCsr,aCsr,x1,y2)
        myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
        timeMyCsrOmpO2[nThreads-1]
        =getTimeC()-timeMyCsrOmpO2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrOmpO2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrOmpO4 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrOmpO4[nThreads-1] = 0.0;  
      printf("My CsrOmpO4 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsr = &matVecCsrOmpO4;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrOmpO4[nThreads-1]
        = getTimeC()-timeMyCsrOmpO4[nThreads-1];
#pragma omp parallel shared(nl,iaCsr,jaCsr,aCsr,x1,y2)
        myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
        timeMyCsrOmpO4[nThreads-1]
        =getTimeC()-timeMyCsrOmpO4[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrOmpO4: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrOmpO6 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrOmpO6[nThreads-1] = 0.0;  
      printf("My CsrOmpO6 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsr = &matVecCsrOmpO6;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrOmpO6[nThreads-1]
        = getTimeC()-timeMyCsrOmpO6[nThreads-1];
#pragma omp parallel shared(nl,iaCsr,jaCsr,aCsr,x1,y2)
        myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
        timeMyCsrOmpO6[nThreads-1]
        =getTimeC()-timeMyCsrOmpO6[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrOmpO6: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrOmpO2I2 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrOmpO2I2[nThreads-1] = 0.0;  
      printf("My CsrOmpO2I2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsr = &matVecCsrOmpO2I2;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrOmpO2I2[nThreads-1]
        = getTimeC()-timeMyCsrOmpO2I2[nThreads-1];
#pragma omp parallel shared(nl,iaCsr,jaCsr,aCsr,x1,y2)
        myMatVecCsr(nl,iaCsr,jaCsr,aCsr,x1,y2);
        timeMyCsrOmpO2I2[nThreads-1]
        =getTimeC()-timeMyCsrOmpO2I2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrOmpO2I2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmp */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmp[nThreads-1] = 0.0;  
      printf("My CsrDOmp %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrD = &matVecCsrDOmp;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmp[nThreads-1]
        = getTimeC()-timeMyCsrDOmp[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2)
        myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
        timeMyCsrDOmp[nThreads-1]
        =getTimeC()-timeMyCsrDOmp[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmp: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpI2 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpI2[nThreads-1] = 0.0;  
      printf("My CsrDOmpI2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrD = &matVecCsrDOmpI2;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpI2[nThreads-1]
        = getTimeC()-timeMyCsrDOmpI2[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2)
        myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
        timeMyCsrDOmpI2[nThreads-1]
        =getTimeC()-timeMyCsrDOmpI2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpI2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpI4 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpI4[nThreads-1] = 0.0;  
      printf("My CsrDOmpI4 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrD = &matVecCsrDOmpI4;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpI4[nThreads-1]
        = getTimeC()-timeMyCsrDOmpI4[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2)
        myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
        timeMyCsrDOmpI4[nThreads-1]
        =getTimeC()-timeMyCsrDOmpI4[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpI4: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpI6 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpI6[nThreads-1] = 0.0;  
      printf("My CsrDOmpI6 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrD = &matVecCsrDOmpI6;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpI6[nThreads-1]
        = getTimeC()-timeMyCsrDOmpI6[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2)
        myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
        timeMyCsrDOmpI6[nThreads-1]
        =getTimeC()-timeMyCsrDOmpI6[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpI6: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpO2 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpO2[nThreads-1] = 0.0;  
      printf("My CsrDOmpO2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrD = &matVecCsrDOmpO2;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpO2[nThreads-1]
        = getTimeC()-timeMyCsrDOmpO2[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2)
        myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
        timeMyCsrDOmpO2[nThreads-1]
        =getTimeC()-timeMyCsrDOmpO2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpO2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpO4 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpO4[nThreads-1] = 0.0;  
      printf("My CsrDOmpO4 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrD = &matVecCsrDOmpO4;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpO4[nThreads-1]
        = getTimeC()-timeMyCsrDOmpO4[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2)
        myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
        timeMyCsrDOmpO4[nThreads-1]
        =getTimeC()-timeMyCsrDOmpO4[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpO4: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpO6 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpO6[nThreads-1] = 0.0;  
      printf("My CsrDOmpO6 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrD = &matVecCsrDOmpO6;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpO6[nThreads-1]
        = getTimeC()-timeMyCsrDOmpO6[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2)
        myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
        timeMyCsrDOmpO6[nThreads-1]
        =getTimeC()-timeMyCsrDOmpO6[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpO6: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpO2I2 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpO2I2[nThreads-1] = 0.0;  
      printf("My CsrDOmpO2I2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrD = &matVecCsrDOmpO2I2;
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpO2I2[nThreads-1]
        = getTimeC()-timeMyCsrDOmpO2I2[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2)
        myMatVecCsrD(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2);
        timeMyCsrDOmpO2I2[nThreads-1]
        =getTimeC()-timeMyCsrDOmpO2I2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpO2I2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpBal */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpBal[nThreads-1] = 0.0;  
      timeOverHeadBal[nThreads-1]  = 0.0;  
      printf("My CsrDOmpBal %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrDOmpBal = &matVecCsrDOmpBal;
/*...*/
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
      partitionCsrByNonzeros(iaCsrD       ,jaCsrD      ,nl
                            ,nThreads     ,nThBeginCsrD
                            ,nThEndCsrD   ,nThSizeCsrD
                            ,nThHeightCsrD, 2);
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpBal[nThreads-1]
        = getTimeC()-timeMyCsrDOmpBal[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2\
                           ,nThBeginCsrD,nThEndCsrD)
        myMatVecCsrDOmpBal(nl
                          ,iaCsrD      ,jaCsrD
                          ,aCsrD       ,adCsrD
                          ,x1          ,y2
                          ,nThBeginCsrD,nThEndCsrD);
        timeMyCsrDOmpBal[nThreads-1]
        =getTimeC()-timeMyCsrDOmpBal[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpBal: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpBalI2 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpBalI2[nThreads-1] = 0.0;  
      timeOverHeadBal[nThreads-1]  = 0.0;  
      printf("My CsrDOmpBalI2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrDOmpBal = &matVecCsrDOmpBalI2;
/*...*/
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
      partitionCsrByNonzeros(iaCsrD       ,jaCsrD      ,nl
                            ,nThreads     ,nThBeginCsrD
                            ,nThEndCsrD   ,nThSizeCsrD
                            ,nThHeightCsrD, 2);
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpBalI2[nThreads-1]
        = getTimeC()-timeMyCsrDOmpBalI2[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2\
                           ,nThBeginCsrD,nThEndCsrD)
        myMatVecCsrDOmpBal(nl
                          ,iaCsrD      ,jaCsrD
                          ,aCsrD       ,adCsrD
                          ,x1          ,y2
                          ,nThBeginCsrD,nThEndCsrD);
        timeMyCsrDOmpBalI2[nThreads-1]
        =getTimeC()-timeMyCsrDOmpBalI2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpBalI2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpBalI4 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpBalI4[nThreads-1] = 0.0;  
      timeOverHeadBal[nThreads-1]  = 0.0;  
      printf("My CsrDOmpBalI4 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrDOmpBal = &matVecCsrDOmpBalI4;
/*...*/
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
      partitionCsrByNonzeros(iaCsrD       ,jaCsrD      ,nl
                            ,nThreads     ,nThBeginCsrD
                            ,nThEndCsrD   ,nThSizeCsrD
                            ,nThHeightCsrD, 2);
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpBalI4[nThreads-1]
        = getTimeC()-timeMyCsrDOmpBalI4[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2\
                           ,nThBeginCsrD,nThEndCsrD)
        myMatVecCsrDOmpBal(nl
                          ,iaCsrD      ,jaCsrD
                          ,aCsrD       ,adCsrD
                          ,x1          ,y2
                          ,nThBeginCsrD,nThEndCsrD);
        timeMyCsrDOmpBalI4[nThreads-1]
        =getTimeC()-timeMyCsrDOmpBalI4[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpBalI4: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpBalI6 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpBalI6[nThreads-1] = 0.0;  
      timeOverHeadBal[nThreads-1]  = 0.0;  
      printf("My CsrDOmpBalI6 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrDOmpBal = &matVecCsrDOmpBalI6;
/*...*/
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
      partitionCsrByNonzeros(iaCsrD       ,jaCsrD      ,nl
                            ,nThreads     ,nThBeginCsrD
                            ,nThEndCsrD   ,nThSizeCsrD
                            ,nThHeightCsrD, 2);
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpBalI6[nThreads-1]
        = getTimeC()-timeMyCsrDOmpBalI6[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2\
                           ,nThBeginCsrD,nThEndCsrD)
        myMatVecCsrDOmpBal(nl
                          ,iaCsrD      ,jaCsrD
                          ,aCsrD       ,adCsrD
                          ,x1          ,y2
                          ,nThBeginCsrD,nThEndCsrD);
        timeMyCsrDOmpBalI6[nThreads-1]
        =getTimeC()-timeMyCsrDOmpBalI6[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpBalI6: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpBalO2 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpBalO2[nThreads-1] = 0.0;  
      timeOverHeadBal[nThreads-1]    = 0.0;  
      printf("My CsrDOmpBalO2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrDOmpBal = &matVecCsrDOmpBalO2;
/*...*/
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
      partitionCsrByNonzeros(iaCsrD       ,jaCsrD      ,nl
                            ,nThreads     ,nThBeginCsrD
                            ,nThEndCsrD   ,nThSizeCsrD
                            ,nThHeightCsrD, 2);
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpBalO2[nThreads-1]
        = getTimeC()-timeMyCsrDOmpBalO2[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2\
                           ,nThBeginCsrD,nThEndCsrD)
        myMatVecCsrDOmpBal(nl
                          ,iaCsrD      ,jaCsrD
                          ,aCsrD       ,adCsrD
                          ,x1          ,y2
                          ,nThBeginCsrD,nThEndCsrD);
        timeMyCsrDOmpBalO2[nThreads-1]
        =getTimeC()-timeMyCsrDOmpBalO2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpBalO2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpBalO4 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpBalO4[nThreads-1] = 0.0;  
      timeOverHeadBal[nThreads-1]    = 0.0;  
      printf("My CsrDOmpBalO4 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrDOmpBal = &matVecCsrDOmpBalO4;
/*...*/
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
      partitionCsrByNonzeros(iaCsrD       ,jaCsrD      ,nl
                            ,nThreads     ,nThBeginCsrD
                            ,nThEndCsrD   ,nThSizeCsrD
                            ,nThHeightCsrD, 2);
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpBalO4[nThreads-1]
        = getTimeC()-timeMyCsrDOmpBalO4[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2\
                           ,nThBeginCsrD,nThEndCsrD)
        myMatVecCsrDOmpBal(nl
                          ,iaCsrD      ,jaCsrD
                          ,aCsrD       ,adCsrD
                          ,x1          ,y2
                          ,nThBeginCsrD,nThEndCsrD);
        timeMyCsrDOmpBalO4[nThreads-1]
        =getTimeC()-timeMyCsrDOmpBalO4[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpBalO4: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpBalO6 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpBalO6[nThreads-1] = 0.0;  
      timeOverHeadBal[nThreads-1]    = 0.0;  
      printf("My CsrDOmpBalO6 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrDOmpBal = &matVecCsrDOmpBalO6;
/*...*/
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
      partitionCsrByNonzeros(iaCsrD       ,jaCsrD      ,nl
                            ,nThreads     ,nThBeginCsrD
                            ,nThEndCsrD   ,nThSizeCsrD
                            ,nThHeightCsrD, 2);
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpBalO6[nThreads-1]
        = getTimeC()-timeMyCsrDOmpBalO6[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2\
                           ,nThBeginCsrD,nThEndCsrD)
        myMatVecCsrDOmpBal(nl
                          ,iaCsrD      ,jaCsrD
                          ,aCsrD       ,adCsrD
                          ,x1          ,y2
                          ,nThBeginCsrD,nThEndCsrD);
        timeMyCsrDOmpBalO6[nThreads-1]
        =getTimeC()-timeMyCsrDOmpBalO6[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpBalO6: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrDOmpBalO2I2 */
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrDOmpBalO2I2[nThreads-1] = 0.0;  
      timeOverHeadBal[nThreads-1]    = 0.0;  
      printf("My CsrDOmpBalO2I2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrDOmpBal = &matVecCsrDOmpBalO2I2;
/*...*/
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
      partitionCsrByNonzeros(iaCsrD       ,jaCsrD      ,nl
                            ,nThreads     ,nThBeginCsrD
                            ,nThEndCsrD   ,nThSizeCsrD
                            ,nThHeightCsrD, 2);
      timeOverHeadBal[nThreads-1] = getTimeC() 
                                  - timeOverHeadBal[nThreads-1];  
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrDOmpBalO2I2[nThreads-1]
        = getTimeC()-timeMyCsrDOmpBalO2I2[nThreads-1];
#pragma omp parallel shared(nl,iaCsrD,jaCsrD,aCsrD,adCsrD,x1,y2\
                           ,nThBeginCsrD,nThEndCsrD)
        myMatVecCsrDOmpBal(nl
                          ,iaCsrD      ,jaCsrD
                          ,aCsrD       ,adCsrD
                          ,x1          ,y2
                          ,nThBeginCsrD,nThEndCsrD);
        timeMyCsrDOmpBalO2I2[nThreads-1]
        =getTimeC()-timeMyCsrDOmpBalO2I2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrDOmpBalO2I2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrCOmp */
    zero(nThBeginCsrC  ,MAX_NUM_THREADS,INTC);
    zero(nThEndCsrC    ,MAX_NUM_THREADS,INTC);
    zero(nThSizeCsrC   ,MAX_NUM_THREADS,INTC);
    zero(nThHeightCsrC ,MAX_NUM_THREADS,INTC);
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrCOmp[nThreads-1]    = 0.0;  
      timeOverHeadCsrC[nThreads-1] = 0.0; 
      printf("My CsrCOmp %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrCOmp = &matVecCsrCOmp;
/*...*/
      hccaAlloc(double,&m,bufferThY ,(nl*nThreads),"bThy" ,false);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];  
      partitionCsrByNonzeros(iaCsrC       ,jaCsrC      ,nl
                            ,nThreads     ,nThBeginCsrC
                            ,nThEndCsrC   ,nThSizeCsrC
                            ,nThHeightCsrC, 3);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1]; 
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrCOmp[nThreads-1]=getTimeC()-timeMyCsrCOmp[nThreads-1];
#pragma omp parallel
        myMatVecCsrCOmp(nl
                       ,iaCsrC        ,jaCsrC
                       ,auCsrC        ,adCsrC
                       ,alCsrC
                       ,x1            ,y2
                       ,nThBeginCsrC  ,nThEndCsrC
                       ,nThHeightCsrC
                       ,bufferThY     ,nThreads);       
        timeMyCsrCOmp[nThreads-1]=getTimeC()-timeMyCsrCOmp[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrCOmp: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
      hccaDealloc(&m,bufferThY,"bThy",false);
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrCOmpI2 */
    zero(nThBeginCsrC  ,MAX_NUM_THREADS,INTC);
    zero(nThEndCsrC    ,MAX_NUM_THREADS,INTC);
    zero(nThSizeCsrC   ,MAX_NUM_THREADS,INTC);
    zero(nThHeightCsrC ,MAX_NUM_THREADS,INTC);
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrCOmpI2[nThreads-1]  = 0.0;  
      timeOverHeadCsrC[nThreads-1] = 0.0; 
      printf("My CsrCOmpI2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrCOmp = &matVecCsrCOmpI2;
/*...*/
      hccaAlloc(double,&m,bufferThY ,(nl*nThreads),"bThy" ,false);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];  
      partitionCsrByNonzeros(iaCsrC       ,jaCsrC      ,nl
                            ,nThreads     ,nThBeginCsrC
                            ,nThEndCsrC   ,nThSizeCsrC
                            ,nThHeightCsrC, 3);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1]; 
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrCOmpI2[nThreads-1]=getTimeC()-timeMyCsrCOmpI2[nThreads-1];
#pragma omp parallel
        myMatVecCsrCOmp(nl
                       ,iaCsrC        ,jaCsrC
                       ,auCsrC        ,adCsrC
                       ,alCsrC
                       ,x1            ,y2
                       ,nThBeginCsrC  ,nThEndCsrC
                       ,nThHeightCsrC
                       ,bufferThY     ,nThreads);       
        timeMyCsrCOmpI2[nThreads-1]=getTimeC()-timeMyCsrCOmpI2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrCOmpI2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
      hccaDealloc(&m,bufferThY,"bThy",false);
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrCOmpI4 */
    zero(nThBeginCsrC  ,MAX_NUM_THREADS,INTC);
    zero(nThEndCsrC    ,MAX_NUM_THREADS,INTC);
    zero(nThSizeCsrC   ,MAX_NUM_THREADS,INTC);
    zero(nThHeightCsrC ,MAX_NUM_THREADS,INTC);
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrCOmpI4[nThreads-1]  = 0.0;  
      timeOverHeadCsrC[nThreads-1] = 0.0; 
      printf("My CsrCOmpI4 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrCOmp = &matVecCsrCOmpI4;
/*...*/
      hccaAlloc(double,&m,bufferThY ,(nl*nThreads),"bThy" ,false);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];  
      partitionCsrByNonzeros(iaCsrC       ,jaCsrC      ,nl
                            ,nThreads     ,nThBeginCsrC
                            ,nThEndCsrC   ,nThSizeCsrC
                            ,nThHeightCsrC, 3);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1]; 
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrCOmpI4[nThreads-1]=getTimeC()-timeMyCsrCOmpI4[nThreads-1];
#pragma omp parallel
        myMatVecCsrCOmp(nl
                       ,iaCsrC        ,jaCsrC
                       ,auCsrC        ,adCsrC
                       ,alCsrC
                       ,x1            ,y2
                       ,nThBeginCsrC  ,nThEndCsrC
                       ,nThHeightCsrC
                       ,bufferThY     ,nThreads);       
        timeMyCsrCOmpI4[nThreads-1]=getTimeC()-timeMyCsrCOmpI4[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrCOmpI4: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
      hccaDealloc(&m,bufferThY,"bThy",false);
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrCOmpI6 */
    zero(nThBeginCsrC  ,MAX_NUM_THREADS,INTC);
    zero(nThEndCsrC    ,MAX_NUM_THREADS,INTC);
    zero(nThSizeCsrC   ,MAX_NUM_THREADS,INTC);
    zero(nThHeightCsrC ,MAX_NUM_THREADS,INTC);
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrCOmpI6[nThreads-1]  = 0.0;  
      timeOverHeadCsrC[nThreads-1] = 0.0; 
      printf("My CsrCOmpI6 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrCOmp = &matVecCsrCOmpI6;
/*...*/
      hccaAlloc(double,&m,bufferThY ,(nl*nThreads),"bThy" ,false);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];  
      partitionCsrByNonzeros(iaCsrC       ,jaCsrC      ,nl
                            ,nThreads     ,nThBeginCsrC
                            ,nThEndCsrC   ,nThSizeCsrC
                            ,nThHeightCsrC, 3);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1]; 
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrCOmpI6[nThreads-1]=getTimeC()-timeMyCsrCOmpI6[nThreads-1];
#pragma omp parallel
        myMatVecCsrCOmp(nl
                       ,iaCsrC        ,jaCsrC
                       ,auCsrC        ,adCsrC
                       ,alCsrC
                       ,x1            ,y2
                       ,nThBeginCsrC  ,nThEndCsrC
                       ,nThHeightCsrC
                       ,bufferThY     ,nThreads);       
        timeMyCsrCOmpI6[nThreads-1]=getTimeC()-timeMyCsrCOmpI6[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrCOmpI6: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
      hccaDealloc(&m,bufferThY,"bThy",false);
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrCOmpO2 */
    zero(nThBeginCsrC  ,MAX_NUM_THREADS,INTC);
    zero(nThEndCsrC    ,MAX_NUM_THREADS,INTC);
    zero(nThSizeCsrC   ,MAX_NUM_THREADS,INTC);
    zero(nThHeightCsrC ,MAX_NUM_THREADS,INTC);
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrCOmpO2[nThreads-1]    = 0.0;  
      timeOverHeadCsrC[nThreads-1]   = 0.0; 
      printf("My CsrCOmpO2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrCOmp = &matVecCsrCOmpO2;
/*...*/
      hccaAlloc(double,&m,bufferThY ,(nl*nThreads),"bThy" ,false);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];  
      partitionCsrByNonzeros(iaCsrC       ,jaCsrC      ,nl
                            ,nThreads     ,nThBeginCsrC
                            ,nThEndCsrC   ,nThSizeCsrC
                            ,nThHeightCsrC, 3);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1]; 
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrCOmpO2[nThreads-1]=getTimeC()-timeMyCsrCOmpO2[nThreads-1];
#pragma omp parallel
        myMatVecCsrCOmp(nl
                       ,iaCsrC        ,jaCsrC
                       ,auCsrC        ,adCsrC
                       ,alCsrC
                       ,x1            ,y2
                       ,nThBeginCsrC  ,nThEndCsrC
                       ,nThHeightCsrC
                       ,bufferThY     ,nThreads);       
        timeMyCsrCOmpO2[nThreads-1]=getTimeC()-timeMyCsrCOmpO2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrCOmpO2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
      hccaDealloc(&m,bufferThY,"bThy",false);
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrCOmpO4 */
    zero(nThBeginCsrC  ,MAX_NUM_THREADS,INTC);
    zero(nThEndCsrC    ,MAX_NUM_THREADS,INTC);
    zero(nThSizeCsrC   ,MAX_NUM_THREADS,INTC);
    zero(nThHeightCsrC ,MAX_NUM_THREADS,INTC);
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrCOmpO4[nThreads-1]    = 0.0;  
      timeOverHeadCsrC[nThreads-1]   = 0.0; 
      printf("My CsrCOmpO4 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrCOmp = &matVecCsrCOmpO4;
/*...*/
      hccaAlloc(double,&m,bufferThY ,(nl*nThreads),"bThy" ,false);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];  
      partitionCsrByNonzeros(iaCsrC       ,jaCsrC      ,nl
                            ,nThreads     ,nThBeginCsrC
                            ,nThEndCsrC   ,nThSizeCsrC
                            ,nThHeightCsrC, 3);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrCOmpO4[nThreads-1]=getTimeC()-timeMyCsrCOmpO4[nThreads-1];
#pragma omp parallel
        myMatVecCsrCOmp(nl
                       ,iaCsrC        ,jaCsrC
                       ,auCsrC        ,adCsrC
                       ,alCsrC
                       ,x1            ,y2
                       ,nThBeginCsrC  ,nThEndCsrC
                       ,nThHeightCsrC
                       ,bufferThY     ,nThreads);       
        timeMyCsrCOmpO4[nThreads-1]=getTimeC()-timeMyCsrCOmpO4[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrCOmpO4: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
      hccaDealloc(&m,bufferThY,"bThy",false);
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrCOmpO6 */
    zero(nThBeginCsrC  ,MAX_NUM_THREADS,INTC);
    zero(nThEndCsrC    ,MAX_NUM_THREADS,INTC);
    zero(nThSizeCsrC   ,MAX_NUM_THREADS,INTC);
    zero(nThHeightCsrC ,MAX_NUM_THREADS,INTC);
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrCOmpO6[nThreads-1]    = 0.0;  
      timeOverHeadCsrC[nThreads-1]   = 0.0; 
      printf("My CsrCOmpO6 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrCOmp = &matVecCsrCOmpO6;
/*...*/
      hccaAlloc(double,&m,bufferThY ,(nl*nThreads),"bThy" ,false);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];  
      partitionCsrByNonzeros(iaCsrC       ,jaCsrC      ,nl
                            ,nThreads     ,nThBeginCsrC
                            ,nThEndCsrC   ,nThSizeCsrC
                            ,nThHeightCsrC, 3);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrCOmpO6[nThreads-1]=getTimeC()-timeMyCsrCOmpO6[nThreads-1];
#pragma omp parallel
        myMatVecCsrCOmp(nl
                       ,iaCsrC        ,jaCsrC
                       ,auCsrC        ,adCsrC
                       ,alCsrC
                       ,x1            ,y2
                       ,nThBeginCsrC  ,nThEndCsrC
                       ,nThHeightCsrC
                       ,bufferThY     ,nThreads);       
        timeMyCsrCOmpO6[nThreads-1]=getTimeC()-timeMyCsrCOmpO6[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrCOmpO6: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
      hccaDealloc(&m,bufferThY,"bThy",false);
/*...................................................................*/
    }
/*...................................................................*/

/* ... MyCsrCOmpO2I2 */
    zero(nThBeginCsrC  ,MAX_NUM_THREADS,INTC);
    zero(nThEndCsrC    ,MAX_NUM_THREADS,INTC);
    zero(nThSizeCsrC   ,MAX_NUM_THREADS,INTC);
    zero(nThHeightCsrC ,MAX_NUM_THREADS,INTC);
    for(nThreads=2;nThreads<=numTotalThreads;nThreads++){
      omp_set_num_threads(nThreads);
      timeMyCsrCOmpO2I2[nThreads-1]    = 0.0;  
      timeOverHeadCsrC[nThreads-1]   = 0.0; 
      printf("My CsrCOmpO2I2 %d\n",omp_get_max_threads());
      zero(y2,nl,"double");
      myMatVecCsrCOmp = &matVecCsrCOmpO2I2;
/*...*/
      hccaAlloc(double,&m,bufferThY ,(nl*nThreads),"bThy" ,false);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];  
      partitionCsrByNonzeros(iaCsrC       ,jaCsrC      ,nl
                            ,nThreads     ,nThBeginCsrC
                            ,nThEndCsrC   ,nThSizeCsrC
                            ,nThHeightCsrC, 3);
      timeOverHeadCsrC[nThreads-1] = getTimeC() 
                                   - timeOverHeadCsrC[nThreads-1];
/*...................................................................*/
      for(i = 0;i < NSAMPLES;i++){
        timeMyCsrCOmpO2I2[nThreads-1]=getTimeC()
        -timeMyCsrCOmpO2I2[nThreads-1];
#pragma omp parallel
        myMatVecCsrCOmp(nl
                       ,iaCsrC        ,jaCsrC
                       ,auCsrC        ,adCsrC
                       ,alCsrC
                       ,x1            ,y2
                       ,nThBeginCsrC  ,nThEndCsrC
                       ,nThHeightCsrC
                       ,bufferThY     ,nThreads);       
        timeMyCsrCOmpO2I2[nThreads-1]=getTimeC()
        -timeMyCsrCOmpO2I2[nThreads-1];
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("MateVecCsrCOmpO2I2: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
      hccaDealloc(&m,bufferThY,"bThy",false);
/*...................................................................*/
    }
/*...................................................................*/

/*melhor tempo Csr*/
    timeMyCsrBest = timeMyCsr;
    strcpy(nameCsr,"hccaCsr");
    if( timeMyCsrBest > timeMyCsrI2){
      timeMyCsrBest = timeMyCsrI2;
      strcpy(nameCsr,"hccaCsrI2");
    }
    if( timeMyCsrBest > timeMyCsrI4){
      timeMyCsrBest = timeMyCsrI4;
      strcpy(nameCsr,"hccaCsrI4");
    }
    if( timeMyCsrBest > timeMyCsrI6){
      timeMyCsrBest = timeMyCsrI6;
      strcpy(nameCsr,"hccaCsrI6");
    }
    if( timeMyCsrBest > timeMyCsrO2){
      timeMyCsrBest = timeMyCsrO2;
      strcpy(nameCsr,"hccaCsrO2");
    }
    if( timeMyCsrBest > timeMyCsrO4){
      timeMyCsrBest = timeMyCsrO4;
      strcpy(nameCsr,"hccaCsrO4");
    }
    if( timeMyCsrBest > timeMyCsrO6){
      timeMyCsrBest = timeMyCsrO6;
      strcpy(nameCsr,"hccaCsrO6");
    }
    if( timeMyCsrBest > timeMyCsrO2I2){
      timeMyCsrBest = timeMyCsrO2I2;
      strcpy(nameCsr,"hccaCsrO2I2");
    }
#if _OPENMP
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      i = nThreads - 1;
      timeMyCsrOmpBest[i] = timeMyCsrOmp[i];
      strcpy(nameCsrOmp[i],"hccaCsrOmp");
      if( timeMyCsrOmpBest[i] > timeMyCsrOmpI2[i]){
        timeMyCsrOmpBest[i] = timeMyCsrOmpI2[i];
        strcpy(nameCsrOmp[i],"hccaCsrOmpI2");
      }
      if( timeMyCsrOmpBest[i] > timeMyCsrOmpI4[i]){
        timeMyCsrOmpBest[i] = timeMyCsrOmpI4[i];
        strcpy(nameCsrOmp[i],"hccaCsrOmpI4");
      }
      if( timeMyCsrOmpBest[i] > timeMyCsrOmpI6[i]){
        timeMyCsrOmpBest[i] = timeMyCsrOmpI6[i];
        strcpy(nameCsrOmp[i],"hccaCsrOmpI6");
      }
      if( timeMyCsrOmpBest[i] > timeMyCsrOmpO2[i]){
        timeMyCsrOmpBest[i] = timeMyCsrOmpO2[i];
        strcpy(nameCsrOmp[i],"hccaCsrOmpO2");
      }
      if( timeMyCsrOmpBest[i] > timeMyCsrOmpO2[i]){
        timeMyCsrOmpBest[i] = timeMyCsrOmpO2[i];
        strcpy(nameCsrOmp[i],"hccaCsrOmpO2");
      }
      if( timeMyCsrOmpBest[i] > timeMyCsrOmpO4[i]){
        timeMyCsrOmpBest[i] = timeMyCsrOmpO4[i];
        strcpy(nameCsrOmp[i],"hccaCsrOmpO4");
      }
      if( timeMyCsrOmpBest[i] > timeMyCsrOmpO6[i]){
        timeMyCsrOmpBest[i] = timeMyCsrOmpO6[i];
        strcpy(nameCsrOmp[i],"hccaCsrOmpO6");
      }
      if( timeMyCsrOmpBest[i] > timeMyCsrOmpO2I2[i]){
        timeMyCsrOmpBest[i] = timeMyCsrOmpO2I2[i];
        strcpy(nameCsrOmp[i],"hccaCsrOmpO2I2");
      }
    }
#endif
/*...................................................................*/

/*melhor tempo CsrD*/
    timeMyCsrDBest = timeMyCsrD;
    strcpy(nameCsrD,"hccaCsrD");
    if( timeMyCsrDBest > timeMyCsrDI2){
      timeMyCsrDBest = timeMyCsrDI2;
      strcpy(nameCsrD,"hccaCsrDI2");
    }
    if( timeMyCsrDBest > timeMyCsrDI4){
      timeMyCsrDBest = timeMyCsrDI4;
      strcpy(nameCsrD,"hccaCsrDI4");
    }
    if( timeMyCsrDBest > timeMyCsrDI6){
      timeMyCsrDBest = timeMyCsrDI6;
      strcpy(nameCsrD,"hccaCsrDI6");
    }
    if( timeMyCsrDBest > timeMyCsrDO2){
      timeMyCsrDBest = timeMyCsrDO2;
      strcpy(nameCsrD,"hccaCsrDO2");
    }
    if( timeMyCsrDBest > timeMyCsrDO4){
      timeMyCsrDBest = timeMyCsrDO4;
      strcpy(nameCsrD,"hccaCsrDO4");
    }
    if( timeMyCsrDBest > timeMyCsrDO6){
      timeMyCsrDBest = timeMyCsrDO6;
      strcpy(nameCsrD,"hccaCsrDO6");
    }
    if( timeMyCsrDBest > timeMyCsrDO2I2){
      timeMyCsrDBest = timeMyCsrDO2I2;
      strcpy(nameCsrD,"hccaCsrDO2I2");
    }

#if _OPENMP
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      i = nThreads - 1;
      timeMyCsrDBalOmpBest[i] = timeMyCsrDOmpBal[i];
      strcpy(nameCsrDBalOmp[i],"hccaCsrDOmpBal");
      if( timeMyCsrDBalOmpBest[i] > timeMyCsrDOmpBalI2[i]){
        timeMyCsrDBalOmpBest[i] = timeMyCsrDOmpBalI2[i];
        strcpy(nameCsrDBalOmp[i],"hccaCsrDOmpBalI2");
      }
      if( timeMyCsrDBalOmpBest[i] > timeMyCsrDOmpBalI4[i]){
        timeMyCsrDBalOmpBest[i] = timeMyCsrDOmpBalI4[i];
        strcpy(nameCsrDBalOmp[i],"hccaCsrDOmpBalI4");
      }
      if( timeMyCsrDBalOmpBest[i] > timeMyCsrDOmpBalI6[i]){
        timeMyCsrDBalOmpBest[i] = timeMyCsrDOmpBalI6[i];
        strcpy(nameCsrDBalOmp[i],"hccaCsrDBalOmpBalI6");
      }
      if( timeMyCsrDBalOmpBest[i] > timeMyCsrDOmpBalO2[i]){
        timeMyCsrDBalOmpBest[i] = timeMyCsrDOmpBalO2[i];
        strcpy(nameCsrDBalOmp[i],"hccaCsrDBalOmpBalO2");
      }
      if( timeMyCsrDBalOmpBest[i] > timeMyCsrDOmpBalO4[i]){
        timeMyCsrDBalOmpBest[i] = timeMyCsrDOmpBalO4[i];
        strcpy(nameCsrDBalOmp[i],"hccaCsrDBalOmpBalO4");
      }
      if( timeMyCsrDBalOmpBest[i] > timeMyCsrDOmpBalO6[i]){
        timeMyCsrDBalOmpBest[i] = timeMyCsrDOmpBalO6[i];
        strcpy(nameCsrDBalOmp[i],"hccaCsrDBalOmpBalO6");
      }
      if( timeMyCsrDBalOmpBest[i] > timeMyCsrDOmpBalO2I2[i]){
        timeMyCsrDBalOmpBest[i] = timeMyCsrDOmpBalO2I2[i];
        strcpy(nameCsrDBalOmp[i],"hccaCsrDBalOmpBalO2I2");
      }
    }
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      i = nThreads - 1;
      timeMyCsrDOmpBest[i] = timeMyCsrDOmp[i];
      strcpy(nameCsrDOmp[i],"hccaCsrDOmp");
      if( timeMyCsrDOmpBest[i] > timeMyCsrDOmpI2[i]){
        timeMyCsrDOmpBest[i] = timeMyCsrDOmpI2[i];
        strcpy(nameCsrDOmp[i],"hccaCsrDOmpI2");
      }
      if( timeMyCsrDOmpBest[i] > timeMyCsrDOmpI4[i]){
        timeMyCsrDOmpBest[i] = timeMyCsrDOmpI4[i];
        strcpy(nameCsrDOmp[i],"hccaCsrDOmpI4");
      }
      if( timeMyCsrDOmpBest[i] > timeMyCsrDOmpI6[i]){
        timeMyCsrDOmpBest[i] = timeMyCsrDOmpI6[i];
        strcpy(nameCsrDOmp[i],"hccaCsrDOmpI6");
      }
      if( timeMyCsrDOmpBest[i] > timeMyCsrDOmpO2[i]){
        timeMyCsrDOmpBest[i] = timeMyCsrDOmpO2[i];
        strcpy(nameCsrDOmp[i],"hccaCsrDOmpO2");
      }
      if( timeMyCsrDOmpBest[i] > timeMyCsrDOmpO4[i]){
        timeMyCsrDOmpBest[i] = timeMyCsrDOmpO4[i];
        strcpy(nameCsrDOmp[i],"hccaCsrDOmpO4");
      }
      if( timeMyCsrDOmpBest[i] > timeMyCsrDOmpO6[i]){
        timeMyCsrDOmpBest[i] = timeMyCsrDOmpO6[i];
        strcpy(nameCsrDOmp[i],"hccaCsrDOmpO6");
      }
      if( timeMyCsrDOmpBest[i] > timeMyCsrDOmpO2I2[i]){
        timeMyCsrDOmpBest[i] = timeMyCsrDOmpO2I2[i];
        strcpy(nameCsrDOmp[i],"hccaCsrDOmpO2I2");
      }
    }
#endif
/*...................................................................*/

/*melhor tempo CsrC*/
    timeMyCsrCBest = timeMyCsrC;
    strcpy(nameCsrC,"hccaCsrC");
    if( timeMyCsrCBest > timeMyCsrCI2){
      timeMyCsrCBest = timeMyCsrCI2;
      strcpy(nameCsrC,"hccaCsrCI2");
    }
    if( timeMyCsrCBest > timeMyCsrCI4){
      timeMyCsrCBest = timeMyCsrCI4;
      strcpy(nameCsrC,"hccaCsrCI4");
    }
    if( timeMyCsrCBest > timeMyCsrCI6){
      timeMyCsrCBest = timeMyCsrCI6;
      strcpy(nameCsrC,"hccaCsrCI6");
    }
    if( timeMyCsrCBest > timeMyCsrCO2){
      timeMyCsrCBest = timeMyCsrCO2;
      strcpy(nameCsrC,"hccaCsrCO2");
    }
    if( timeMyCsrCBest > timeMyCsrCO4){
      timeMyCsrCBest = timeMyCsrCO4;
      strcpy(nameCsrC,"hccaCsrCO4");
    }
    if( timeMyCsrCBest > timeMyCsrCO6){
      timeMyCsrCBest = timeMyCsrCO6;
      strcpy(nameCsrC,"hccaCsrCO6");
    }
    if( timeMyCsrCBest > timeMyCsrCO2I2){
      timeMyCsrCBest = timeMyCsrCO2I2;
      strcpy(nameCsrC,"hccaCsrCO2I2");
    }

#if _OPENMP
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      i = nThreads - 1;
      timeMyCsrCOmpBest[i] = timeMyCsrCOmp[i];
      strcpy(nameCsrCOmp[i],"hccaCsrCOmp");
      if( timeMyCsrCOmpBest[i] > timeMyCsrCOmpI2[i]){
        timeMyCsrCOmpBest[i] = timeMyCsrCOmpI2[i];
        strcpy(nameCsrCOmp[i],"hccaCsrCOmpI2");
      }
      if( timeMyCsrCOmpBest[i] > timeMyCsrCOmpI4[i]){
        timeMyCsrCOmpBest[i] = timeMyCsrCOmpI4[i];
        strcpy(nameCsrCOmp[i],"hccaCsrCOmpI4");
      }
      if( timeMyCsrCOmpBest[i] > timeMyCsrCOmpI6[i]){
        timeMyCsrCOmpBest[i] = timeMyCsrCOmpI6[i];
        strcpy(nameCsrCOmp[i],"hccaCsrCOmpI6");
      }
      if( timeMyCsrCOmpBest[i] > timeMyCsrCOmpO2[i]){
        timeMyCsrCOmpBest[i] = timeMyCsrCOmpO2[i];
        strcpy(nameCsrCOmp[i],"hccaCsrCOmpO2");
      }
      if( timeMyCsrCOmpBest[i] > timeMyCsrCOmpO2[i]){
        timeMyCsrCOmpBest[i] = timeMyCsrCOmpO2[i];
        strcpy(nameCsrCOmp[i],"hccaCsrCOmpO2");
      }
      if( timeMyCsrCOmpBest[i] > timeMyCsrCOmpO4[i]){
        timeMyCsrCOmpBest[i] = timeMyCsrCOmpO4[i];
        strcpy(nameCsrCOmp[i],"hccaCsrCOmpO4");
      }
      if( timeMyCsrCOmpBest[i] > timeMyCsrCOmpO6[i]){
        timeMyCsrCOmpBest[i] = timeMyCsrCOmpO6[i];
        strcpy(nameCsrCOmp[i],"hccaCsrCOmpO6");
      }
      if( timeMyCsrCOmpBest[i] > timeMyCsrCOmpO2I2[i]){
        timeMyCsrCOmpBest[i] = timeMyCsrCOmpO2I2[i];
        strcpy(nameCsrCOmp[i],"hccaCsrCOmpO2I2");
      }
    }
#endif
/*...................................................................*/

/*... Produto da operacao y=Ax*/  
    fileOut = fopen(nameOut0,"w");
    fprintf(fileOut,"%21s %16.8lf %16.8lf\n","mklCsr"      
           ,timeMklCsr,timeMklCsr/timeMklCsr);
#ifdef _CUDABLAS_
    fprintf(fileOut,"%21s %16.8lf %16.8lf\n","cuda"     
           ,timeCudaCsr,timeCudaCsr/timeMklCsr);
#endif
    fprintf(fileOut,"%21s %16.8lf %16.8lf\n",nameCsr      
           ,timeMyCsrBest,timeMyCsrDBest/timeMklCsr);
    fprintf(fileOut,"%21s %16.8lf %16.8lf\n",nameCsrD
           ,timeMyCsrDBest,timeMyCsrDBest/timeMklCsr);
    fprintf(fileOut,"%21s %16.8lf %16.8lf\n",nameCsrC
           ,timeMyCsrCBest,timeMyCsrDBest/timeMklCsr);
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      i = nThreads - 1;
      fprintf(fileOut,"%21s %16.8lf %16.8lf %2d\n"
                     ,"mklCsrOmp"   ,timeMklCsrOmp[i]    
                     ,timeMklCsrOmp[i]/timeMklCsrOmp[i],i+1);
      fprintf(fileOut,"%21s %16.8lf %16.8lf %2d\n"
                     ,nameCsrOmp[i] ,timeMyCsrOmpBest[i] 
                     ,timeMyCsrOmpBest[i]/timeMklCsrOmp[i],i+1);
      fprintf(fileOut,"%21s %16.8lf %16.8lf %2d\n"
                     ,nameCsrDOmp[i],timeMyCsrDOmpBest[i]
                     ,timeMyCsrDOmpBest[i]/timeMklCsrOmp[i],i+1);
      fprintf(fileOut,"%21s %16.8lf %16.8lf %2d\n"
                     ,nameCsrDBalOmp[i] ,timeMyCsrDBalOmpBest[i]
                     ,timeMyCsrDBalOmpBest[i]/timeMklCsrOmp[i],i+1);
      fprintf(fileOut,"%21s %16.8lf %16.8lf %2d\n"
                     ,nameCsrCOmp[i],timeMyCsrCOmpBest[i]
                     ,timeMyCsrCOmp[i]/timeMklCsrOmp[i],i+1);
    }
    fclose(fileOut);
/*...................................................................*/
    
#endif

/*... Produto da operacao y=Ax*/  
    fileOut = fopen(nameOut1,"w");
    fprintf(fileOut,"neq=%d nad=%d Band Max=%ld Band Med=%ld " 
                    "nl Max=%ld nl Med=%ld\n"
                    ,nl,nadCsr,bandMax,bandMed,nlMax,nlMed);
    fprintf(fileOut,"%6s %24s %24s %24s\n"
                   ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
    for(i=0;i<nl;i++)
      fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
      ,i+1,y1[i],y2[i],x1[i]);
    fclose(fileOut);
/*...................................................................*/
    
    fileOut = fopen(nameOut2,"w");


/*... MklCsr*/
#ifdef _MKL_
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
           "%10s=%16.8lf\n"
          ,"MklCsr"
          ,"time"   
          ,timeMklCsr
          ,"gflop"
          ,gFlopCsr     
          ,"gflops"
          ,(NSAMPLES*gFlopCsr)/timeMklCsr   
          ,"ntime"
          ,timeMklCsr/timeMklCsr);
#endif
/*...................................................................*/

/*... cudaCsr*/
#ifdef _CUDABLAS_
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
           "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
          ,"cudaCsr"
          ,"time"   
          ,timeCudaCsr
          ,"gflop"
          ,gFlopCsr     
          ,"gflops"
          ,(NSAMPLES*gFlopCsr)/timeCudaCsr   
          ,"ntime"
          ,timeCudaCsr/timeCudaCsr
          ,"overHead"
          ,timeOverHeadCuda
          ,"Time Event"
          ,timeCudaCsrEvent/1000.0);
#endif
/*...................................................................*/

/*... Csr*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsr"    
            ,"time"   
            ,timeMyCsr
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsr    
            ,"ntime"
            ,timeMklCsr/timeMyCsr
            ,"speedups"
            ,timeMyCsr/timeMyCsr);
/*...................................................................*/

/*... CsrO2*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrO2"    
            ,"time"   
            ,timeMyCsrO2
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrO2    
            ,"ntime"
            ,timeMklCsr/timeMyCsrO2
            ,"speedups"
            ,timeMyCsr/timeMyCsrO2);
/*...................................................................*/

/*... CsrO4*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrO4"    
            ,"time"   
            ,timeMyCsrO4
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrO4    
            ,"ntime"
            ,timeMklCsr/timeMyCsrO4
            ,"speedups"
            ,timeMyCsr/timeMyCsrO4);
/*...................................................................*/

/*... CsrO6*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrO6"    
            ,"time"   
            ,timeMyCsrO6
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrO6    
            ,"ntime"
            ,timeMklCsr/timeMyCsrO6  
            ,"speedups"
            ,timeMyCsr/timeMyCsrO6);
/*...................................................................*/

/*... CsrI2*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrI2"    
            ,"time"   
            ,timeMyCsrI2
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrI2    
            ,"ntime"
            ,timeMklCsr/timeMyCsrI2  
            ,"speedups"
            ,timeMyCsr/timeMyCsrI2);
/*...................................................................*/

/*... CsrI4*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrI4"    
            ,"time"   
            ,timeMyCsrI4
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrI4    
            ,"ntime"
            ,timeMklCsr/timeMyCsrI4  
            ,"speedups"
            ,timeMyCsr/timeMyCsrI4);
/*...................................................................*/

/*... CsrI6*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrI6"    
            ,"time"   
            ,timeMyCsrI6
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrI6    
            ,"ntime"
            ,timeMklCsr/timeMyCsrI6  
            ,"speedups"
            ,timeMyCsr/timeMyCsrI6);
/*...................................................................*/

/*... CsrO2I2*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrO2I2"    
            ,"time"   
            ,timeMyCsrO2I2
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrO2I2    
            ,"ntime"
            ,timeMklCsr/timeMyCsrO2I2
            ,"speedups"
            ,timeMyCsr/timeMyCsrO2I2);
/*...................................................................*/

/*... CsrD*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrD"    
            ,"time"   
            ,timeMyCsrD
            ,"gflop"
            ,gFlopCsrD     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrD)/timeMyCsrD    
            ,"ntime"
            ,timeMklCsr/timeMyCsrD
            ,"speedups"
            ,timeMyCsrD/timeMyCsrD);
/*...................................................................*/

/*... CsrDO2*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDO2"    
            ,"time"   
            ,timeMyCsrDO2
            ,"gflop"
            ,gFlopCsrD     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrD)/timeMyCsrDO2    
            ,"ntime"
            ,timeMklCsr/timeMyCsrDO2
            ,"speedups"
            ,timeMyCsrD/timeMyCsrDO2);
/*...................................................................*/

/*... CsrDO4*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDO4"    
            ,"time"   
            ,timeMyCsrDO4
            ,"gflop"
            ,gFlopCsrD     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrD)/timeMyCsrDO4    
            ,"ntime"
            ,timeMklCsr/timeMyCsrDO4
            ,"speedups"
            ,timeMyCsrD/timeMyCsrDO4);
/*...................................................................*/

/*... CsrDO6*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDO6"    
            ,"time"   
            ,timeMyCsrDO6
            ,"gflop"
            ,gFlopCsrD     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrD)/timeMyCsrDO6    
            ,"ntime"
            ,timeMklCsr/timeMyCsrDO6
            ,"speedups"
            ,timeMyCsrD/timeMyCsrDO6);
/*...................................................................*/

/*... CsrDI2*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDI2"    
            ,"time"   
            ,timeMyCsrDI2
            ,"gflop"
            ,gFlopCsrD     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrD)/timeMyCsrDI2    
            ,"ntime"
            ,timeMklCsr/timeMyCsrDI2
            ,"speedups"
            ,timeMyCsrD/timeMyCsrDI2);
/*...................................................................*/

/*... CsrDI4*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDI4"    
            ,"time"   
            ,timeMyCsrDI4
            ,"gflop"
            ,gFlopCsrD     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrD)/timeMyCsrDI4    
            ,"ntime"
            ,timeMklCsr/timeMyCsrDI4  
            ,"speedups"
            ,timeMyCsrD/timeMyCsrDI4);
/*...................................................................*/

/*... CsrDI6*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDI6"    
            ,"time"   
            ,timeMyCsrDI6
            ,"gflop"
            ,gFlopCsrD     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrD)/timeMyCsrDI6    
            ,"ntime"
            ,timeMklCsr/timeMyCsrDI6
            ,"speedups"
            ,timeMyCsrD/timeMyCsrDI6);
/*...................................................................*/

/*... CsrDO2I2*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDO2I2"    
            ,"time"   
            ,timeMyCsrDO2I2
            ,"gflop"
            ,gFlopCsrD     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrD)/timeMyCsrDO2I2    
            ,"ntime"
            ,timeMklCsr/timeMyCsrDO2I2
            ,"speedups"
            ,timeMyCsrD/timeMyCsrDO2I2);
/*...................................................................*/

/*... CsrC*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrC"    
            ,"time"   
            ,timeMyCsrC    
            ,"gflop"
            ,gFlopCsrC     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrC)/timeMyCsrC        
            ,"ntime"
            ,timeMklCsr/timeMyCsrC
            ,"speedups"
            ,timeMyCsrC/timeMyCsrC);
/*...................................................................*/

/*... CsrCI2*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCI2"    
            ,"time"   
            ,timeMyCsrCI2    
            ,"gflop"
            ,gFlopCsrC     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrC)/timeMyCsrCI2        
            ,"ntime"
            ,timeMklCsr/timeMyCsrCI2
            ,"speedups"
            ,timeMyCsrC/timeMyCsrCI2);
/*...................................................................*/

/*... CsrCI4*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCI4"    
            ,"time"   
            ,timeMyCsrCI4    
            ,"gflop"
            ,gFlopCsrC     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrC)/timeMyCsrCI4        
            ,"ntime"
            ,timeMklCsr/timeMyCsrCI4
            ,"speedups"
            ,timeMyCsrC/timeMyCsrCI4);
/*...................................................................*/

/*... CsrCI6*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCI6"    
            ,"time"   
            ,timeMyCsrCI6    
            ,"gflop"
            ,gFlopCsrC     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrC)/timeMyCsrCI6        
            ,"ntime"
            ,timeMklCsr/timeMyCsrCI6
            ,"speedups"
            ,timeMyCsrC/timeMyCsrCI6);
/*...................................................................*/

/*... CsrCO2*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCO2"    
            ,"time"   
            ,timeMyCsrCO2    
            ,"gflop"
            ,gFlopCsrC     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrC)/timeMyCsrCO2        
            ,"ntime"
            ,timeMklCsr/timeMyCsrCO2
            ,"speedups"
            ,timeMyCsrC/timeMyCsrCO2);
/*...................................................................*/

/*... CsrCO4*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCO4"    
            ,"time"   
            ,timeMyCsrCO4    
            ,"gflop"
            ,gFlopCsrC     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrC)/timeMyCsrCO4        
            ,"ntime"
            ,timeMklCsr/timeMyCsrCO4
            ,"speedups"
            ,timeMyCsrC/timeMyCsrCO4);
/*...................................................................*/

/*... CsrCO6*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCO6"    
            ,"time"   
            ,timeMyCsrCO6    
            ,"gflop"
            ,gFlopCsrC     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrC)/timeMyCsrCO6        
            ,"ntime"
            ,timeMklCsr/timeMyCsrCO6
            ,"speedups"
            ,timeMyCsrC/timeMyCsrCO6);
/*...................................................................*/

/*... CsrCO2I2*/
    fprintf(fileOut,"%20s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
                    "%10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCO2I2"    
            ,"time"   
            ,timeMyCsrCO2I2    
            ,"gflop"
            ,gFlopCsrC     
            ,"gflops"
            ,(NSAMPLES*gFlopCsrC)/timeMyCsrCO2I2        
            ,"ntime"
            ,timeMklCsr/timeMyCsrCO2I2
            ,"speedups"
            ,timeMyCsrC/timeMyCsrCO2I2);
/*...................................................................*/


#ifdef _OPENMP
  #if _MKL_
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MklCsrOmp-"   
            ,nThreads
            ,"time"
            ,timeMklCsrOmp[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMklCsrOmp[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMklCsrOmp[nThreads-1]
            ,"speedups"
            ,timeMklCsr/timeMklCsrOmp[nThreads-1]);
    }
  #endif

/*... CsrOmp*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrOmp-"   
            ,nThreads
            ,"time"
            ,timeMyCsrOmp[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrOmp[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrOmp[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrOmp[nThreads-1]);
    }
/*...................................................................*/

/*... CsrOmpI2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrOmpI2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrOmpI2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrOmpI2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrOmpI2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrOmpI2[nThreads-1]);
    }
/*...................................................................*/

/*... CsrOmpI4*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrOmpI4-"   
            ,nThreads
            ,"time"
            ,timeMyCsrOmpI4[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrOmpI4[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrOmpI4[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrOmpI4[nThreads-1]);
    }
/*...................................................................*/

/*... CsrOmpI6*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrOmpI6-"   
            ,nThreads
            ,"time"
            ,timeMyCsrOmpI6[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrOmpI6[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrOmpI6[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrOmpI6[nThreads-1]);
    }
/*...................................................................*/

/*... CsrOmpO2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrOmpO2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrOmpO2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrOmpO2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrOmpO2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrOmpO2[nThreads-1]);
    }
/*...................................................................*/

/*... CsrOmpO4*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrOmpO4-"   
            ,nThreads
            ,"time"
            ,timeMyCsrOmpO4[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrOmpO4[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrOmpO4[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrOmpO4[nThreads-1]);
    }
/*...................................................................*/

/*... CsrOmpO6*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrOmpO6-"   
            ,nThreads
            ,"time"
            ,timeMyCsrOmpO6[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrOmpO6[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrOmpO6[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrOmpO6[nThreads-1]);
    }
/*...................................................................*/

/*... CsrOmpO2I2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrOmpO2I2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrOmpO2I2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrOmpO2I2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrOmpO2I2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrOmpO2I2[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmp*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmp-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmp[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmp[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmp[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmp[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpI2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpI2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpI2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpI2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpI2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpI2[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpI4*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpI4-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpI4[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpI4[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpI4[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpI4[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpI6*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpI6-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpI6[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpI6[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpI6[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpI6[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpO2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpO2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpO2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpO2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpO2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpO2[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpO4*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpO4-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpO4[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpO4[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpO4[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpO4[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpO6*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpO6-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpO6[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpO6[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpO6[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpO6[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpO2I2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpO2I2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpO2I2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpO2I2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpO2I2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpO2I2[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpBal*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpBal-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpBal[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpBal[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpBal[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpBal[nThreads-1]
            ,"overHead"
            ,timeOverHeadBal[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpBalI2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpBalI2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpBalI2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpBalI2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpBalI2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpBalI2[nThreads-1]
            ,"overHead"
            ,timeOverHeadBal[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpBalI4*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpBalI4-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpBalI4[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpBalI4[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpBalI4[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpBalI4[nThreads-1]
            ,"overHead"
            ,timeOverHeadBal[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpBalI6*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpBalI6-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpBalI6[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpBalI6[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpBalI6[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpBalI6[nThreads-1]
            ,"overHead"
            ,timeOverHeadBal[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpBalO2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpBalO2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpBalO2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpBalO2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpBalO2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpBalO2[nThreads-1]
            ,"overHead"
            ,timeOverHeadBal[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpBalO4*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpBalO4-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpBalO4[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpBalO4[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpBalO4[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpBalO4[nThreads-1]
            ,"overHead"
            ,timeOverHeadBal[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpBalO6*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpBalO6-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpBalO6[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpBalO6[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpBalO6[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpBalO6[nThreads-1]
            ,"overHead"
            ,timeOverHeadBal[nThreads-1]);
    }
/*...................................................................*/

/*... CsrDOmpBalO2I2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrDOmpBalO2I2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrDOmpBalO2I2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrDOmpBalO2I2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrDOmpBalO2I2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrDOmpBalO2I2[nThreads-1]
            ,"overHead"
            ,timeOverHeadBal[nThreads-1]);
    }
/*...................................................................*/

/*... CsrCOmp*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCOmp-"   
            ,nThreads
            ,"time"
            ,timeMyCsrCOmp[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrCOmp[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrCOmp[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrCOmp[nThreads-1]
            ,"overHead"
            ,timeOverHeadCsrC[nThreads-1]);
    }
/*...................................................................*/

/*... CsrCOmpI2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCOmpI2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrCOmpI2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrCOmpI2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrCOmpI2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrCOmpI2[nThreads-1]
            ,"overHead"
            ,timeOverHeadCsrC[nThreads-1]);
    }
/*...................................................................*/

/*... CsrCOmpI4*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCOmpI4-"   
            ,nThreads
            ,"time"
            ,timeMyCsrCOmpI4[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrCOmpI4[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrCOmpI4[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrCOmpI4[nThreads-1]
            ,"overHead"
            ,timeOverHeadCsrC[nThreads-1]);
    }
/*...................................................................*/

/*... CsrCOmpI6*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCOmpI6-"   
            ,nThreads
            ,"time"
            ,timeMyCsrCOmpI6[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrCOmpI6[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrCOmpI6[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrCOmpI6[nThreads-1]
            ,"overHead"
            ,timeOverHeadCsrC[nThreads-1]);
    }
/*...................................................................*/

/*... CsrCOmpO2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCOmpO2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrCOmpO2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrCOmpO2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrCOmpO2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrCOmpO2[nThreads-1]
            ,"overHead"
            ,timeOverHeadCsrC[nThreads-1]);
    }
/*...................................................................*/

/*... CsrCOmpO4*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCOmpO4-"   
            ,nThreads
            ,"time"
            ,timeMyCsrCOmpO4[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrCOmpO4[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrCOmpO4[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrCOmpO4[nThreads-1]
            ,"overHead"
            ,timeOverHeadCsrC[nThreads-1]);
    }
/*...................................................................*/

/*... CsrCOmpO6*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCOmpO6-"   
            ,nThreads
            ,"time"
            ,timeMyCsrCOmpO6[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrCOmpO6[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrCOmpO6[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrCOmpO6[nThreads-1]
            ,"overHead"
            ,timeOverHeadCsrC[nThreads-1]);
    }
/*...................................................................*/

/*... CsrCOmpO2I2*/
    for(nThreads=2;nThreads<=numTotalThreads;nThreads+=2){
      fprintf(fileOut,"%18s%2d %10s=%16.8lf %10s=%16.8lf "
                      "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
            ,"MyCsrCOmpO2I2-"   
            ,nThreads
            ,"time"
            ,timeMyCsrCOmpO2I2[nThreads-1]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
            ,(NSAMPLES*gFlopCsr)/timeMyCsrCOmpO2I2[nThreads-1]   
            ,"ntime"
            ,timeMklCsrOmp[nThreads-1]/timeMyCsrCOmpO2I2[nThreads-1]
            ,"speedups"
            ,timeMyCsr/timeMyCsrCOmpO2I2[nThreads-1]
            ,"overHead"
            ,timeOverHeadCsrC[nThreads-1]);
    }
/*...................................................................*/
#endif
/*...................................................................*/

/*...*/
    hccaDealloc(&m,iCol  ,"iCol",false);
    hccaDealloc(&m,iLin  ,"iLin",false);
    hccaDealloc(&m,val   ,"val" ,false);
    hccaDealloc(&m,y2    ,"y2"  ,false);
    hccaDealloc(&m,y1    ,"y1"  ,false);
    hccaDealloc(&m,x1    ,"x1"  ,false);
    hccaDealloc(&m,aCsr  ,"aCsr",false);
    hccaDealloc(&m,iaCsr ,"iaCsr",false);
    hccaDealloc(&m,jaCsr ,"jaCsr",false);
    hccaDealloc(&m,iaCsrC,"iaCsrC",false);
    hccaDealloc(&m,jaCsrC,"jaCsrC",false);
    hccaDealloc(&m,auCsrC,"auCsrC",false);
    hccaDealloc(&m,adCsrC,"adCsrC",false);
    hccaDealloc(&m,alCsrC,"alCsrC",false);
/*...................................................................*/
    
/*...*/
    fclose(fileOut);
    fclose(fileIn);
/*...................................................................*/
  }
/*...................................................................*/
  
/*... matriz esparsa CSR GPU*/
  else if(!strcmp(argv[1],nameArgs1[3])){
/*...*/
    strcpy(prename,argv[3]);
    strcpy(nameOut0,prename);
    strcpy(nameOut1,prename);
    strcpy(nameOut2,prename);
    strcat(nameOut0,"_bestTime.txt");
    strcat(nameOut1,"_matvec.txt");
    strcat(nameOut2,"_time.txt");
/*.....................................................................*/

/*...*/
    fileIn = openFile(argv[2],"r");
/*.....................................................................*/

/*... leitura de arquivo mtx*/  
    fprintf(stderr,"Lendo o arquivo.\n");
#ifdef _MMIO_
    mm_read_banner(fileIn,&matCode);
    mm_read_mtx_crd_size(fileIn,&nl,&ncl,&nnz);
/*.....................................................................*/

/*... alocacao COO*/  
    hccaAlloc(double,&m,val,nnz,"val",false);
    hccaAlloc(int,&m,iLin,nnz,"iLin" ,false);
    hccaAlloc(int,&m,iCol,nnz,"iCol" ,false);
    zero(val ,nnz,"double");
    zero(iLin,nnz,"int");
    zero(iCol,nnz,"int");
  
    mm_read_mtx_crd_data(fileIn,nl,ncl,nnz,iLin,iCol,val,matCode);
    fprintf(stderr,"Arquivo lido.\n");
#endif
/*...................................................................*/

/*...*/
    nadCsr    = nnz; 
    nadCsrD = nnz - nl; 
    nadCsrC   = (nnz - nl)/2; 
/*.....................................................................*/

/*... alocacao ACsr, iaCSr, jaCsr*/ 
    hccaAlloc(double,&m,aCsr , nadCsr   ,"aCsr"  ,false);
    hccaAlloc(int   ,&m,iaCsr,(nl+1),"iaCsr" ,false);
    hccaAlloc(int   ,&m,jaCsr ,nadCsr,"jaCsr" ,false);
    zero(aCsr    ,nadCsr,"double");
    zero(iaCsr   ,(nl+1),"int");
    zero(jaCsr   ,nadCsr,"int");

/*... alocacao ACsrD, iaCsrD, jaCsrD*/ 
    hccaAlloc(double,&m,aCsrD ,nadCsrD   ,"aCsrD"  ,false);
    hccaAlloc(double,&m,adCsrD,nl        ,"adCsrD" ,false);
    hccaAlloc(int   ,&m,iaCsrD,(nl+1)    ,"iaCsrD" ,false);
    hccaAlloc(int   ,&m,jaCsrD,nadCsrD   ,"jaCsrD" ,false);
    zero(aCsrD ,nadCsrD,"double");
    zero(adCsrD,       nl,"double");
    zero(iaCsrD,   (nl+1),"int"   );
    zero(jaCsrD,nadCsrD,"int"   );

/*... alocacao ACsrC, iaCsrC, jaCsrC*/ 
    hccaAlloc(double,&m,auCsrC   ,nadCsrC    ,"auCsrC"   ,false);
    hccaAlloc(double,&m,alCsrC   ,nadCsrC    ,"alCsrC"   ,false);
    hccaAlloc(double,&m,adCsrC  ,nl          ,"adCsrC"   ,false);
    hccaAlloc(int   ,&m,iaCsrC  ,(nl+1)      ,"iaCsrC"   ,false);
    hccaAlloc(int   ,&m,jaCsrC  ,nadCsrC     ,"jaCsrC"   ,false);
    zero(auCsrC,nadCsrC,"double");
    zero(alCsrC,nadCsrC,"double");
    zero(adCsrC,     nl,"double");
    zero(iaCsrC, (nl+1),"int"   );
    zero(jaCsrC,nadCsrC,"int"   );
/*...................................................................*/

/*... x, y1 e y2*/
    hccaAlloc(double,&m,x1   ,nl    ,"x1" ,false);
    hccaAlloc(double,&m,y1   ,nl    ,"y1" ,false);
    hccaAlloc(double,&m,y2   ,nl    ,"y2" ,false);
//  mapVector(&m);
/*.....................................................................*/
    
    fprintf(stderr,"Coo -> Csr\n");
    hccaAlloc(int,&m,aux  ,nl    ,"aux",false);
/*... CSR*/
    tyCsr = csr;
    cooToCsr(iLin  ,iCol  ,val
            ,iaCsr ,jaCsr
            ,aCsr  ,aCsr  ,aCsr 
            ,nl    ,nnz   ,tyCsr
            ,aux   ,true  ,true  ,true);
/*... CSRD*/
    tyCsr = csrD;  
    cooToCsr(iLin     ,iCol  ,val
            ,iaCsrD ,jaCsrD
            ,aCsrD  ,adCsrD  ,aCsrD 
            ,nl       ,nnz       ,tyCsr
            ,aux   ,true  ,false  ,true);
/*... CSRC  */
    tyCsr = csrC;   
    cooToCsr(iLin     ,iCol  ,val
            ,iaCsrC   ,jaCsrC  
            ,auCsrC   ,adCsrC    ,alCsrC   
            ,nl       ,nnz       ,tyCsr
            ,aux      ,false ,false  ,true);
/*.....................................................................*/
    hccaDealloc(&m,aux,"aux",false);
/*.....................................................................*/

/*... gFlop*/
    tyCsr = csr;
    gFlopCsr    = flopMatVecCsr(nl,nadCsr ,tyCsr)/1000000000.0;
    tyCsr = csrD;  
    gFlopCsrD   = flopMatVecCsr(nl,nadCsrD,tyCsr)/1000000000.0;
    tyCsr = csrC;
    gFlopCsrC   = flopMatVecCsr(nl,nadCsrC,tyCsr)/1000000000.0;
/*.....................................................................*/

/*... banda da mariz*/
     bandMax   = bandCsr(iaCsrD,jaCsrD,nl,1);
     bandMed   = bandCsr(iaCsrD,jaCsrD,nl,2);
     bandMin   = bandCsr(iaCsrD,jaCsrD,nl,3);
     printf("banda Max   = %ld\n",bandMax);
     printf("banda Media = %ld\n",bandMed);
     printf("banda Min   = %ld\n",bandMin);
/*.....................................................................*/

/*... numero de elementos nao nulos por linha da mariz*/
     nlMax = nlCsr(iaCsrD,nl,1);
     nlMed = nlCsr(iaCsrD,nl,2);
     nlMin = nlCsr(iaCsrD,nl,3);
     printf("nl Max   = %ld\n",nlMax);
     printf("nl Med   = %ld\n",nlMed);
     printf("nl Min   = %ld\n",nlMin);
/*.....................................................................*/

/*... geracao randomica do vetor denso x*/
    if(!strcmp(argv[4],"true"))
      flagRandom = 1;
    if(flagRandom)
      randomMatrix(x1,nl,0);
    else
      for(i = 0;i < nl;i++){
        if(i%2)
          x1[i] =  fmod(i+1,10.0) + 0.2;
        else
          x1[i] = -fmod(i+1,5)    - 0.1;
      }
/*.....................................................................*/

#ifdef _CUDABLAS_
/*... cudaCsr*/
    printf("cudaCsr\n");
    timeCudaCsr      = 0.0;
    timeOverHeadCuda = 0.0;
    timeCudaCsrEvent = 0.0;

/*...*/
    initCuSparse(nl  ,iaCsr            ,jaCsr 
               ,aCsr ,&timeOverHeadCuda); 
/*...................................................................*/
    for(i = 0;i < NSAMPLES;i++){
/*...*/ 
      timeCudaCsr = getTimeC() - timeCudaCsr;
/*...................................................................*/
     
/*...*/
      csrCuSparse(nl               ,nadCsr
                 ,x1               ,y1   
                 ,&timeOverHeadCuda,&timeCudaCsrEvent); 
/*...................................................................*/

/*...*/ 
      timeCudaCsr = getTimeC() - timeCudaCsr;
/*...................................................................*/
    }
/*...................................................................*/

/*...gpuCsrScalar*/
    printf("CsrGpuScalar\n");
    initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                  ,&devA,&devX,&devY);
/*...................................................................*/

     zero(y2,nl,"double");
     nThreads = 32;
//     if(nl%2) nBlock = nl/nThreads-1;
//     else  nBlock= nl/nThreads;
     nBlock = 1024;
     printf("CsrGpuScalar Block %d nTh %d %d\n",nBlock,nThreads,nBlock*nThreads);
     timeMyCudaCsrScalar = 0.0e0;
     for(i = 0;i < NSAMPLES;i++){
/*...*/ 
      timeMyCudaCsrScalar=getTimeC()-timeMyCudaCsrScalar;
/*...................................................................*/

/*...*/ 
       matVecCsrGpuCudaScalar(nl,x1,y2,&devIa,&devJa,&devA
                             ,&devX,&devY,nBlock,nThreads,1,1); 
/*...................................................................*/

/*...*/ 
       timeMyCudaCsrScalar=getTimeC()-timeMyCudaCsrScalar;
/*...................................................................*/

/*...*/ 
       if(xDiffY(y2,y1,MATVECZERO,nl)){
         printf("CsrGpuScalar: Vetores diferentes\n");
         fileOut = fopen(nameOut1,"w");
/*...*/
         fprintf(fileOut,"%6s %24s %24s %24s\n"
                ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
         for(i=0;i<nl;i++)
           fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
           ,i+1,y1[i],y2[i],x1[i]);
         fclose(fileOut);
/*...................................................................*/
         exit(EXIT_FAILURE);
       }
/*...................................................................*/
    } 
/*...................................................................*/
    finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*...gpuCsrScalar16*/
    printf("CsrGpuScalar16\n");
    initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                  ,&devA,&devX,&devY);
    nBlock = IBLOCKS;
/*...................................................................*/

    for(j=0;j<BLOCKS;j++){
      zero(y2,nl,"double");
      printf("CsrGpuScalar16 %d\n",nBlock);
      timeMyCudaCsrScalar16[j] = 0.0e0;
      for(i = 0;i < NSAMPLES;i++){
/*...*/ 
        timeMyCudaCsrScalar16[j]=getTimeC()-timeMyCudaCsrScalar16[j];
/*...................................................................*/

/*...*/ 
        matVecCsrGpuCudaScalar(nl,x1,y2,&devIa,&devJa,&devA
                             ,&devX,&devY,nBlock,16,1,1); 
/*...................................................................*/

/*...*/ 
        timeMyCudaCsrScalar16[j]=getTimeC()-timeMyCudaCsrScalar16[j];
/*...................................................................*/

/*...*/ 
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("CsrGpuScalar16: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
      nBlock *= 2;
    } 
/*...................................................................*/
    finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*...gpuCsrScalar32*/
    printf("CsrGpuScalar32\n");
    initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                  ,&devA,&devX,&devY);
    nBlock = IBLOCKS;
/*...................................................................*/

    for(j=0;j<BLOCKS;j++){
      zero(y2,nl,"double");
      printf("CsrGpuScalar32 %d\n",nBlock);
      timeMyCudaCsrScalar32[j] = 0.0e0;
      for(i = 0;i < NSAMPLES;i++){
/*...*/ 
        timeMyCudaCsrScalar32[j]=getTimeC()-timeMyCudaCsrScalar32[j];
/*...................................................................*/

/*...*/ 
        matVecCsrGpuCudaScalar(nl,x1,y2,&devIa,&devJa,&devA
                             ,&devX,&devY,nBlock,32,1,1); 
/*...................................................................*/

/*...*/ 
        timeMyCudaCsrScalar32[j]=getTimeC()-timeMyCudaCsrScalar32[j];
/*...................................................................*/

/*...*/ 
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("CsrGpuScalar32: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
      nBlock *= 2;
    } 
/*...................................................................*/
    finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*...gpuCsrScalar64*/
    printf("CsrGpuScalar64\n");
    initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                  ,&devA,&devX,&devY);
    nBlock = IBLOCKS;
/*...................................................................*/

    for(j=0;j<BLOCKS;j++){
      zero(y2,nl,"double");
      printf("CsrGpuScalar64 %d\n",nBlock);
      timeMyCudaCsrScalar64[j] = 0.0e0;
      for(i = 0;i < NSAMPLES;i++){
/*...*/ 
        timeMyCudaCsrScalar64[j]=getTimeC()-timeMyCudaCsrScalar64[j];
/*...................................................................*/

/*...*/ 
        matVecCsrGpuCudaScalar(nl,x1,y2,&devIa,&devJa,&devA
                             ,&devX,&devY,nBlock,64,1,1); 
/*...................................................................*/

/*...*/ 
        timeMyCudaCsrScalar64[j]=getTimeC()-timeMyCudaCsrScalar64[j];
/*...................................................................*/

/*...*/ 
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("CsrGpuScalar64: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
      nBlock *= 2;
    } 
/*...................................................................*/
    finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*...gpuCsrScalar128*/
    printf("CsrGpuScalar128\n");
    initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                  ,&devA,&devX,&devY);
    nBlock = IBLOCKS;
/*...................................................................*/

    for(j=0;j<BLOCKS;j++){
      zero(y2,nl,"double");
      printf("CsrGpuScalar128 %d\n",nBlock);
      timeMyCudaCsrScalar128[j] = 0.0e0;
      for(i = 0;i < NSAMPLES;i++){
/*...*/ 
        timeMyCudaCsrScalar128[j]=getTimeC()-timeMyCudaCsrScalar128[j];
/*...................................................................*/

/*...*/ 
        matVecCsrGpuCudaScalar(nl,x1,y2,&devIa,&devJa,&devA
                             ,&devX,&devY,nBlock,128,1,1); 
/*...................................................................*/

/*...*/ 
        timeMyCudaCsrScalar128[j]=getTimeC()-timeMyCudaCsrScalar128[j];
/*...................................................................*/

/*...*/ 
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("CsrGpuScalar128: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
      nBlock *= 2;
    } 
/*...................................................................*/
    finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*...gpuCsrScalar256*/
    printf("CsrGpuScalar256\n");
    initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                  ,&devA,&devX,&devY);
    nBlock = IBLOCKS;
/*...................................................................*/

    for(j=0;j<BLOCKS;j++){
      zero(y2,nl,"double");
      printf("CsrGpuScalar256 %d\n",nBlock);
      timeMyCudaCsrScalar256[j] = 0.0e0;
      for(i = 0;i < NSAMPLES;i++){
/*...*/ 
        timeMyCudaCsrScalar256[j]=getTimeC()-timeMyCudaCsrScalar256[j];
/*...................................................................*/

/*...*/ 
        matVecCsrGpuCudaScalar(nl,x1,y2,&devIa,&devJa,&devA
                             ,&devX,&devY,nBlock,256,1,1); 
/*...................................................................*/

/*...*/ 
        timeMyCudaCsrScalar256[j]=getTimeC()-timeMyCudaCsrScalar256[j];
/*...................................................................*/

/*...*/ 
        if(xDiffY(y2,y1,MATVECZERO,nl)){
          printf("CsrGpuScalar256: Vetores diferentes\n");
          fileOut = fopen(nameOut1,"w");
/*...*/
          fprintf(fileOut,"%6s %24s %24s %24s\n"
                 ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
          for(i=0;i<nl;i++)
            fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
            ,i+1,y1[i],y2[i],x1[i]);
          fclose(fileOut);
/*...................................................................*/
          exit(EXIT_FAILURE);
        }
/*...................................................................*/
      }
/*...................................................................*/
      nBlock *= 2;
    } 
/*...................................................................*/
    finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*...gpuCsrVecL32*/
      printf("CsrGpuVectorL32\n");
      initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                    ,&devA,&devX,&devY);
      nBlock = IBLOCKS;
/*...................................................................*/
      for(j=0;j<BLOCKS;j++){
        zero(y2,nl,"double");
        printf("CsrGpuVectorL32 %d\n",nBlock);
        timeMyCudaCsrVectorL32[j] = 0.0e0;
        for(i = 0;i < NSAMPLES;i++){
/*...*/ 
          timeMyCudaCsrVectorL32[j]=getTimeC()-timeMyCudaCsrVectorL32[j];
/*...................................................................*/

/*...*/ 
          matVecCsrGpuCudaVector(nl,x1,y2,&devIa,&devJa,&devA
                                ,&devX,&devY,nBlock,32,1,1); 
/*...................................................................*/

/*...*/ 
          timeMyCudaCsrVectorL32[j]=getTimeC()-timeMyCudaCsrVectorL32[j];
/*...................................................................*/

/*...*/ 
          if(xDiffY(y2,y1,MATVECZERO,nl)){
            printf("CsrGpuVectorL32: Vetores diferentes\n");
            fileOut = fopen(nameOut1,"w");
/*...*/
            fprintf(fileOut,"%6s %24s %24s %24s\n"
                   ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
            for(i=0;i<nl;i++)
              fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
              ,i+1,y1[i],y2[i],x1[i]);
            fclose(fileOut);
/*...................................................................*/
            exit(EXIT_FAILURE);
          }
/*...................................................................*/
      }
/*...................................................................*/
        nBlock *= 2;
    } 
/*...................................................................*/
     finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*...gpuCsrVecCuspRt2008*/
      printf("CsrGpuVecCuspRt2008\n");
      initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                    ,&devA,&devX,&devY);
      nBlock = IBLOCKS;
/*...................................................................*/
      for(j=0;j<BLOCKS;j++){
        zero(y2,nl,"double");
        printf("CsrGpuVectorCuspRt2008 %d\n",nBlock);
        timeMyCudaCsrVectorCuspRt2008[j] = 0.0e0;
        for(i = 0;i < NSAMPLES;i++){
/*...*/ 
          timeMyCudaCsrVectorCuspRt2008[j]
          =getTimeC()-timeMyCudaCsrVectorCuspRt2008[j];
/*...................................................................*/

/*...*/ 
          matVecCsrGpuCudaVectorCusp(nl,x1,y2,&devIa,&devJa,&devA
                                    ,&devX,&devY,nBlock,1,1,0); 
/*...................................................................*/

/*...*/ 
          timeMyCudaCsrVectorCuspRt2008[j]
          =getTimeC()-timeMyCudaCsrVectorCuspRt2008[j];
/*...................................................................*/

/*...*/ 
          if(xDiffY(y2,y1,MATVECZERO,nl)){
            printf("CsrGpuVectorCuspRt2008: Vetores diferentes\n");
            fileOut = fopen(nameOut1,"w");
/*...*/
            fprintf(fileOut,"%6s %24s %24s %24s\n"
                   ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
            for(i=0;i<nl;i++)
              fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
              ,i+1,y1[i],y2[i],x1[i]);
            fclose(fileOut);
/*...................................................................*/
            exit(EXIT_FAILURE);
          }
/*...................................................................*/
      }
/*...................................................................*/
        nBlock *= 2;
    } 
/*...................................................................*/
     finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*...gpuCsrVecCuspSc2009*/
      printf("CsrGpuVecCuspSc2009\n");
      initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                    ,&devA,&devX,&devY);
      nBlock = IBLOCKS;
/*...................................................................*/
      for(j=0;j<BLOCKS;j++){
        zero(y2,nl,"double");
        printf("CsrGpuVectorCuspSc2009 %d\n",nBlock);
        timeMyCudaCsrVectorCuspSc2009[j] = 0.0e0;
        for(i = 0;i < NSAMPLES;i++){
/*...*/ 
          timeMyCudaCsrVectorCuspSc2009[j]
          =getTimeC()-timeMyCudaCsrVectorCuspSc2009[j];
/*...................................................................*/

/*...*/ 
          matVecCsrGpuCudaVectorCusp(nl,x1,y2,&devIa,&devJa,&devA
                                    ,&devX,&devY,nBlock,1,1,1); 
/*...................................................................*/

/*...*/ 
          timeMyCudaCsrVectorCuspSc2009[j]
          =getTimeC()-timeMyCudaCsrVectorCuspSc2009[j];
/*...................................................................*/

/*...*/ 
          if(xDiffY(y2,y1,MATVECZERO,nl)){
            printf("CsrGpuVectorCuspSc2009: Vetores diferentes\n");
            fileOut = fopen(nameOut1,"w");
/*...*/
            fprintf(fileOut,"%6s %24s %24s %24s\n"
                   ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
            for(i=0;i<nl;i++)
              fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
              ,i+1,y1[i],y2[i],x1[i]);
            fclose(fileOut);
/*...................................................................*/
            exit(EXIT_FAILURE);
          }
/*...................................................................*/
      }
/*...................................................................*/
        nBlock *= 2;
    } 
/*...................................................................*/
     finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*...gpuCsrVecCuspV1*/
      printf("CsrGpuVecCuspV1\n");
      initCsrGpuCuda(nl,iaCsr,jaCsr,aCsr,x1,y2,&devIa,&devJa
                    ,&devA,&devX,&devY);
      nBlock = IBLOCKS;
/*...................................................................*/
      for(j=0;j<BLOCKS;j++){
        zero(y2,nl,"double");
        printf("CsrGpuVectorCuspV1 %d\n",nBlock);
        timeMyCudaCsrVectorCuspV1[j] = 0.0e0;
        for(i = 0;i < NSAMPLES;i++){
/*...*/ 
          timeMyCudaCsrVectorCuspV1[j]
          =getTimeC()-timeMyCudaCsrVectorCuspV1[j];
/*...................................................................*/

/*...*/ 
          matVecCsrGpuCudaVectorCusp(nl,x1,y2,&devIa,&devJa,&devA
                                    ,&devX,&devY,nBlock,1,1,1); 
/*...................................................................*/

/*...*/ 
          timeMyCudaCsrVectorCuspV1[j]
          =getTimeC()-timeMyCudaCsrVectorCuspV1[j];
/*...................................................................*/

/*...*/ 
          if(xDiffY(y2,y1,MATVECZERO,nl)){
            printf("CsrGpuVectorCuspV1: Vetores diferentes\n");
            fileOut = fopen(nameOut1,"w");
/*...*/
            fprintf(fileOut,"%6s %24s %24s %24s\n"
                   ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
            for(i=0;i<nl;i++)
              fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
              ,i+1,y1[i],y2[i],x1[i]);
            fclose(fileOut);
/*...................................................................*/
            exit(EXIT_FAILURE);
          }
/*...................................................................*/
      }
/*...................................................................*/
        nBlock *= 2;
    } 
/*...................................................................*/
     finalizeCsrGpuCuda(&devIa,&devJa,&devA,&devX,&devY);      
/*...................................................................*/

/*... Produto da operacao y=Ax*/  
    fileOut = fopen(nameOut1,"w");
    fprintf(fileOut,"neq=%d nad=%d Band Max=%ld Band Med=%ld " 
                    "nl Max=%ld nl Med=%ld\n"
                    ,nl,nadCsr,bandMax,bandMed,nlMax,nlMed);
    fprintf(fileOut,"%6s %24s %24s %24s\n"
                   ,"ndim","vetor y1 (controle)","vetor y2","vetor x1");
    for(i=0;i<nl;i++)
      fprintf(fileOut,"%6d %24.10lf %24.10lf %24.10lf\n"
      ,i+1,y1[i],y2[i],x1[i]);
    fclose(fileOut);
/*...................................................................*/
    

    fileOut = fopen(nameOut2,"w");
/*... cudaCsr*/
    fprintf(fileOut,"%30s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
           "%10s=%16.8lf %10s=%16.8lf %10s=%16.8lf\n"
          ,"cudaCsr"
          ,"time"   
          ,timeCudaCsr
          ,"gflop"
          ,gFlopCsr     
          ,"gflops"
          ,(NSAMPLES*gFlopCsr)/timeCudaCsr   
          ,"ntime"
          ,timeCudaCsr/timeCudaCsr
          ,"overHead"
          ,timeOverHeadCuda
          ,"Time Event"
          ,timeCudaCsrEvent/1000.0);
/*...................................................................*/

/*... gpuCsrScalar*/
    fprintf(fileOut,"%30s %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuScalar"
            ,"time"   
            ,timeMyCudaCsrScalar
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrScalar   
           ,"ntime"
           ,timeMyCudaCsrScalar/timeCudaCsr);
/*...................................................................*/

/*... gpuCsrScalar16*/
    nBlock = IBLOCKS;
    for(j=0;j<BLOCKS;j++){
      fprintf(fileOut,"%25s%5d %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuScalar16-"
            ,nBlock
            ,"time"   
            ,timeMyCudaCsrScalar16[j]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrScalar16[j]   
           ,"ntime"
           ,timeMyCudaCsrScalar16[j]/timeCudaCsr);
      nBlock *= 2;
    }
/*...................................................................*/

/*... gpuCsrScalar32*/
    nBlock = IBLOCKS;
    for(j=0;j<BLOCKS;j++){
      fprintf(fileOut,"%25s%5d %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuScalar32-"
            ,nBlock
            ,"time"   
            ,timeMyCudaCsrScalar32[j]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrScalar32[j]   
           ,"ntime"
           ,timeMyCudaCsrScalar32[j]/timeCudaCsr);
      nBlock *= 2;
    }
/*...................................................................*/
 
/*... gpuCsrScalar64*/
    nBlock = IBLOCKS;
    for(j=0;j<BLOCKS;j++){
      fprintf(fileOut,"%25s%5d %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuScalar64-"
            ,nBlock
            ,"time"   
            ,timeMyCudaCsrScalar64[j]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrScalar64[j]   
           ,"ntime"
           ,timeMyCudaCsrScalar64[j]/timeCudaCsr);
      nBlock *= 2;
    }
/*...................................................................*/

/*... gpuCsrScalar128*/
    nBlock = IBLOCKS;
    for(j=0;j<BLOCKS;j++){
      fprintf(fileOut,"%25s%5d %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuScalar128-"
            ,nBlock
            ,"time"   
            ,timeMyCudaCsrScalar128[j]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrScalar128[j]   
           ,"ntime"
           ,timeMyCudaCsrScalar128[j]/timeCudaCsr);
      nBlock *= 2;
    }
/*...................................................................*/

/*... gpuCsrScalar256*/
    nBlock = IBLOCKS;
    for(j=0;j<BLOCKS;j++){
      fprintf(fileOut,"%25s%5d %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuScalar256-"
            ,nBlock
            ,"time"   
            ,timeMyCudaCsrScalar256[j]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrScalar256[j]   
           ,"ntime"
           ,timeMyCudaCsrScalar256[j]/timeCudaCsr);
      nBlock *= 2;
    }
/*...................................................................*/

/*... gpuCsrVecL32*/
    nBlock = IBLOCKS;
    for(j=0;j<BLOCKS;j++){
      fprintf(fileOut,"%25s%5d %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuVecL32-"
            ,nBlock
            ,"time"   
            ,timeMyCudaCsrVectorL32[j]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrVectorL32[j]   
           ,"ntime"
           ,timeMyCudaCsrVectorL32[j]/timeCudaCsr);
      nBlock *= 2;
    }
/*...................................................................*/

/*... gpuCsrVecCuspRt2008*/
    nBlock = IBLOCKS;
    for(j=0;j<BLOCKS;j++){
      fprintf(fileOut,"%25s%5d %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuVecCuspRt2008-"
            ,nBlock
            ,"time"   
            ,timeMyCudaCsrVectorCuspRt2008[j]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrVectorCuspRt2008[j]   
           ,"ntime"
           ,timeMyCudaCsrVectorCuspRt2008[j]/timeCudaCsr);
      nBlock *= 2;
    }
/*...................................................................*/

/*... gpuCsrVecCuspSc2009*/
    nBlock = IBLOCKS;
    for(j=0;j<BLOCKS;j++){
      fprintf(fileOut,"%25s%5d %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuVecCuspSc2009-"
            ,nBlock
            ,"time"   
            ,timeMyCudaCsrVectorCuspSc2009[j]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrVectorCuspSc2009[j]   
           ,"ntime"
           ,timeMyCudaCsrVectorCuspSc2009[j]/timeCudaCsr);
      nBlock *= 2;
    }
/*...................................................................*/

/*... gpuCsrVecCuspV1*/
    nBlock = IBLOCKS;
    for(j=0;j<BLOCKS;j++){
      fprintf(fileOut,"%25s%5d %10s=%16.8lf %10s=%16.8lf %10s=%16.8lf "
             "%10s=%16.8lf\n"
            ,"cudaGpuVecCuspV1-"
            ,nBlock
            ,"time"   
            ,timeMyCudaCsrVectorCuspV1[j]
            ,"gflop"
            ,gFlopCsr     
            ,"gflops"
           ,(NSAMPLES*gFlopCsr)/timeMyCudaCsrVectorCuspV1[j]   
           ,"ntime"
           ,timeMyCudaCsrVectorCuspV1[j]/timeCudaCsr);
      nBlock *= 2;
    }
/*...................................................................*/

/*...*/
    fclose(fileOut);
/*...................................................................*/
  
  } 
/*.....................................................................*/
#endif

/*...*/
  else{
    fprintf(stderr,"Opcao invalida!!\n");
    ERRO_ARGS(argv,NARGS,nameArgs2);
  }
/*...................................................................*/
 

  return EXIT_SUCCESS;
}
