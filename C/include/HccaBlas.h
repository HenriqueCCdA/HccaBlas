#ifndef _HCCABLAS_
  #define _HCCABLAS_
  #define NFUNC           144
  #define HCCABLASZERO 1.0e-8
  #include<Define.h>
  #include<stdio.h>
  #ifdef _OPENMP
    #include<omp.h>
    double tmpDotOmp;
    double tmpDotOmp1;
    double tmpDotOmp2;
    double tmpDotOmp3;
    double tmpDotOmp4;
    double tmpDotOmp5;
    double tmpDotOmp6;
    double tmpDotOmp7;
    double tmpDotOmp8;
  #endif

/*...*/
  void getNameHccaBlas(void);
  long flopMatVecFull(INT nLin,INT nCol);
  long flopMatVecCsr(INT neq,INT nad,short ty);
  long flopDot(INT nDim);
/*produto vetorial*/  
  void prodVet(double *restrict a,double *restrict b
              ,double *restrict c);
/*...................................................................*/

/*level 1*/
  INT xDiffY(double *restrict x,double *restrict y
            ,double const tol  , INT n);
  double dot(double *restrict x1,double *restrict x2,INT n);
  double dotO2L2(double *restrict x1,double *restrict x2,INT n);
  double   dotL2(double *restrict x1,double *restrict x2,INT n);
  double   dotL4(double *restrict x1,double *restrict x2,INT n);
  double   dotL6(double *restrict x1,double *restrict x2,INT n);
  double   dotL8(double *restrict x1,double *restrict x2,INT n);
  double   dotO2(double *restrict x1,double *restrict x2,INT n);
  double   dotO4(double *restrict x1,double *restrict x2,INT n);
  double   dotO6(double *restrict x1,double *restrict x2,INT n);
  double   dotO8(double *restrict x1,double *restrict x2,INT n);
#if _OPENMP
  double     dotOmp(double *restrict x1,double *restrict x2,INT n);
  double dotOmpO2L2(double *restrict x1,double *restrict x2,INT n);
  double dotOmpO2L4(double *restrict x1,double *restrict x2,INT n);
  double   dotOmpL2(double *restrict x1,double *restrict x2,INT n);
  double   dotOmpL4(double *restrict x1,double *restrict x2,INT n);
  double   dotOmpL6(double *restrict x1,double *restrict x2,INT n);
  double   dotOmpL8(double *restrict x1,double *restrict x2,INT n);
  double   dotOmpO2(double *restrict x1,double *restrict x2,INT n);
  double   dotOmpO4(double *restrict x1,double *restrict x2,INT n);
  double   dotOmpO6(double *restrict x1,double *restrict x2,INT n);
  double   dotOmpO8(double *restrict x1,double *restrict x2,INT n);
#endif
/*...................................................................*/

/*level 2*/
/* ... matriz cheia*/
  void matVecFull(double *restrict a
                 ,double *restrict x
                 ,double *restrict y
                 ,INT nLin          ,INT nCol);
  void matVecFullO2(double *restrict a
                 ,double *restrict x
                 ,double *restrict y
                 ,INT nLin          ,INT nCol);
  void matVecFullO4(double *restrict a
                   ,double *restrict x
                   ,double *restrict y
                   ,INT nLin          ,INT nCol);
  void matVecFullO2I2(double *restrict a
                     ,double *restrict x
                     ,double *restrict y
                     ,INT nLin          ,INT nCol);
  void matVecFullO4I2(double *restrict a
                     ,double *restrict x
                     ,double *restrict y
                     ,INT nLin          ,INT nCol);
  void matVecFullO4I4(double *restrict a
                     ,double *restrict x
                     ,double *restrict y
                     ,INT nLin          ,INT nCol);
/*...................................................................*/

/*...Csr*/
  typedef enum {csr=1,csrD=2,csrC=3} typeCsr;

/*... CsrD*/ 
  void     matVecCsrD(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void   matVecCsrDI2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void   matVecCsrDI4(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void   matVecCsrDI6(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void   matVecCsrDO2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void   matVecCsrDO4(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void   matVecCsrDO6(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void matVecCsrDO2I2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
#ifdef _OPENMP
/*... CsrDOmp*/ 
  void       matVecCsrDOmp(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void       matVecCsrDOmpI2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void       matVecCsrDOmpI4(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void       matVecCsrDOmpI6(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void       matVecCsrDOmpO2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void       matVecCsrDOmpO4(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void       matVecCsrDOmpO6(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
  void       matVecCsrDOmpO2I2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y);
/*balanciamento manual do trabalho entre as threads*/  
  void matVecCsrDOmpBal(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y
                     ,INT  *thBegin     ,INT *thEnd  );
  void matVecCsrDOmpBalI2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y
                     ,INT  *thBegin     ,INT *thEnd  );
  void matVecCsrDOmpBalI4(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y
                     ,INT  *thBegin     ,INT *thEnd  );
  void matVecCsrDOmpBalI6(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y
                     ,INT  *thBegin     ,INT *thEnd  );
  void matVecCsrDOmpBalO2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y
                     ,INT  *thBegin     ,INT *thEnd  );
  void matVecCsrDOmpBalO4(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y
                     ,INT  *thBegin     ,INT *thEnd  );
  void matVecCsrDOmpBalO6(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y
                     ,INT  *thBegin     ,INT *thEnd  );
  void matVecCsrDOmpBalO2I2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y
                     ,INT  *thBegin     ,INT *thEnd  );


#endif         
/*... CsrC*/ 
  void     matVecCsrC(INT neq           
                     ,INT *restrict ia   ,INT *restrict ja
                     ,double *restrict au,double *restrict ad
                     ,double *restrict al
                     ,double *restrict x ,double *restrict y);
  void   matVecCsrCI2(INT neq           
                     ,INT *restrict ia   ,INT *restrict ja
                     ,double *restrict au,double *restrict ad
                     ,double *restrict al
                     ,double *restrict x ,double *restrict y);
  void   matVecCsrCI4(INT neq           
                     ,INT *restrict ia   ,INT *restrict ja
                     ,double *restrict au,double *restrict ad
                     ,double *restrict al
                     ,double *restrict x ,double *restrict y);
  void   matVecCsrCI6(INT neq           
                     ,INT *restrict ia   ,INT *restrict ja
                     ,double *restrict au,double *restrict ad
                     ,double *restrict al
                     ,double *restrict x ,double *restrict y);
  void   matVecCsrCO2(INT neq           
                     ,INT *restrict ia   ,INT *restrict ja
                     ,double *restrict au,double *restrict ad
                     ,double *restrict al
                     ,double *restrict x ,double *restrict y);
  void   matVecCsrCO4(INT neq           
                     ,INT *restrict ia   ,INT *restrict ja
                     ,double *restrict au,double *restrict ad
                     ,double *restrict al
                     ,double *restrict x ,double *restrict y);
  void   matVecCsrCO6(INT neq           
                     ,INT *restrict ia   ,INT *restrict ja
                     ,double *restrict au,double *restrict ad
                     ,double *restrict al
                     ,double *restrict x ,double *restrict y);
  void matVecCsrCO2I2(INT neq           
                     ,INT *restrict ia   ,INT *restrict ja
                     ,double *restrict au,double *restrict ad
                     ,double *restrict al
                     ,double *restrict x ,double *restrict y);
#ifdef _OPENMP
  void     matVecCsrCOmp(INT neq           
                        ,INT *restrict ia    ,INT *restrict ja
                        ,double *restrict au ,double *restrict ad
                        ,double *restrict al
                        ,double *restrict x  ,double *restrict y
                        ,INT  *thBegin       ,INT *thEnd  
                        ,INT  *thHeight    
                        ,double *restrict thY,int nThreads);          
  void     matVecCsrCOmpI2(INT neq           
                          ,INT *restrict ia    ,INT *restrict ja
                          ,double *restrict au ,double *restrict ad
                          ,double *restrict al
                          ,double *restrict x  ,double *restrict y
                          ,INT  *thBegin       ,INT *thEnd  
                          ,INT  *thHeight    
                          ,double *restrict thY,int nThreads);          
  void     matVecCsrCOmpI4(INT neq           
                          ,INT *restrict ia    ,INT *restrict ja
                          ,double *restrict au ,double *restrict ad
                          ,double *restrict al
                          ,double *restrict x  ,double *restrict y
                          ,INT  *thBegin       ,INT *thEnd  
                          ,INT  *thHeight    
                          ,double *restrict thY,int nThreads);          
  void     matVecCsrCOmpI6(INT neq           
                          ,INT *restrict ia    ,INT *restrict ja
                          ,double *restrict au ,double *restrict ad
                          ,double *restrict al
                          ,double *restrict x  ,double *restrict y
                          ,INT  *thBegin       ,INT *thEnd  
                          ,INT  *thHeight    
                          ,double *restrict thY,int nThreads);          
  void     matVecCsrCOmpO2(INT neq           
                          ,INT *restrict ia    ,INT *restrict ja
                          ,double *restrict au ,double *restrict ad
                          ,double *restrict al
                          ,double *restrict x  ,double *restrict y
                          ,INT  *thBegin       ,INT *thEnd  
                          ,INT  *thHeight    
                          ,double *restrict thY,int nThreads);          
  void     matVecCsrCOmpO4(INT neq           
                          ,INT *restrict ia    ,INT *restrict ja
                          ,double *restrict au ,double *restrict ad
                          ,double *restrict al
                          ,double *restrict x  ,double *restrict y
                          ,INT  *thBegin       ,INT *thEnd  
                          ,INT  *thHeight    
                          ,double *restrict thY,int nThreads);          
  void     matVecCsrCOmpO6(INT neq           
                          ,INT *restrict ia    ,INT *restrict ja
                          ,double *restrict au ,double *restrict ad
                          ,double *restrict al
                          ,double *restrict x  ,double *restrict y
                          ,INT  *thBegin       ,INT *thEnd  
                          ,INT  *thHeight    
                          ,double *restrict thY,int nThreads);          
  void     matVecCsrCOmpO2I2(INT neq           
                            ,INT *restrict ia    ,INT *restrict ja
                            ,double *restrict au ,double *restrict ad
                            ,double *restrict al
                            ,double *restrict x  ,double *restrict y
                            ,INT  *thBegin       ,INT *thEnd  
                            ,INT  *thHeight    
                            ,double *restrict thY,int nThreads);          
#endif         
/*... Csr*/ 
  void     matVecCsr(INT neq           
                    ,INT *restrict ia  ,INT *restrict ja
                    ,double *restrict a,double *restrict x
                    ,double *restrict y);
  void   matVecCsrI2(INT neq           
                    ,INT *restrict ia  ,INT *restrict ja
                    ,double *restrict a,double *restrict x
                    ,double *restrict y);
  void   matVecCsrI4(INT neq           
                    ,INT *restrict ia  ,INT *restrict ja
                    ,double *restrict a,double *restrict x
                    ,double *restrict y);
  void   matVecCsrI6(INT neq           
                    ,INT *restrict ia  ,INT *restrict ja
                    ,double *restrict a,double *restrict x
                    ,double *restrict y);
  void   matVecCsrO2(INT neq           
                    ,INT *restrict ia  ,INT *restrict ja
                    ,double *restrict a,double *restrict x
                    ,double *restrict y);
  void   matVecCsrO4(INT neq           
                    ,INT *restrict ia  ,INT *restrict ja
                    ,double *restrict a,double *restrict x
                    ,double *restrict y);
  void   matVecCsrO6(INT neq           
                    ,INT *restrict ia  ,INT *restrict ja
                    ,double *restrict a,double *restrict x
                    ,double *restrict y);
  void matVecCsrO2I2(INT neq           
                    ,INT *restrict ia  ,INT *restrict ja
                    ,double *restrict a,double *restrict x
                    ,double *restrict y);
#ifdef _OPENMP
/*... CsrOmp*/ 
  void       matVecCsrOmp(INT neq           
                         ,INT *restrict ia  ,INT *restrict ja
                         ,double *restrict a,double *restrict x
                         ,double *restrict y);
  void       matVecCsrOmpI2(INT neq           
                           ,INT *restrict ia  ,INT *restrict ja
                           ,double *restrict a,double *restrict x
                           ,double *restrict y);
  void       matVecCsrOmpI4(INT neq           
                           ,INT *restrict ia  ,INT *restrict ja
                           ,double *restrict a,double *restrict x
                           ,double *restrict y);
  void       matVecCsrOmpI6(INT neq           
                           ,INT *restrict ia  ,INT *restrict ja
                           ,double *restrict a,double *restrict x
                           ,double *restrict y);
  void       matVecCsrOmpO2(INT neq           
                           ,INT *restrict ia  ,INT *restrict ja
                           ,double *restrict a,double *restrict x
                           ,double *restrict y);
  void       matVecCsrOmpO4(INT neq           
                           ,INT *restrict ia  ,INT *restrict ja
                           ,double *restrict a,double *restrict x
                           ,double *restrict y);
  void       matVecCsrOmpO6(INT neq           
                           ,INT *restrict ia  ,INT *restrict ja
                           ,double *restrict a,double *restrict x
                           ,double *restrict y);
  void     matVecCsrOmpO2I2(INT neq           
                           ,INT *restrict ia  ,INT *restrict ja
                           ,double *restrict a,double *restrict x
                           ,double *restrict y);
#endif
/*...................................................................*/
#endif
