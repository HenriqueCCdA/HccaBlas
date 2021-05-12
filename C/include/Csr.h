#ifndef _CSR_H  
  #define _CSR_H
  #include<Mystdbool.h>
  #include<Define.h>

/*...*/
  INT csrIa(INT *ia  ,INT *id    ,INT *num    ,INT  *adj  ,short *nViz
           ,INT numel,INT neq    ,short maxViz,short ndf  ,bool upper
           ,bool diag, bool lower);
  void csrJa(INT *ia     ,INT *ja 
            ,INT *id  ,INT *num ,INT  *adj    ,short *nViz
            ,INT numel,INT neq   ,short maxViz,short ndf
            ,bool upper,bool diag,bool lower);
/*...................................................................*/

/*...*/
  INT bandCsr(INT *ia,INT *ja,INT  neq,short type);
  INT nlCsr(INT *ia,INT  neq,short type);
/*...................................................................*/

/*... OPENMP*/
  #ifdef _OPENMP
  #include<omp.h>
  void partitionCsrByNonzeros(INT *ia          ,INT *ja         
                             ,INT neq
                             ,int numThreads   ,INT *threadBegin
                             ,INT *threadEnd   ,INT *threadSize
                             ,INT *threadHeigth,short type);
  void computeEffectiveWork(INT *ia         ,INT *ja         ,INT neq
                         ,INT *threadBegin,INT *threadEnd  
                         ,INT *threadSize ,INT *thHeigth   );
  #endif
/*...................................................................*/

/*...*/
  void sortGraphCsr(INT *ia,INT *ja,INT n);
/*...................................................................*/

#endif/*_CSR_H*/
