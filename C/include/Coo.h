#ifndef _COO_H  
  #define _COO_H
   #ifdef _AD_
    #undef  _AD_
  #endif  
  #define _AD_ false
  #include<Memoria.h>
  #include<stdio.h>
  #include<stdlib.h>
  #include<Define.h>
  #include<Graph.h>
  #ifdef _MMIO_
    #include<mmio/mmio.h>
  #endif
/*...*/
  void writeCoo(Memoria *m,INT *ia   ,INT *ja,INT neq
               ,INT nad,short type
               ,bool unsym,char* name);

/*...*/
  void csrToCoo(int  *linha,int  *col,double *val,INT *ia,INT *ja
               ,INT neq ,INT nad);
  void cooToFull(int *lin, int *col,double *val,double *a,INT neq
                ,INT nad);
  void cooToCsr(int *lin  , int *col ,double *val
               ,INT *ia   ,INT *ja
               ,double *au,double *ad,double *al
               ,INT neq   ,INT nad   ,short type  
               ,int *aux  ,bool upper,bool diag ,bool lower);
/*...*/
  void matrixCheck(double *val,int *lin,int *col, int nl,int nnz);


#endif/*_COO_H*/
