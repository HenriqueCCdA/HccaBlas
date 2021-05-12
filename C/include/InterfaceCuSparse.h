#ifndef _INTERFCUSPARSE_H  
  #define _INTERFCUSPARSE_H
  #include<Define.h>
  #include<HccaTime.h>
  void csrCuSparse(INT neq             ,INT nad 
                  ,double *x           ,double *y
                  ,double *timeOverHead,double *timeKernell);   
  void initCuSparse(INT neq  ,INT *ia            ,INT*ja
                  ,double *a,double *timeOverHead);  
#endif/*_COO_H*/
