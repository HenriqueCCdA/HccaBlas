#ifndef _MYTIME_H_
  #define _MYTIME_H_
  #if _OPENMP
    #include<Openmp.h>
  #elif _WIN32 
    #include<time.h>  
  #else  
    #include<sys/time.h>
  #endif  
  #include <stdio.h>
  typedef struct Time{
    double solv  ;
    double dot   ;
    double matvec;
    double pform;
    double tform;
    double color;
    double total;
  }Time;
  Time tm;
#ifdef __cplusplus
  extern "C"{
#endif
    double getTimeC(void);
#ifdef __cplusplus
  }
#endif
#endif/*_MYTIME_H_*/
