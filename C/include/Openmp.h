#ifndef _OPENMP_H
  #define _OPENMP_H
  #include<HccaStdBool.h>
  #include<stdio.h>
  #include<Memoria.h>
  #if _OPENMP
    #include<omp.h>
  #endif  
  typedef struct Omp{
    short ncore;
    bool openmp;
    bool loopwise;
    bool solv;
    bool pform;
  }Omp;
  double  dot_omp;

  bool my_omp(void);
  int set_num_threads(int,bool);


#endif/*_OPENMP_H*/
