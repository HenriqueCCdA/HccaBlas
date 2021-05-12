#include<HccaTime.h>

double getTimeC(void){
/**********************************************************************
 * GETTIMEC : Obter tempo                                             *
 * -------------------------------------------------------------------*
 * Parametros de Entrada:                                             *
 * -------------------------------------------------------------------*
 * -------------------------------------------------------------------*
 * Parametros de Saida:                                               *
 * -------------------------------------------------------------------*
 * -------------------------------------------------------------------*
*********************************************************************/

/*funcao do openmp*/
#if _OPENMP
  double t=0.0;
  t = omp_get_wtime();
  return t;
/*funcao do nativa do C (baixa precisao)*/
#elif _WIN32
   clock_t start;
   double  t;

   start = clock();
   t = (double)(start) / CLOCKS_PER_SEC;
   return t;
#else
  double t =0.0;
  struct timeval start;

  gettimeofday(&start,NULL);
  t = (double) (start.tv_sec + start.tv_usec/1000000.0);
  
  return t;
#endif

}

/*********************************************************************/
