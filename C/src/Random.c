#include<HccaRandom.h>
/********************************************************************* 
 * RANDOMMATRIX : gera um matriz de numero reais aleatorios          * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a    -> nao definido                                              * 
 * nLin -> numero de linhas                                          * 
 * nCol -> numero de colunas                                         * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * a    -> matrix de numeros reais aleatorios                        * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void randomMatrix(double *a,int nLin,int nCol){
  int i,j;
  srand((unsigned) time(NULL)/2);
  
/*vetor*/  
  if(nCol == 0){
    for(i=0;i<nLin;i++)
      a[i] = pow(-1,i)*rand()/100000000.0;
  }
/*Matriz*/  
  else{
    for(i=0;i<nLin;i++)
      for(j=0;j<nCol;j++)
        MAT2D(i,j,a,nCol) = pow(-1,i)*rand()/1000.0;
  }
}
