#include<HccaBlas.h>

/********************************************************************** 
 * HCCABLAS : biblioteca de algebra linear                            * 
 *------------------------------------------------------------------- * 
 * nome das funcoes                                                   * 
 *------------------------------------------------------------------- * 
 * pordVet        -> produto vetorial                                 * 
 * xDiffy         -> verifica da diferenca entre o vetor x e y        * 
 * flopMatVecFull -> FLOP para operacao matriz*vector, onde a matriz  *
 * denso                                                              * 
 * ------------------------- DOT ------------------------------------ * 
 * dot            -> produto interno entre dois vetores               * 
 * dotO2L2        -> produto interno entre dois vetores               * 
 * dotL2          -> produto interno entre dois vetores               * 
 * dotL4          -> produto interno entre dois vetores               * 
 * dotL6          -> produto interno entre dois vetores               * 
 * dotL8          -> produto interno entre dois vetores               * 
 * dotO2          -> produto interno entre dois vetores               * 
 * dotO4          -> produto interno entre dois vetores               * 
 * dotO6          -> produto interno entre dois vetores               * 
 * dotO8          -> produto interno entre dois vetores               * 
 * ---------------------- MATVECFULL -------------------------------- * 
 * matVecFull     -> matriz vetor para matriz cheia                   * 
 * matVecFullO2   -> matriz vetor para matriz cheia                   * 
 * matVecFullO4   -> matriz vetor para matriz cheia                   * 
 * matVecFullO2I2 -> matriz vetor para matriz cheia                   * 
 * matVecFullO4I2 -> matriz vetor para matriz cheia                   * 
 * matVecFullO4I4 -> matriz vetor para matriz cheia                   * 
 * -------------------------- CSRD ---------------------------------- * 
 * matVecCsrD     -> matriz vetor para matriz geral no formato CSRD   * 
 * matVecCsrDI2   -> matriz vetor para matriz geral no formato CSRD   * 
 * matVecCsrDI4   -> matriz vetor para matriz geral no formato CSRD   * 
 * matVecCsrDI6   -> matriz vetor para matriz geral no formato CSRD   * 
 * matVecCsrDO2   -> matriz vetor para matriz geral no formato CSRD   * 
 * matVecCsrDO4   -> matriz vetor para matriz geral no formato CSRD   * 
 * matVecCsrDO6   -> matriz vetor para matriz geral no formato CSRD   * 
 * matVecCsrDO2I2 -> matriz vetor para matriz geral no formato CSRD   * 
 * -------------------------- CSRC ---------------------------------- * 
 * matVecCsrC     -> matriz vetor para matriz geral no formato CSRC   * 
 * matVecCsrCI2   -> matriz vetor para matriz geral no formato CSRC   * 
 * matVecCsrCI4   -> matriz vetor para matriz geral no formato CSRC   * 
 * matVecCsrCI6   -> matriz vetor para matriz geral no formato CSRC   * 
 * matVecCsrCO2   -> matriz vetor para matriz geral no formato CSRC   * 
 * matVecCsrCO4   -> matriz vetor para matriz geral no formato CSRC   * 
 * matVecCsrCO6   -> matriz vetor para matriz geral no formato CSRC   * 
 * matVecCsrCO2I2 -> matriz vetor para matriz geral no formato CSRC   * 
 * -------------------------- CSR ----------------------------------- * 
 * matVecCsr      -> matriz vetor para matriz geral no formato CSR    * 
 * matVecCsrI2    -gvim riz vetor para matriz geral no formato CSR    * 
 * matVecCsrI4    -> matriz vetor para matriz geral no formato CSR    * 
 * matVecCsrI6    -> matriz vetor para matriz geral no formato CSR    * 
 * matVecCsrO2    -> matriz vetor para matriz geral no formato CSR    * 
 * matVecCsrO4    -> matriz vetor para matriz geral no formato CSR    * 
 * matVecCsrO6    -> matriz vetor para matriz geral no formato CSR    * 
 * matVecCsrO2I2  -> matriz vetor para matriz geral no formato CSR    * 
 * ------------------------ OPENMP ---------------------------------- * 
 * ------------------------- DOT ------------------------------------ * 
 * dotOmp         -> produto interno entre dois vetores               * 
 * dotOmpO2L2     -> produto interno entre dois vetores               * 
 * dotOmpO2L4     -> produto interno entre dois vetores               * 
 * dotOmpL2       -> produto interno entre dois vetores               * 
 * dotOmpL4       -> produto interno entre dois vetores               * 
 * dotOmpL6       -> produto interno entre dois vetores               * 
 * dotOmpL8       -> produto interno entre dois vetores               * 
 * dotOmpO2       -> produto interno entre dois vetores               * 
 * dotOmpO4       -> produto interno entre dois vetores               * 
 * dotOmpO6       -> produto interno entre dois vetores               * 
 * dotOmpO8       -> produto interno entre dois vetores               * 
 * -------------------------- CSRD ---------------------------------- * 
 * matVecCsrDOmp       -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpI2     -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpI4     -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpI6     -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpO2     -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpO4     -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpO6     -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpO2I2   -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpBal    -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpBalI2  -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpBalI4  -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpBalI6  -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpBalO2  -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpBalO4  -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpBalO6  -> mat vet para matriz geral no formato CSRD   * 
 * matVecCsrDOmpBalO2I2-> mat vet para matriz geral no formato CSRD   * 
 * -------------------------- CSRC ---------------------------------- * 
 * matVecCsrCOmp       -> mat vet para matriz strucSym no formato CSRC* 
 * matVecCsrOmpI2  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpI4  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpI6  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpO2  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpO4  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpO6  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpO2I2-> matriz vetor para matriz geral no formato CSR   * 
 * -------------------------- CSR ----------------------------------- * 
 * matVecCsrOmp    -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpI2  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpI4  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpI6  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpO2  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpO4  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpO6  -> matriz vetor para matriz geral no formato CSR   * 
 * matVecCsrOmpO2I2-> matriz vetor para matriz geral no formato CSR   * 
 *------------------------------------------------------------------- * 
 *********************************************************************/

/*********************************************************************/ 
void getNameHccaBlas(void){
  int i;
  char hccaBlas[NFUNC][30] = {
 "prodVet"             ,"xDiffy"            ,"flopMatVecFull"     /* 1*/
,"dot"                 ,"dotO2L2"           ,"dotL2"              /* 2*/
,"dotL4"               ,"dotL6"             ,"dotL8"              /* 3*/
,"dotO2"               ,"dotO4"             ,"dotO6"              /* 4*/
,"dotO8"               ,""                  ,""                   /* 5*/
,"matVecFull"          ,"matVecFullO2"      ,"matVecFullO4"       /* 6*/
,"matVecFullO2I2"      ,"matVecFullO4I2"    ,"matVecFullO4I4"     /* 7*/
,"matVecCsrD"          ,"matVecCsrDI2"      ,"matVecCsrDI4"       /* 8*/
,"matVecCsrDI6"        ,""                  ,""                   /* 9*/
,"matVecCsrDO2"        ,"matVecCsrDO4"      ,"matVecCsrDO6"       /*10*/
,"matVecCsrDO2I2"      ,""                  ,""                   /*11*/  
,"matVecCsrC"          ,"matVecCsrCI2"      ,"matVecCsrCI4"       /*12*/
,"matVecCsrCI6"        ,""                  ,""                   /*13*/
,"matVecCsrCO2"        ,"matVecCsrCO4"      ,"matVecCsrCO6"       /*14*/
,"matVecCsrCO2I2"      ,""                  ,""                   /*15*/
,"matVecCsrCO2I2"      ,""                  ,""                   /*16*/
,"matVecCsrI2"         ,"matVecCsrI4"       ,"matVecCsrI6"        /*17*/
,"matVecCsrO2"         ,"matVecCsrO4"       ,"matVecCsrO6"        /*18*/
,"matVecCsrO2I2"       ,""                  ,""                   /*19*/
,"matVecCsr"           ,"matVecCsrI2"       ,"matVecCsrI4"        /*20*/
,"matVecCsrI6"         ,""                  ,""                   /*21*/
,"matVecCsrO2"         ,"matVecCsrO4"       ,"matVecCsrO6"        /*22*/
,"matVecCsrO2I2"       ,""                  ,""                   /*23*/
,"matVecCsrI2"         ,"matVecCsrI4"       ,"matVecCsrI6"        /*24*/
,"matVecCsrO2"         ,"matVecCsrO4"       ,""                   /*25*/
,"matVecCsrO2I2"       ,""                  ,""                   /*26*/
,""                    ,""                  ,""                   /*27*/
,""                    ,""                  ,""                   /*28*/
,"dotOmp"              ,"dotOmpOmpO2L2"     ,"dotOmpL2"           /*29*/
,"dotOmpL4"            ,"dotOmpL6"          ,"dotOmpL8"           /*30*/
,"dotOmpO2"            ,"dotOmpO4"          ,"dotOmpO6"           /*31*/
,"dotOmpO8"            ,""                  ,""                   /*32*/
,"matVecCsrDOmp"       ,"matVecCsrDOmpO2"   ,"matVecCsrDOmpO4"    /*33*/
,"matVecCsrDOmpO6"     ,""                  ,""                   /*34*/
,"matVecCsrDOmpI2 "    ,"matVecCsrDOmpI4"   ,"matVecCsrDOmpI6"    /*35*/
,"matVecCsrDOmpO2I2"   ,""                  ,""                   /*36*/
,"matVecCsrDOmpBal"    ,"matVecCsrDOmpBalO2","matVecCsrDOmpBalO4" /*37*/
,"matVecCsrDOmpBalO6"  ,""                  ,""                   /*38*/
,"matVecCsrDOmpBalI2 " ,"matVecCsrDOmpBalI4","matVecCsrDOmpBalI6" /*39*/
,"matVecCsrDOmpBalO2I2",""                  ,""                   /*40*/
,"matVecCsrCOmp   "    ,"matVecCsrCOmpO2"   ,"matVecCsrCOmpO4"    /*41*/
,"matVecCsrCOmpO6"     ,""                  ,""                   /*42*/
,"matVecCsrCOmpI2 "    ,"matVecCsrCOmpI4"   ,"matVecCsrCOmpI6"    /*43*/
,"matVecCsrCOmpO2I2"   ,""                  ,""                   /*44*/
,"matVecCsrOmp"        ,"matVecCsrOmpO2"    ,"matVecCsrOmpO4"     /*45*/
,"matVecCsrOmpO6"      ,""                  ,""                   /*46*/
,"matVecCsrOmpI2"      ,"matVecCsrOmpI4"    ,"matVecCsrOmpI6"     /*47*/
,"matVecCsrOmpO2I2"    ,""                  ,""                   /*48*/
};
/*...*/
  printf("File name: %s\n",__FILE__);
  for(i=0;i<NFUNC;i++)
    printf("function name: %s\n",hccaBlas[i]);
/*...................................................................*/
}
/*********************************************************************/ 

/********************************************************************* 
 * FLOPDOT: Calculo do numero de flops do produto interno            * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * nDim -> numero de dimensoes                                       * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * Numero de flops                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
long flopDot(INT nDim){
  return (2*nDim-1);
}
/*********************************************************************/ 

/********************************************************************* 
 * FLOPMATVECFULL : Calculo do numero de flops da operacao de matiz  * 
 * por um vetor, onde a matriz e uma matriz densa                    * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * nLin -> numero de linhas                                          * 
 * nCol -> numero de colunas                                         * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * Numero de flops                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
long flopMatVecFull(INT nLin,INT nCol){
  return nLin*nCol;
}
/*********************************************************************/ 

/********************************************************************* 
 * FLOPMATVECCSR  : Calculo do numero de flops da operacao de matiz  * 
 * por um vetor, onde a matriz e uma matriz esparsa no formato CSR   * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq  -> numero de linhas                                          * 
 * nad  -> numero total de elementos nao nulos                       * 
 * ty   -> tipo do formato csr                                       * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * Numero de flops                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
long flopMatVecCsr(INT neq,INT nad,short ty){
  
  INT flop;
  switch(ty){
/*... csr padrao*/
    case 1:
      flop = 2*nad;
/*... vetor a*/
    break;
/*... csrWdm*/
    case 2:
      flop = 2*nad + neq;
    break;
/*... csrC*/
    case 3:
      flop = 4*nad + neq;
    break;
/*...*/
    default:
      ERRO_OP(__FILE__,__func__,ty);
      break;
  }
/*...................................................................*/
  return flop;
}
/*********************************************************************/ 

/********************************************************************* 
 * XDIFFY: diferencao entre o vetor x e y                            * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x  ->vetor x                                                      * 
 * y  ->vetor y                                                      * 
 * tol->tolerancia da comparacao                                     * 
 * n  ->numero da dimensao                                           * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *      numero de diferencao enter os vetores                        * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
INT xDiffY(double *restrict x,double *restrict y, double const tol
          ,INT n )
{
  double c;
  INT i,diff=0;
  
  for(i=0;i<n;i++){
    c = x[i] - y[i];
    if( c < 0.0 ) c = -c;
    if( c > tol){
      printf("i=%d diff=%e\n",i,c);
      diff++;
    }
  }
  
  return diff;
}
/*********************************************************************/ 

/********************************************************************* 
 * DOT: produto interno entre dois vetores                           * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dot(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  double tmp=0.0;
  for(i=0;i<n;i++)
    tmp += x1[i]*x2[i];
  return tmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTO2L2: produto interno entre dois vetores                       * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotO2L2(double *restrict x1,double *restrict x2,INT n)
{
  INT i,resto;
  double tmp1=0.0,tmp2=0.0;
  
  resto = n%4;
  if(resto == 1)
     tmp1 = x1[0]*x2[0]; 
  else if(resto == 2)
     tmp1 = x1[0]*x2[0] + x1[1]*x2[1]; 
  else if(resto == 3)
     tmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
    
  for(i=resto;i<n;i+=4){
    tmp1 += x1[i  ]*x2[i  ] + x1[i+1]*x2[i+1]; 
    tmp2 += x1[i+2]*x2[i+2] + x1[i+3]*x2[i+3]; 
  }
  return (tmp1+tmp2);
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTL2: produto interno entre dois vetores                         * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotL2(double *restrict x1,double *restrict x2,INT n)
{
  INT i,resto;
  double tmp1=0.0,tmp2=0.0;
  resto = n%2;
  
  if(resto)
    tmp1 = x1[0]*x2[0];
    
  for(i=resto;i<n;i+=2){
    tmp1 += x1[i  ]*x2[i  ]; 
    tmp2 += x1[i+1]*x2[i+1];
  }
  return (tmp1+tmp2);
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTL4: produto interno entre dois vetores                         * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotL4(double *restrict x1,double *restrict x2,INT n)
{
  INT i,resto;
  double tmp1=0.0,tmp2=0.0,tmp3=0.0,tmp4=0.0;
  
  
  resto = n%4;
  if(resto == 1)
     tmp1 = x1[0]*x2[0]; 
  else if(resto == 2)
     tmp1 = x1[0]*x2[0] + x1[1]*x2[1]; 
  else if(resto == 3)
     tmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
    
  
  for(i=resto;i<n;i+=4){
    tmp1 += x1[i  ]*x2[i  ];
    tmp2 += x1[i+1]*x2[i+1];
    tmp3 += x1[i+2]*x2[i+2];
    tmp4 += x1[i+3]*x2[i+3];
  }
  return (tmp1+tmp2+tmp3+tmp4);
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTL6: produto interno entre dois vetores                         * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotL6(double *restrict x1,double *restrict x2,INT n)
{
  INT i,resto;
  double tmp1=0.0,tmp2=0.0,tmp3=0.0,tmp4=0.0;
  double tmp5=0.0,tmp6=0.0;
  
  
  resto = n%6;
  if(resto == 1)
     tmp1= x1[0]*x2[0]; 
  else if(resto == 2)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1]; 
  else if(resto == 3)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
  else if(resto == 3)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
  else if(resto == 4)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] + x1[3]*x2[3]; 
  else if(resto == 5)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] 
         + x1[2]*x2[2] + x1[3]*x2[3]  
         + x1[4]*x2[4];               
  
  for(i=resto;i<n;i+=6){
    tmp1 += x1[i  ]*x2[i  ];
    tmp2 += x1[i+1]*x2[i+1];
    tmp3 += x1[i+2]*x2[i+2];
    tmp4 += x1[i+3]*x2[i+3];
    tmp5 += x1[i+4]*x2[i+4];
    tmp6 += x1[i+5]*x2[i+5];
  }
  return (tmp1+tmp2+tmp3+tmp4+tmp5+tmp6);
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTL8: produto interno entre dois vetores                         * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotL8(double *restrict x1,double *restrict x2,INT n)
{
  INT i,resto;
  double tmp1=0.0,tmp2=0.0,tmp3=0.0,tmp4=0.0;
  double tmp5=0.0,tmp6=0.0,tmp7=0.0,tmp8=0.0;
  
  
  resto = n%8;
  if(resto == 1)
     tmp1= x1[0]*x2[0]; 
  else if(resto == 2)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1]; 
  else if(resto == 3)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
  else if(resto == 3)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
  else if(resto == 4)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] + x1[3]*x2[3]; 
  else if(resto == 5)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] 
         + x1[2]*x2[2] + x1[3]*x2[3]  
         + x1[4]*x2[4];               
  else if(resto == 6)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] 
         + x1[2]*x2[2] + x1[3]*x2[3]  
         + x1[4]*x2[4] + x1[5]*x2[5]; 
  else if(resto == 7)
     tmp1= x1[0]*x2[0] + x1[1]*x2[1] 
         + x1[2]*x2[2] + x1[3]*x2[3]  
         + x1[4]*x2[4] + x1[5]*x2[5]  
         + x1[6]*x2[6];
  
  for(i=resto;i<n;i+=8){
    tmp1 += x1[i  ]*x2[i  ];
    tmp2 += x1[i+1]*x2[i+1];
    tmp3 += x1[i+2]*x2[i+2];
    tmp4 += x1[i+3]*x2[i+3];
    tmp5 += x1[i+4]*x2[i+4];
    tmp6 += x1[i+5]*x2[i+5];
    tmp7 += x1[i+6]*x2[i+6];
    tmp8 += x1[i+7]*x2[i+7];
  }
  return (tmp1+tmp2+tmp3+tmp4+tmp5+tmp6+tmp7+tmp8);
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTO2: produto interno entre dois vetores                         * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotO2(double *restrict x1,double *restrict x2,INT n)
{
  INT i,resto;
  double tmp=0.0;
  resto = n%2;
  
  if(resto)
    tmp = x1[0]*x2[0];
    
  for(i=resto;i<n;i+=2)
    tmp += x1[i]*x2[i] + x1[i+1]*x2[i+1];
  return tmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTO4: produto interno entre dois vetores                         * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotO4(double *restrict x1,double *restrict x2,INT n)
{
  INT i,resto;
  double tmp=0.0;
  
  resto = n%4;
  if(resto == 1)
     tmp = x1[0]*x2[0]; 
  else if(resto == 2)
     tmp = x1[0]*x2[0] + x1[1]*x2[1]; 
  else if(resto == 3)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
    
  
  for(i=resto;i<n;i+=4)
    tmp += x1[i  ]*x2[i  ] + x1[i+1]*x2[i+1] 
         + x1[i+2]*x2[i+2] + x1[i+3]*x2[i+3];
  return tmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTO6: produto interno entre dois vetores                         * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotO6(double *restrict x1,double *restrict x2,INT n)
{
  INT i,resto;
  double tmp=0.0;
  
  resto = n%6;
  if(resto == 1)
     tmp = x1[0]*x2[0]; 
  else if(resto == 2)
     tmp = x1[0]*x2[0] + x1[1]*x2[1]; 
  else if(resto == 3)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
  else if(resto == 4)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] + x1[3]*x2[3];
  else if(resto == 5)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] 
         + x1[3]*x2[3] + x1[4]*x2[4];
    
  
  for(i=resto;i<n;i+=6)
    tmp += x1[i  ]*x2[i  ] + x1[i+1]*x2[i+1] 
         + x1[i+2]*x2[i+2] + x1[i+3]*x2[i+3] 
         + x1[i+4]*x2[i+4] + x1[i+5]*x2[i+5];
  return tmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTO8: produto interno entre dois vetores                         * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotO8(double *restrict x1,double *restrict x2,INT n)
{
  INT i,resto;
  double tmp=0.0;
  
  resto = n%8;
  if(resto == 1)
     tmp = x1[0]*x2[0]; 
  else if(resto == 2)
     tmp = x1[0]*x2[0] + x1[1]*x2[1]; 
  else if(resto == 3)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
  else if(resto == 3)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]; 
  else if(resto == 4)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] + x1[3]*x2[3]; 
  else if(resto == 5)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] 
         + x1[2]*x2[2] + x1[3]*x2[3]  
         + x1[4]*x2[4];               
  else if(resto == 6)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] 
         + x1[2]*x2[2] + x1[3]*x2[3]  
         + x1[4]*x2[4] + x1[5]*x2[5]; 
  else if(resto == 7)
     tmp = x1[0]*x2[0] + x1[1]*x2[1] 
         + x1[2]*x2[2] + x1[3]*x2[3]  
         + x1[4]*x2[4] + x1[5]*x2[5]  
         + x1[6]*x2[6];

  for(i=resto;i<n;i+=8)
    tmp += x1[i  ]*x2[i  ] + x1[i+1]*x2[i+1] 
         + x1[i+2]*x2[i+2] + x1[i+3]*x2[i+3]
         + x1[i+4]*x2[i+4] + x1[i+5]*x2[i+5]
         + x1[i+6]*x2[i+6] + x1[i+7]*x2[i+7];
  return tmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * PRODVET: produto vetorial                                         * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a  - vetor a                                                      * 
 * b  - vetor b                                                      * 
 * c  - indefinido                                                   * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * c  - produto vetoria entre a e b                                  * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void prodVet(double *restrict a,double *restrict b, double *restrict c)
{
  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0];
}
/*********************************************************************/ 

/********************************************************************* 
 * MATVECFULL : operacao matriz vetor para matriz cheias             * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a  - vetor a                                                      * 
 * x  - vetor b                                                      * 
 * y  - indefinido                                                   * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y  - produto vetoria entre a e b                                  * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecFull(double *restrict a
               ,double *restrict x
               ,double *restrict y
               ,INT nLin          ,INT nCol)
{
  INT i,j;
  
  for(i=0;i<nLin;i++){
    y[i] =0.0;
    for(j=0;j<nCol;j++){
      y[i] += MAT2D(i,j,a,nCol)*x[j];
    }
  }
}
/*********************************************************************/ 

/********************************************************************* 
 * MATVECFULLO2: operayyo matriz vetor para matriz cheias            * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a  - vetor a                                                      * 
 * x  - vetor b                                                      * 
 * y  - indefinido                                                   * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y  - produto vetoria entre a e b                                  * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecFullO2(double *restrict a
                 ,double *restrict x
                 ,double *restrict y
                 ,INT nLin          ,INT nCol)
{
  INT i,j;
  int resto;
  
  resto = nLin%2;
/*...*/
  if(resto){
    y[0] = 0.0;
    for(j=0;j<nCol;j++){
      y[0]   += MAT2D(0  ,j,a,nCol)*x[j];
    }
  }
/*...................................................................*/

/*...*/
  for(i=resto;i<nLin;i+=2){
    y[i]   =0.0;
    y[i+1] =0.0;
    for(j=0;j<nCol;j++){
      y[i]   += MAT2D(i    ,j,a,nCol)*x[j];
      y[i+1] += MAT2D((i+1),j,a,nCol)*x[j];
    }
  }
/*...................................................................*/
}
/*********************************************************************/ 

/********************************************************************* 
 * MATVECFULLO4: operacao matriz vetor para matriz cheias            * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a  - vetor a                                                      * 
 * x  - vetor b                                                      * 
 * y  - indefinido                                                   * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y  - produto vetoria entre a e b                                  * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecFullO4(double *restrict a
                 ,double *restrict x
                 ,double *restrict y
                 ,INT nLin          ,INT nCol)
{
  INT i,j;
  int resto;
  
  resto = nLin%4;
/*...*/
  if(resto == 1){
    y[0] = 0.0;
    for(j=0;j<nCol;j++){
      y[0]   += MAT2D(0  ,j,a,nCol)*x[j];
    }
  }
  else if(resto == 2){
    y[0] = 0.0;
    y[1] = 0.0;
    for(j=0;j<nCol;j++){
      y[0]   += MAT2D(0  ,j,a,nCol)*x[j];
      y[1]   += MAT2D(1  ,j,a,nCol)*x[j];
    }
  }
  else if(resto == 3){
    y[0] = 0.0;
    y[1] = 0.0;
    y[2] = 0.0;
    for(j=0;j<nCol;j++){
      y[0]   += MAT2D(0  ,j,a,nCol)*x[j];
      y[1]   += MAT2D(1  ,j,a,nCol)*x[j];
      y[2]   += MAT2D(2  ,j,a,nCol)*x[j];
    }
  }
/*...................................................................*/

/*...*/
  for(i=resto;i<nLin;i+=4){
    y[i]   =0.0;
    y[i+1] =0.0;
    y[i+2] =0.0;
    y[i+3] =0.0;
    for(j=0;j<nCol;j++){
      y[i]   += MAT2D(i    ,j,a,nCol)*x[j];
      y[i+1] += MAT2D((i+1),j,a,nCol)*x[j];
      y[i+2] += MAT2D((i+2),j,a,nCol)*x[j];
      y[i+3] += MAT2D((i+3),j,a,nCol)*x[j];
    }
  }
/*...................................................................*/
}
/*********************************************************************/ 

/********************************************************************* 
 * MATVECFULLO2I2: operacao matriz vetor para matriz cheias          * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a  - vetor a                                                      * 
 * x  - vetor b                                                      * 
 * y  - indefinido                                                   * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y  - produto vetoria entre a e b                                  * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecFullO2I2(double *restrict a
                   ,double *restrict x
                   ,double *restrict y
                   ,INT nLin          ,INT nCol)
{
  INT i,j;
  int resto1,resto2;
  double aux1,aux2,aux3,aux4;
  
  resto1 = nLin%2;
  resto2 = nCol%2;
/*...*/
  if(resto1){
    aux1 = 0.0;
    aux2 = 0.0;
    if(resto2)
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
    for(j=resto2;j<nCol;j+=2){
      aux1   += MAT2D(0  ,j    ,a,nCol)*x[j];
      aux2   += MAT2D(0  ,(j+1),a,nCol)*x[j+1];
    }
    y[0] = aux1 + aux2;
  }
/*...................................................................*/

/*...*/
  for(i=resto1;i<nLin;i+=2){
    aux1   = 0.0;
    aux2   = 0.0;
    aux3   = 0.0;
    aux4   = 0.0;
    if(resto2){
      aux1   += MAT2D(i    ,0  ,a,nCol)*x[0];
      aux2   += MAT2D((i+1),0  ,a,nCol)*x[0];
    }

    for(j=resto2;j<nCol;j+=2){
      aux1 += MAT2D(i    ,j    ,a,nCol)*x[j];
      aux3 += MAT2D(i    ,(j+1),a,nCol)*x[j+1];
      aux2 += MAT2D((i+1),j    ,a,nCol)*x[j];  
      aux4 += MAT2D((i+1),(j+1),a,nCol)*x[j+1];
    }
    y[i  ] = aux1 + aux3;
    y[i+1] = aux2 + aux4;
  }
/*...................................................................*/
}
/*********************************************************************/ 

/********************************************************************* 
 * MATVECFULLO4I2: operacao matriz vetor para matriz cheias          * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a  - vetor a                                                      * 
 * x  - vetor b                                                      * 
 * y  - indefinido                                                   * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y  - produto vetoria entre a e b                                  * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecFullO4I2(double *restrict a
                   ,double *restrict x
                   ,double *restrict y
                   ,INT nLin          ,INT nCol)
{
  INT i,j;
  int resto1,resto2;
  double aux1,aux2,aux3,aux4,aux5,aux6,aux7,aux8;
  
  resto1 = nLin%4;
  resto2 = nCol%2;
/*...*/
  if(resto1 == 1){
    aux1 = 0.0;
    aux3 = 0.0;
    if(resto2)
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
    for(j=resto2;j<nCol;j+=2){
      aux1   += MAT2D(0  ,j    ,a,nCol)*x[j];
      aux3   += MAT2D(0  ,(j+1),a,nCol)*x[j+1];
    }
    y[0] = aux1 + aux3;
  }
  else if(resto1 == 2){
    aux1 = 0.0;
    aux2 = 0.0;
    aux3 = 0.0;
    aux4 = 0.0;
    if(resto2){
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux2 = MAT2D(1  ,0  ,a,nCol)*x[0];
    }
    for(j=resto2;j<nCol;j+=2){
      aux1   += MAT2D(0  ,    j,a,nCol)*x[j];
      aux3   += MAT2D(0  ,(j+1),a,nCol)*x[j+1];
      aux2   += MAT2D(1  ,    j,a,nCol)*x[j];
      aux4   += MAT2D(1  ,(j+1),a,nCol)*x[j+1];
    }
    y[0] = aux1 + aux3;
    y[1] = aux2 + aux4;
  }
  else if(resto1 == 3){
    aux1 = 0.0;
    aux2 = 0.0;
    aux3 = 0.0;
    aux4 = 0.0;
    aux5 = 0.0;
    aux6 = 0.0;
    if(resto2){
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux2 = MAT2D(1  ,0  ,a,nCol)*x[0];
      aux5 = MAT2D(2  ,0  ,a,nCol)*x[0];
    }
    for(j=resto2;j<nCol;j+=2){
      aux1   += MAT2D(0  ,    j,a,nCol)*x[j];
      aux3   += MAT2D(0  ,(j+1),a,nCol)*x[j+1];
      aux2   += MAT2D(1  ,    j,a,nCol)*x[j];
      aux4   += MAT2D(1  ,(j+1),a,nCol)*x[j+1];
      aux5   += MAT2D(2  ,    j,a,nCol)*x[j];  
      aux6   += MAT2D(2  ,(j+1),a,nCol)*x[j+1];
    }
    y[0] = aux1 + aux3;
    y[1] = aux2 + aux4;
    y[2] = aux5 + aux6;
  }
/*...................................................................*/

/*...*/
  for(i=resto1;i<nLin;i+=4){
    aux1 = 0.0;
    aux2 = 0.0;
    aux3 = 0.0;
    aux4 = 0.0;
    aux5 = 0.0;
    aux6 = 0.0;
    aux7 = 0.0;
    aux8 = 0.0;
    if(resto2){
      aux1 = MAT2D(i    ,0  ,a,nCol)*x[0];
      aux2 = MAT2D((i+1),0  ,a,nCol)*x[0];
      aux5 = MAT2D((i+2),0  ,a,nCol)*x[0];
      aux7 = MAT2D((i+3),0  ,a,nCol)*x[0];
    }
    for(j=resto2;j<nCol;j+=2){
      aux1   += MAT2D(i    ,    j,a,nCol)*x[j];
      aux3   += MAT2D(i    ,(j+1),a,nCol)*x[j+1];
      aux2   += MAT2D((i+1),    j,a,nCol)*x[j];
      aux4   += MAT2D((i+1),(j+1),a,nCol)*x[j+1];
      aux5   += MAT2D((i+2),    j,a,nCol)*x[j];
      aux6   += MAT2D((i+2),(j+1),a,nCol)*x[j+1];
      aux7   += MAT2D((i+3),    j,a,nCol)*x[j];
      aux8   += MAT2D((i+3),(j+1),a,nCol)*x[j+1];
    }
    y[i]   = aux1 + aux3;
    y[i+1] = aux2 + aux4;
    y[i+2] = aux5 + aux6;
    y[i+3] = aux7 + aux8;
  }
/*...................................................................*/
}
/*********************************************************************/ 

/********************************************************************* 
 * MATVECFULLO4I4: operacao matriz vetor para matriz cheias          * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * a  - vetor a                                                      * 
 * x  - vetor b                                                      * 
 * y  - indefinido                                                   * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y  - produto vetoria entre a e b                                  * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecFullO4I4(double *restrict a
                   ,double *restrict x
                   ,double *restrict y
                   ,INT nLin          ,INT nCol)
{
  INT i,j;
  int resto1,resto2;
  double aux1 , aux2, aux3, aux4;
  double aux5 , aux6, aux7, aux8;
  double aux9 ,aux10,aux11,aux12;
  double aux13,aux14,aux15,aux16;
  
  resto1 = nLin%4;
  resto2 = nCol%4;
/*...*/
  if(resto1 == 1){
    aux1 = 0.0;
    aux3 = 0.0;
    aux5 = 0.0;
    aux7 = 0.0;
    if(resto2 == 1){
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
    }
    else if(resto2 == 2){
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux3 = MAT2D(0  ,1  ,a,nCol)*x[1];
    }
    else if(resto2 == 3){ 
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux3 = MAT2D(0  ,1  ,a,nCol)*x[1];
      aux5 = MAT2D(0  ,2  ,a,nCol)*x[2];
    }
    for(j=resto2;j<nCol;j+=4){
      aux1   += MAT2D(0  ,    j,a,nCol)*x[j];
      aux3   += MAT2D(0  ,(j+1),a,nCol)*x[j+1];
      aux5   += MAT2D(0  ,(j+2),a,nCol)*x[j+2];
      aux7   += MAT2D(0  ,(j+3),a,nCol)*x[j+3];
    }
    y[0] = aux1 + aux3 + aux5 + aux7;
  }
  else if(resto1 == 2){
    aux1 = 0.0;
    aux2 = 0.0;
    aux3 = 0.0;
    aux4 = 0.0;
    aux5 = 0.0;
    aux6 = 0.0;
    aux7 = 0.0;
    aux8 = 0.0;
    if(resto2 == 1){
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux2 = MAT2D(1  ,0  ,a,nCol)*x[0];
    }
    else if(resto2 == 2){
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux3 = MAT2D(0  ,1  ,a,nCol)*x[1];
      aux2 = MAT2D(1  ,0  ,a,nCol)*x[0];
      aux4 = MAT2D(1  ,1  ,a,nCol)*x[1];
    }
    else if(resto2 == 3){ 
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux3 = MAT2D(0  ,1  ,a,nCol)*x[1];
      aux5 = MAT2D(0  ,2  ,a,nCol)*x[2];
      aux2 = MAT2D(1  ,0  ,a,nCol)*x[0];
      aux4 = MAT2D(1  ,1  ,a,nCol)*x[1];
      aux6 = MAT2D(1  ,2  ,a,nCol)*x[2];
    }
    for(j=resto2;j<nCol;j+=4){
      aux1   += MAT2D(0  ,    j,a,nCol)*x[j];
      aux3   += MAT2D(0  ,(j+1),a,nCol)*x[j+1];
      aux5   += MAT2D(0  ,(j+2),a,nCol)*x[j+2];
      aux7   += MAT2D(0  ,(j+3),a,nCol)*x[j+3];
      aux2   += MAT2D(1  ,    j,a,nCol)*x[j];
      aux4   += MAT2D(1  ,(j+1),a,nCol)*x[j+1];
      aux6   += MAT2D(1  ,(j+2),a,nCol)*x[j+2];
      aux8   += MAT2D(1  ,(j+3),a,nCol)*x[j+3];
    }
    y[0] = aux1 + aux3 + aux5 + aux7;
    y[1] = aux2 + aux4 + aux6 + aux8;
  }
  else if(resto1 == 3){
    aux1 = 0.0;
    aux2 = 0.0;
    aux3 = 0.0;
    aux4 = 0.0;
    aux5 = 0.0;
    aux6 = 0.0;
    aux7 = 0.0;
    aux8 = 0.0;
    aux9 = 0.0;
    aux10= 0.0;
    aux11= 0.0;
    aux12= 0.0;
    if(resto2 == 1){
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux2 = MAT2D(1  ,0  ,a,nCol)*x[0];
      aux9 = MAT2D(2  ,0  ,a,nCol)*x[0];
    }
    else if(resto2 == 2){
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux3 = MAT2D(0  ,1  ,a,nCol)*x[1];
      aux2 = MAT2D(1  ,0  ,a,nCol)*x[0];
      aux4 = MAT2D(1  ,1  ,a,nCol)*x[1];
      aux9 = MAT2D(2  ,1  ,a,nCol)*x[0];
      aux10= MAT2D(2  ,1  ,a,nCol)*x[1];
    }
    else if(resto2 == 3){ 
      aux1 = MAT2D(0  ,0  ,a,nCol)*x[0];
      aux3 = MAT2D(0  ,1  ,a,nCol)*x[1];
      aux5 = MAT2D(0  ,2  ,a,nCol)*x[2];
      aux2 = MAT2D(1  ,0  ,a,nCol)*x[0];
      aux4 = MAT2D(1  ,1  ,a,nCol)*x[1];
      aux6 = MAT2D(1  ,2  ,a,nCol)*x[2];
      aux9 = MAT2D(2  ,0  ,a,nCol)*x[0];
      aux10= MAT2D(2  ,1  ,a,nCol)*x[1];
      aux11= MAT2D(2  ,2  ,a,nCol)*x[2];
    }
    for(j=resto2;j<nCol;j+=4){
      aux1   += MAT2D(0  ,    j,a,nCol)*x[j];
      aux3   += MAT2D(0  ,(j+1),a,nCol)*x[j+1];
      aux5   += MAT2D(0  ,(j+2),a,nCol)*x[j+2];
      aux7   += MAT2D(0  ,(j+3),a,nCol)*x[j+3];
      aux2   += MAT2D(1  ,    j,a,nCol)*x[j];
      aux4   += MAT2D(1  ,(j+1),a,nCol)*x[j+3];
      aux6   += MAT2D(1  ,(j+2),a,nCol)*x[j+2];
      aux8   += MAT2D(1  ,(j+3),a,nCol)*x[j+3];
      aux9   += MAT2D(2  ,    j,a,nCol)*x[j];
      aux10  += MAT2D(2  ,(j+1),a,nCol)*x[j+3];
      aux11  += MAT2D(2  ,(j+2),a,nCol)*x[j+2];
      aux12  += MAT2D(2  ,(j+3),a,nCol)*x[j+3];
    }
    y[0] = aux1 +  aux3 + aux5  +  aux7;
    y[1] = aux2 +  aux4 + aux6  +  aux8;
    y[2] = aux9 + aux10 + aux11 + aux12;
  }
/*...................................................................*/

/*...*/
  for(i=resto1;i<nLin;i+=4){
    aux1  = 0.0;
    aux2  = 0.0;
    aux3  = 0.0;
    aux4  = 0.0;
    aux5  = 0.0;
    aux6  = 0.0;
    aux7  = 0.0;
    aux8  = 0.0;
    aux9  = 0.0;
    aux10 = 0.0;
    aux11 = 0.0;
    aux12 = 0.0;
    aux13 = 0.0;
    aux14 = 0.0;
    aux15 = 0.0;
    aux16 = 0.0;
    if(resto2 == 1){
      aux1  = MAT2D(i    ,0  ,a,nCol)*x[0];
      aux2  = MAT2D((i+1),0  ,a,nCol)*x[0];
      aux9  = MAT2D((i+2),0  ,a,nCol)*x[0];
      aux13 = MAT2D((i+3),0  ,a,nCol)*x[0];
    }
    else if(resto2 == 2){
      aux1 = MAT2D(i    ,0  ,a,nCol)*x[0];
      aux3 = MAT2D(i    ,1  ,a,nCol)*x[1];
      aux2 = MAT2D((i+1),0  ,a,nCol)*x[0];
      aux4 = MAT2D((i+1),1  ,a,nCol)*x[1];
      aux9 = MAT2D((i+2),0  ,a,nCol)*x[0];
      aux10= MAT2D((i+2),1  ,a,nCol)*x[1];
      aux13= MAT2D((i+3),0  ,a,nCol)*x[0];
      aux14= MAT2D((i+3),1  ,a,nCol)*x[1];
    }
    else if(resto2 == 3){
      aux1 = MAT2D(i    ,0  ,a,nCol)*x[0];
      aux3 = MAT2D(i    ,1  ,a,nCol)*x[1];
      aux5 = MAT2D(i    ,2  ,a,nCol)*x[2];
      aux2 = MAT2D((i+1),0  ,a,nCol)*x[0];
      aux4 = MAT2D((i+1),1  ,a,nCol)*x[1];
      aux6 = MAT2D((i+1),2  ,a,nCol)*x[2];
      aux9 = MAT2D((i+2),0  ,a,nCol)*x[0];
      aux10= MAT2D((i+2),1  ,a,nCol)*x[1];
      aux11= MAT2D((i+2),2  ,a,nCol)*x[2];
      aux13= MAT2D((i+3),0  ,a,nCol)*x[0];
      aux14= MAT2D((i+3),1  ,a,nCol)*x[1];
      aux15= MAT2D((i+3),2  ,a,nCol)*x[2];
    }
    for(j=resto2;j<nCol;j+=4){
      aux1   += MAT2D(i    ,    j,a,nCol)*x[j];
      aux3   += MAT2D(i    ,(j+1),a,nCol)*x[j+1];
      aux5   += MAT2D(i    ,(j+2),a,nCol)*x[j+2];
      aux7   += MAT2D(i    ,(j+3),a,nCol)*x[j+3];
      aux2   += MAT2D((i+1),    j,a,nCol)*x[j];
      aux4   += MAT2D((i+1),(j+1),a,nCol)*x[j+1];
      aux6   += MAT2D((i+1),(j+2),a,nCol)*x[j+2];
      aux8   += MAT2D((i+1),(j+3),a,nCol)*x[j+3];
      aux9   += MAT2D((i+2),    j,a,nCol)*x[j];
      aux10  += MAT2D((i+2),(j+1),a,nCol)*x[j+1];
      aux11  += MAT2D((i+2),(j+2),a,nCol)*x[j+2];
      aux12  += MAT2D((i+2),(j+3),a,nCol)*x[j+3];
      aux13  += MAT2D((i+3),    j,a,nCol)*x[j];
      aux14  += MAT2D((i+3),(j+1),a,nCol)*x[j+1];
      aux15  += MAT2D((i+3),(j+2),a,nCol)*x[j+2];
      aux16  += MAT2D((i+3),(j+3),a,nCol)*x[j+3];
    }
    y[i]   =  aux1 +  aux3 + aux5  + aux7;
    y[i+1] =  aux2 +  aux4 + aux6  + aux8;
    y[i+2] =  aux9 + aux10 + aux11 + aux12;
    y[i+3] = aux13 + aux14 + aux15 + aux16;
  }
/*...................................................................*/
}
/*********************************************************************/

/********************************************************************* 
 * MATVECCSRD  :produto matriz vetor para uma matriz generica no     *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrD(INT neq           
              ,INT *restrict ia  ,INT *restrict ja
              ,double *restrict a,double *restrict ad
              ,double *restrict x,double *restrict y)
{
  INT i,j;
  double tmp;
  for(i=0;i<neq;i++){
    tmp = ad[i]*x[i];
    for(j=ia[i];j<ia[i+1];j++)
      tmp += a[j]*x[ja[j]];
    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDI2:  produto matriz vetor para uma matriz generica no   *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDI2(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i;
  INT ia1,ia2,ja1;
  int resto,n;
  double tmp;

  for(i=0;i<neq;i++){
    tmp   = ad[i]*x[i];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    
    n     = ia2 - ia1;
    resto = n%2;
    
    if(resto)
      tmp += a[ia1]*x[ja[ia1]];

    for(ja1=ia1+resto;ja1<ia2;ja1+=2)
      tmp += a[  ja1]*x[ja[  ja1]]
           + a[ja1+1]*x[ja[ja1+1]];

    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDI4:  produto matriz vetor para uma matriz generica no   *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDI4(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i;
  INT ia1,ia2,ja1;
  int resto,n;
  double tmp;

  for(i=0;i<neq;i++){
    tmp   = ad[i]*x[i];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    
    n     = ia2 - ia1;
    resto = n%4;
    
    if(resto == 3)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      tmp += a[ia1]*x[ja[ia1]]; 

    for(ja1=ia1+resto;ja1<ia2;ja1+=4)
      tmp += a[  ja1]*x[ja[  ja1]]
           + a[ja1+1]*x[ja[ja1+1]] 
           + a[ja1+2]*x[ja[ja1+2]] 
           + a[ja1+3]*x[ja[ja1+3]];

    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDI6:  produto matriz vetor para uma matriz generica no   *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDI6(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i;
  INT ia1,ia2,ja1;
  int resto,n;
  double tmp;

  for(i=0;i<neq;i++){
    tmp   = ad[i]*x[i];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    
    n     = ia2 - ia1;
    resto = n%6;
    
    if(resto == 5)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]  
           + a[ia1+4]*x[ja[ia1+4]]; 
    else if(resto == 4)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]; 
    else if(resto == 3)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      tmp += a[ia1]*x[ja[ia1]]; 

    for(ja1=ia1+resto;ja1<ia2;ja1+=6)
      tmp += a[  ja1]*x[ja[  ja1]]
           + a[ja1+1]*x[ja[ja1+1]] 
           + a[ja1+2]*x[ja[ja1+2]] 
           + a[ja1+3]*x[ja[ja1+3]] 
           + a[ja1+4]*x[ja[ja1+4]] 
           + a[ja1+5]*x[ja[ja1+5]];

    y[i] = tmp;
  }
} 
/*********************************************************************/ 
 
/********************************************************************* 
 * MATVECCSRDO2:  produto matriz vetor para uma matriz generica no   *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDO2(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ja1;
  int resto;
  double tmp1,tmp2;

  resto = neq%2;
  
  if(resto){
    tmp1  = ad[0]*x[0];
    for(ja1=0;ja1<ia[1];ja1++)
      tmp1 += a[ja1]*x[ja[ja1]];
    y[0] = tmp1;
  }
  
  for(i=resto;i<neq;i+=2){
    tmp1  = ad[i  ]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    if(ia1 == ia2) goto linhai1;

/*linha i*/
    for(ja1=ia1;ja1<ia2;ja1++)
      tmp1 += a[ja1]*x[ja[ja1]];
/*...................................................................*/

/*linha i+1*/
linhai1:
    y[i] = tmp1;
    if(ia2 == ia3) goto linhai2;

    for(ja1=ia2;ja1<ia3;ja1++)
      tmp2 += a[ja1]*x[ja[ja1]];
/*...................................................................*/
linhai2:
    y[i+1] = tmp2;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCRSDO4:  produto matriz vetor para uma matriz generica no   *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDO4(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i,j;
  INT ia1,ia2,ia3,ia4,ia5;
  int resto;
  double tmp1,tmp2,tmp3,tmp4;

  resto = neq %4;
  
  if(resto == 3){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    tmp3  = ad[2]*x[2];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    for(j=ia[2];j<ia[3];j++)
      tmp3 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
    y[2] = tmp3;
  }

  else if(resto == 2){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
  }
  
  else if(resto == 1){
    tmp1  = ad[0]*x[0];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    y[0] = tmp1;
  }
  
  for(i=resto;i<neq;i+=4){
    tmp1  = ad[i  ]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    tmp3  = ad[i+2]*x[i+2];
    tmp4  = ad[i+3]*x[i+3];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    ia4   = ia[i+3];
    ia5   = ia[i+4];
    if(ia1 == ia2) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
/*...................................................................*/

/*linha i+1*/
linhai1:
    y[i] = tmp1;
    if(ia2 == ia3) goto linhai2;

    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
/*...................................................................*/

/*linha i+2*/
linhai2:
    y[i+1] = tmp2;
    if(ia4 == ia3) goto linhai3;

    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
/*...................................................................*/

/*linha i+3*/
linhai3:
    y[i+2] = tmp3;
    if(ia5 == ia4) goto linhai4;

    for(j=ia4;j<ia5;j++)
      tmp4 += a[j]*x[ja[j]];
/*...................................................................*/
linhai4:
    y[i+3] = tmp4;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCRSDO6:  produto matriz vetor para uma matriz generica no   *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDO6(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i,j;
  INT ia1,ia2,ia3,ia4,ia5,ia6,ia7;
  int resto;
  double tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;

  resto = neq%6;
  
  if(resto == 5){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    tmp3  = ad[2]*x[2];
    tmp4  = ad[3]*x[3];
    tmp5  = ad[4]*x[4];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    for(j=ia[2];j<ia[3];j++)
      tmp3 += a[j]*x[ja[j]];
    for(j=ia[3];j<ia[4];j++)
      tmp4 += a[j]*x[ja[j]];
    for(j=ia[4];j<ia[5];j++)
      tmp5 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
    y[2] = tmp3;
    y[3] = tmp4;
    y[4] = tmp5;
  }
  if(resto == 4){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    tmp3  = ad[2]*x[2];
    tmp4  = ad[3]*x[3];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    for(j=ia[2];j<ia[3];j++)
      tmp3 += a[j]*x[ja[j]];
    for(j=ia[3];j<ia[4];j++)
      tmp4 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
    y[2] = tmp3;
    y[3] = tmp4;
  }
  if(resto == 3){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    tmp3  = ad[2]*x[2];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    for(j=ia[2];j<ia[3];j++)
      tmp3 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
    y[2] = tmp3;
  }
  
  else if(resto == 2){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
  }
  
  else if(resto == 1){
    tmp1  = ad[0]*x[0];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    y[0] = tmp1;
  }
  
for(i=resto;i<neq;i+=6){
    tmp1  = ad[i  ]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    tmp3  = ad[i+2]*x[i+2];
    tmp4  = ad[i+3]*x[i+3];
    tmp5  = ad[i+4]*x[i+4];
    tmp6  = ad[i+5]*x[i+5];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    ia4   = ia[i+3];
    ia5   = ia[i+4];
    ia6   = ia[i+5];
    ia7   = ia[i+6];
    if(ia1 == ia2) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
/*...................................................................*/

linhai1:
    y[i] = tmp1;

/*linha i+1*/
    if(ia2 == ia3) goto linhai2;

    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
/*...................................................................*/

linhai2:
    y[i+1] = tmp2;
/*linha i+2*/
    if(ia4 == ia3) goto linhai3;

    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
/*...................................................................*/

linhai3:
    y[i+2] = tmp3;
/*linha i+3*/
    if(ia5 == ia4) goto linhai4;

    for(j=ia4;j<ia5;j++)
      tmp4 += a[j]*x[ja[j]];
/*...................................................................*/

linhai4:
    y[i+3] = tmp4;
/*linha i+4*/
    if(ia6 == ia5) goto linhai5;

    for(j=ia5;j<ia6;j++)
      tmp5 += a[j]*x[ja[j]];
/*...................................................................*/

linhai5:
    y[i+4] = tmp5;
/*linha i+5*/
    if(ia7 == ia6) goto linhai6;

    for(j=ia6;j<ia7;j++)
      tmp6 += a[j]*x[ja[j]];
/*...................................................................*/
linhai6:
    y[i+5] = tmp6;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCRSDO2I2: produto matriz vetor para uma matriz generica no  *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDO2I2(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i,j;
  INT ia1,ia2,ia3;
  int resto1,resto2,n;
  double tmp1,tmp2;
  
  resto1 = neq % 2;
  
  if(resto1){
    tmp1  = ad[0]*x[0];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    y[0] = tmp1;
  }
  
  for(i=resto1;i<neq;i+=2){
    tmp1  = ad[i  ]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    if(ia1 == ia2) goto linhai1;
    n      = ia2 - ia1;
    resto2 = n%2;
    if(resto2)
      tmp1 += a[ia1]*x[ja[ia1]];

/*linha i*/
    for(j=ia1+resto2;j<ia2;j+=2)
      tmp1 +=a[  j]*x[ja[  j]] 
           + a[j+1]*x[ja[j+1]];
/*...................................................................*/

linhai1:
    y[i] = tmp1;
/*linha i+1*/
    if(ia2 == ia3) goto linhai2;
    n      = ia3 - ia2;
    resto2 = n%2;
    if(resto2)
      tmp2 += a[ia2]*x[ja[ia2]];

    for(j=ia2+resto2;j<ia3;j+=2)
      tmp2 += a[  j]*x[  ja[j]]
            + a[j+1]*x[ja[j+1]];
/*...................................................................*/
linhai2:
    y[i+1] = tmp2;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRC  : produto matriz vetor para uma matriz no formato csrc* 
 * (y=Ax, A uma matriz estruturalmente simentrica)                   * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: ja guarda os indiceis da para inferior da matriz             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrC(INT neq           
              ,INT *restrict ia   ,INT *restrict ja
              ,double *restrict au,double *restrict ad
              ,double *restrict al
              ,double *restrict x ,double *restrict y)
{
  INT i,k,jak;
  double xi,tmp;
 
  y[0] = ad[0]*x[0]; 
  for(i=1;i<neq;i++){
    xi  = x[i];
    tmp = ad[i]*xi;
    for(k=ia[i];k<ia[i+1];k++){
      jak = ja[k];
/*... produto da linha i pelo vetor x*/
      tmp    += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[k]*xi;
       
    }
    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCI2: produto matriz vetor para uma matriz no formato csrc* 
 * (y=Ax, A uma matriz estruturalmente simentrica)                   * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: ja guarda os indiceis da para inferior da matriz             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCI2(INT neq           
              ,INT *restrict ia   ,INT *restrict ja
              ,double *restrict au,double *restrict ad
              ,double *restrict al
              ,double *restrict x ,double *restrict y)
{
  INT i,ia1,ia2,ja1,jak1,jak2;
  int resto,n;
  double xi,tmp;
 
  y[0] = ad[0]*x[0]; 
  for(i=1;i<neq;i++){
    xi  = x[i];
    tmp = ad[i]*xi;
    ia1 = ia[  i];
    ia2 = ia[i+1];

    n     = ia2 - ia1;
    resto = n%2;
    
    if(resto){ 
      jak1     = ja[ia1];
      tmp     += al[ia1]*x[jak1];
      y[jak1] += au[ia1]*xi;
    }

    for(ja1=ia1+resto;ja1<ia2;ja1+=2){
      jak1 = ja[  ja1];
      jak2 = ja[ja1+1];
/*... produto da linha i pelo vetor x*/
      tmp     += al[ja1]*x[jak1] + al[ja1+1]*x[jak2];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak1] += au[  ja1]*xi;
      y[jak2] += au[ja1+1]*xi;
    }
    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCI4: produto matriz vetor para uma matriz no formato csrc* 
 * (y=Ax, A uma matriz estruturalmente simentrica)                   * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: ja guarda os indiceis da para inferior da matriz             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCI4(INT neq           
              ,INT *restrict ia   ,INT *restrict ja
              ,double *restrict au,double *restrict ad
              ,double *restrict al
              ,double *restrict x ,double *restrict y)
{
  INT i,ia1,ia2,ja1,jak1,jak2,jak3,jak4;
  int resto,n;
  double xi,tmp;
 
  y[0] = ad[0]*x[0]; 
  for(i=1;i<neq;i++){
    xi  = x[i];
    tmp = ad[i]*xi;
    ia1 = ia[  i];
    ia2 = ia[i+1];

    n     = ia2 - ia1;
    resto = n%4;
    
    if(resto == 3){ 
      jak1     = ja[  ia1];
      jak2     = ja[ia1+1];
      jak3     = ja[ia1+2];
      tmp     += al[ia1]*x[jak1] + al[ia1+1]*x[jak2] + al[ia1+2]*x[jak3];
      y[jak1] += au[  ia1]*xi;
      y[jak2] += au[ia1+1]*xi;
      y[jak3] += au[ia1+2]*xi;
    }
    else if(resto == 2){
      jak1     = ja[  ia1];
      jak2     = ja[ia1+1];
      tmp     += al[ia1]*x[jak1] + al[ia1+1]*x[jak2];
      y[jak1] += au[  ia1]*xi;
      y[jak2] += au[ia1+1]*xi;
    }
    else if(resto == 1){
      jak1     = ja[ia1];
      tmp     += al[ia1]*x[jak1];
      y[jak1] += au[ia1]*xi;

    }

    for(ja1=ia1+resto;ja1<ia2;ja1+=4){
      jak1 = ja[  ja1];
      jak2 = ja[ja1+1];
      jak3 = ja[ja1+2];
      jak4 = ja[ja1+3];
/*... produto da linha i pelo vetor x*/
      tmp     += al[  ja1]*x[jak1] + al[ja1+1]*x[jak2] 
               + al[ja1+2]*x[jak3] + al[ja1+3]*x[jak4];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak1] += au[  ja1]*xi;
      y[jak2] += au[ja1+1]*xi;
      y[jak3] += au[ja1+2]*xi;
      y[jak4] += au[ja1+3]*xi;
    }
    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCI6: produto matriz vetor para uma matriz no formato csrc* 
 * (y=Ax, A uma matriz estruturalmente simentrica)                   * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: ja guarda os indiceis da para inferior da matriz             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCI6(INT neq           
              ,INT *restrict ia   ,INT *restrict ja
              ,double *restrict au,double *restrict ad
              ,double *restrict al
              ,double *restrict x ,double *restrict y)
{
  INT i,ia1,ia2,ja1,jak1,jak2,jak3,jak4,jak5,jak6;
  int resto,n;
  double xi,tmp;
 
  y[0] = ad[0]*x[0]; 
  for(i=1;i<neq;i++){
    xi  = x[i];
    tmp = ad[i]*xi;
    ia1 = ia[  i];
    ia2 = ia[i+1];

    n     = ia2 - ia1;
    resto = n%6;
    
    if(resto == 5){ 
      jak1     = ja[  ia1];
      jak2     = ja[ia1+1];
      jak3     = ja[ia1+2];
      jak4     = ja[ia1+3];
      jak5     = ja[ia1+4];
      tmp     += al[  ia1]*x[jak1] + al[ia1+1]*x[jak2] + al[ia1+2]*x[jak3] 
              +  al[ia1+3]*x[jak4] + al[ia1+4]*x[jak5];
      y[jak1] += au[  ia1]*xi;
      y[jak2] += au[ia1+1]*xi;
      y[jak3] += au[ia1+2]*xi;
      y[jak4] += au[ia1+3]*xi;
      y[jak5] += au[ia1+4]*xi;
    }
    else if(resto == 4){ 
      jak1     = ja[  ia1];
      jak2     = ja[ia1+1];
      jak3     = ja[ia1+2];
      jak4     = ja[ia1+3];
      tmp     += al[  ia1]*x[jak1] + al[ia1+1]*x[jak2] + al[ia1+2]*x[jak3] 
               + al[ia1+3]*x[jak4];
      y[jak1] += au[  ia1]*xi;
      y[jak2] += au[ia1+1]*xi;
      y[jak3] += au[ia1+2]*xi;
      y[jak4] += au[ia1+3]*xi;
    }
    else if(resto == 3){ 
      jak1     = ja[  ia1];
      jak2     = ja[ia1+1];
      jak3     = ja[ia1+2];
      tmp     += al[ia1]*x[jak1] + al[ia1+1]*x[jak2] + al[ia1+2]*x[jak3];
      y[jak1] += au[  ia1]*xi;
      y[jak2] += au[ia1+1]*xi;
      y[jak3] += au[ia1+2]*xi;
    }
    else if(resto == 2){
      jak1     = ja[  ia1];
      jak2     = ja[ia1+1];
      tmp     += al[ia1]*x[jak1] + al[ia1+1]*x[jak2];
      y[jak1] += au[  ia1]*xi;
      y[jak2] += au[ia1+1]*xi;
    }
    else if(resto == 1){
      jak1     = ja[ia1];
      tmp     += al[ia1]*x[jak1];
      y[jak1] += au[ia1]*xi;

    }

    for(ja1=ia1+resto;ja1<ia2;ja1+=6){
      jak1 = ja[  ja1];
      jak2 = ja[ja1+1];
      jak3 = ja[ja1+2];
      jak4 = ja[ja1+3];
      jak5 = ja[ja1+4];
      jak6 = ja[ja1+5];
/*... produto da linha i pelo vetor x*/
      tmp     += al[  ja1]*x[jak1] + al[ja1+1]*x[jak2] 
               + al[ja1+2]*x[jak3] + al[ja1+3]*x[jak4] 
               + al[ja1+4]*x[jak5] + al[ja1+5]*x[jak6];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak1] += au[  ja1]*xi;
      y[jak2] += au[ja1+1]*xi;
      y[jak3] += au[ja1+2]*xi;
      y[jak4] += au[ja1+3]*xi;
      y[jak5] += au[ja1+4]*xi;
      y[jak6] += au[ja1+5]*xi;
    }
    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCO2: produto matriz vetor para uma matriz no formato csrc* 
 * (y=Ax, A uma matriz estruturalmente simentrica)                   * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: ja guarda os indiceis da para inferior da matriz             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCO2(INT neq           
                 ,INT *restrict ia   ,INT *restrict ja
                 ,double *restrict au,double *restrict ad
                 ,double *restrict al
                 ,double *restrict x ,double *restrict y)
{
  INT i,j,jak;
  INT ia1,ia2,ia3;
  int resto;
  double xi1,xi2,tmp1,tmp2;
 
  resto = neq%2;
  if(resto){
    y[0] = ad[0]*x[0]; 
  }
  for(i=resto;i<neq;i+=2){
    xi1  = x[  i];
    xi2  = x[i+1];
    tmp1 = ad[  i]*xi1;
    tmp2 = ad[i+1]*xi2;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    ia3  = ia[i+2];

    if(ia2 == ia1) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp1   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi1;
    }
/*...................................................................*/

linhai1:
    y[i] = tmp1;

    if(ia3 == ia2) goto linhai2;
/*linha i+1*/
    for(j=ia2;j<ia3;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp2    += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi2;
    }
/*...................................................................*/

linhai2:
    y[i+1] = tmp2;
  }
/*...................................................................*/
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCO4: produto matriz vetor para uma matriz no formato csrc* 
 * (y=Ax, A uma matriz estruturalmente simentrica)                   * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: ja guarda os indiceis da para inferior da matriz             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCO4(INT neq           
                 ,INT *restrict ia   ,INT *restrict ja
                 ,double *restrict au,double *restrict ad
                 ,double *restrict al
                 ,double *restrict x ,double *restrict y)
{
  INT i,j;
  INT ia1,ia2,ia3,ia4,ia5,jak;
  int resto;
  double xi1,xi2,xi3,xi4,tmp1,tmp2,tmp3,tmp4;
 
  resto = neq%4;

  if(resto){
    for(i=0;i<resto;i++){
      xi1  = x[i];
      tmp1 = ad[i]*xi1;
      for(j=ia[i];j<ia[i+1];j++){
        jak     = ja[j];
        tmp1   += al[j]*x[jak];
        y[jak] += au[j]*xi1;
      }
      y[i] = tmp1;
    }
  }

  for(i=resto;i<neq;i+=4){
    xi1  = x[  i];
    xi2  = x[i+1];
    xi3  = x[i+2];
    xi4  = x[i+3];
    tmp1 = ad[  i]*xi1;
    tmp2 = ad[i+1]*xi2;
    tmp3 = ad[i+2]*xi3;
    tmp4 = ad[i+3]*xi4;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    ia3  = ia[i+2];
    ia4  = ia[i+3];
    ia5  = ia[i+4];
    if(ia1 == ia2) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp1   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi1;
    }
/*...................................................................*/

linhai1:
    y[i] = tmp1;

    if(ia3 == ia2) goto linhai2;
/*linha i+1*/
    for(j=ia2;j<ia3;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp2   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi2;
    }
/*...................................................................*/

linhai2:
    y[i+1] = tmp2;

    if(ia4 == ia3) goto linhai3;
/*linha i+2*/
    for(j=ia3;j<ia4;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp3   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi3;
    }
/*...................................................................*/

linhai3:
    y[i+2] = tmp3;

    if(ia5 == ia4) goto linhai4;
/*linha i+3*/
    for(j=ia4;j<ia5;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp4   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi4;
    }
/*...................................................................*/

linhai4:
    y[i+3] = tmp4;
  }
/*...................................................................*/
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCO6: produto matriz vetor para uma matriz no formato csrc* 
 * (y=Ax, A uma matriz estruturalmente simentrica)                   * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: ja guarda os indiceis da para inferior da matriz             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCO6(INT neq           
                 ,INT *restrict ia   ,INT *restrict ja
                 ,double *restrict au,double *restrict ad
                 ,double *restrict al
                 ,double *restrict x ,double *restrict y)
{
  INT i,j;
  INT ia1,ia2,ia3,ia4,ia5,ia6,ia7,jak;
  int resto;
  double xi1,xi2,xi3,xi4,xi5,xi6,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
 
  resto = neq%6;

  if(resto){
    for(i=0;i<resto;i++){
      xi1  = x[i];
      tmp1 = ad[i]*xi1;
      for(j=ia[i];j<ia[i+1];j++){
        jak     = ja[j];
        tmp1   += al[j]*x[jak];
        y[jak] += au[j]*xi1;
      }
      y[i] = tmp1;
    }
  }

  for(i=resto;i<neq;i+=6){
    xi1  = x[  i];
    xi2  = x[i+1];
    xi3  = x[i+2];
    xi4  = x[i+3];
    xi5  = x[i+4];
    xi6  = x[i+5];
    tmp1 = ad[  i]*xi1;
    tmp2 = ad[i+1]*xi2;
    tmp3 = ad[i+2]*xi3;
    tmp4 = ad[i+3]*xi4;
    tmp5 = ad[i+4]*xi5;
    tmp6 = ad[i+5]*xi6;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    ia3  = ia[i+2];
    ia4  = ia[i+3];
    ia5  = ia[i+4];
    ia6  = ia[i+5];
    ia7  = ia[i+6];
    if(ia1 == ia2) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp1   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi1;
    }
/*...................................................................*/

linhai1:
    y[i] = tmp1;

    if(ia3 == ia2) goto linhai2;
/*linha i+1*/
    for(j=ia2;j<ia3;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp2   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi2;
    }
/*...................................................................*/

linhai2:
    y[i+1] = tmp2;

    if(ia4 == ia3) goto linhai3;
/*linha i+2*/
    for(j=ia3;j<ia4;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp3   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi3;
    }
/*...................................................................*/

linhai3:
    y[i+2] = tmp3;

    if(ia5 == ia4) goto linhai4;
/*linha i+3*/
    for(j=ia4;j<ia5;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp4   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi4;
    }
/*...................................................................*/

linhai4:
    y[i+3] = tmp4;
    
    if(ia6 == ia5) goto linhai5;
/*linha i + 4*/
    for(j=ia5;j<ia6;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp5   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi5;
    }
/*...................................................................*/

linhai5:
    y[i+4] = tmp5;
    
    if(ia7 == ia6) goto linhai6;
/*linha i + 5*/
    for(j=ia6;j<ia7;j++){
      jak = ja[j];
/*... produto da linha i pelo vetor x*/
      tmp6   += al[j]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak] += au[j]*xi6;
    }
/*...................................................................*/

linhai6:
    y[i+5] = tmp6;
  }
/*...................................................................*/
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCO2I2: produto matriz vetor para uma matriz no formato   *
 * csrc(y=Ax, A uma matriz estruturalmente simentrica)               * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: ja guarda os indiceis da para inferior da matriz             * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCO2I2(INT neq           
                 ,INT *restrict ia   ,INT *restrict ja
                 ,double *restrict au,double *restrict ad
                 ,double *restrict al
                 ,double *restrict x ,double *restrict y)
{
  INT i,j;
  INT ia1,ia2,ia3,jak1,jak2;
  int resto1,resto2,n;
  double xi1,xi2,tmp1,tmp2;
 
  resto1 = neq%2;
  if(resto1){
    y[0] = ad[0]*x[0]; 
  }
  for(i=resto1;i<neq;i+=2){
    xi1  = x[  i];
    xi2  = x[i+1];
    tmp1 = ad[  i]*xi1;
    tmp2 = ad[i+1]*xi2;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    ia3  = ia[i+2];
    n   = ia2-ia1;
    if(!n) goto linhai1;
    resto2 = n%2;
    
/*linha i*/
    if(resto2){ 
      jak1     = ja[ia1];
      tmp1    += al[ia1]*x[jak1];
      y[jak1] += au[ia1]*xi1;
    }

    for(j=ia1+resto2;j<ia2;j+=2){
      jak1 = ja[  j];
      jak2 = ja[j+1];
/*... produto da linha i pelo vetor x*/
      tmp1    += al[j]*x[jak1] + al[j+1]*x[jak2];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak1] += au[  j]*xi1;
      y[jak2] += au[j+1]*xi1;
    }
/*...................................................................*/

linhai1:
    y[i] = tmp1;
    
    n   = ia3-ia2;
    if(!n) goto linhai2;
    resto2 = n%2;

/*linha i+1*/
    if(resto2){ 
      jak1     = ja[ia2];
      tmp2    += al[ia2]*x[jak1];
      y[jak1] += au[ia2]*xi2;
    }

    for(j=ia2+resto2;j<ia3;j+=2){
      jak1 = ja[  j];
      jak2 = ja[j+1];
/*... produto da linha i pelo vetor x*/
      tmp2    += al[j]*x[jak1] + al[j+1]*x[jak2];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      y[jak1] += au[  j]*xi2;
      y[jak2] += au[j+1]*xi2;
    }
/*...................................................................*/

linhai2:
    y[i+1] = tmp2;
  }
/*...................................................................*/
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSR:produto matriz vetor para uma matriz generica no formato*
 * csr padrao (y=Ax, A uma matriz geral)                             * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsr(INT neq           
              ,INT *restrict ia  ,INT *restrict ja
              ,double *restrict a,double *restrict x
              ,double *restrict y)
{
  INT i,j;
  for(i=0;i<neq;i++){
    y[i] = 0.0;
    for(j=ia[i];j<ia[i+1];j++)
      y[i] += a[j]*x[ja[j]];
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRI2:produto matriz vetor para uma matriz generica no      *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrI2(INT neq           
                ,INT *restrict ia  ,INT *restrict ja
                ,double *restrict a,double *restrict x
                ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ja1;
  int resto,n;
  
  for(i=0;i<neq;i++){
    y[i] = 0.0;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    n    = ia2 - ia1;
    resto = n%2;
    if(resto)
      y[i] = a[ia1]*x[ja[ia1]]; 
    
    for(ja1=ia1+resto;ja1<ia2;ja1+=2){
      y[i] += a[  ja1]*x[  ja[ja1]] 
            + a[ja1+1]*x[ja[ja1+1]];
    }
/*...................................................................*/
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRI4:produto matriz vetor para uma matriz generica no      *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrI4(INT neq           
                ,INT *restrict ia  ,INT *restrict ja
                ,double *restrict a,double *restrict x
                ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ja1;
  int resto,n;
  
  for(i=0;i<neq;i++){
    y[i] = 0.0;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    n    = ia2 - ia1;
    resto = n%4;
    
    if(resto == 3)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      y[i] = a[ia1]*x[ja[ia1]]; 
    
    for(ja1=ia1+resto;ja1<ia2;ja1+=4){
      y[i] += a[  ja1]*x[  ja[ja1]] 
            + a[ja1+1]*x[ja[ja1+1]] 
            + a[ja1+2]*x[ja[ja1+2]] 
            + a[ja1+3]*x[ja[ja1+3]];
    }
/*...................................................................*/
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRI6:produto matriz vetor para uma matriz generica no      *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrI6(INT neq           
                ,INT *restrict ia  ,INT *restrict ja
                ,double *restrict a,double *restrict x
                ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ja1;
  int resto,n;
  
  for(i=0;i<neq;i++){
    y[i] = 0.0e0;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    n    = ia2 - ia1;
    resto = n%6;
    if(resto == 5)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]  
           + a[ia1+4]*x[ja[ia1+4]]; 
    else if(resto == 4)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]; 
    else if(resto == 3)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      y[i] = a[ia1]*x[ja[ia1]]; 
  
    for(ja1=ia1+resto;ja1<ia2;ja1+=6){
      y[i] += a[  ja1]*x[  ja[ja1]] 
            + a[ja1+1]*x[ja[ja1+1]] 
            + a[ja1+2]*x[ja[ja1+2]] 
            + a[ja1+3]*x[ja[ja1+3]] 
            + a[ja1+4]*x[ja[ja1+4]] 
            + a[ja1+5]*x[ja[ja1+5]];
    }
/*...................................................................*/
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRO2:produto matriz vetor para uma matriz generica no      *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrO2(INT neq           
                ,INT *restrict ia  ,INT *restrict ja
                ,double *restrict a,double *restrict x
                ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ja1,ja2;
  int resto;
  
  resto = neq % 2;
  
  if(resto){
    y[0] = 0.0e0; 
    for(ja1=0;ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
  }

  for(i=resto;i<neq;i+=2){
    y[i]   = 0.0e0;
    y[i+1] = 0.0e0;
    ia1    = ia[  i];
    ia2    = ia[i+1];
    ia3    = ia[i+2];
    if(ia1 == ia2) goto linhai1;
/*linha i*/
    for(ja1=ia1;ja1<ia2;ja1++)
      y[i] += a[ja1]*x[ja[ja1]];
/*...................................................................*/

/*linha i+1*/
linhai1:
    if(ia2 == ia3) goto linhai2;
    
    for(ja2=ia2;ja2<ia3;ja2++)
      y[i+1] += a[ja2]*x[ja[ja2]];
/*...................................................................*/
linhai2:
    continue;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRO4:produto matriz vetor para uma matriz generica no      *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrO4(INT neq           
                ,INT *restrict ia  ,INT *restrict ja
                ,double *restrict a,double *restrict x
                ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ia4,ia5,ja1,ja2,ja3,ja4;
  int resto;
  
  resto = neq % 4;
  
  if(resto == 3){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    y[2] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[2];ja1<ia[3];ja1++)
      y[2] += a[ja1]*x[ja[ja1]];
  }
  
  else if(resto == 2){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]]; 
  }
  
  else if(resto == 1){
    y[0] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
  }
  

  for(i=resto;i<neq;i+=4){
    y[i]   = 0.0e0;
    y[i+1] = 0.0e0;
    y[i+2] = 0.0e0;
    y[i+3] = 0.0e0;
    ia1    = ia[  i];
    ia2    = ia[i+1];
    ia3    = ia[i+2];
    ia4    = ia[i+3];
    ia5    = ia[i+4];
    if(ia2 == ia1) goto linhai1;
/*linha i*/
    for(ja1=ia1;ja1<ia2;ja1++)
      y[i] += a[ja1]*x[ja[ja1]];
/*...................................................................*/
linhai1:
    if(ia2 == ia3) goto linhai2;
/*linha i+1*/
    for(ja2=ia2;ja2<ia3;ja2++)
      y[i+1] += a[ja2]*x[ja[ja2]];
/*...................................................................*/
linhai2:
    if(ia4 == ia3) goto linhai3;
/*linha i+2*/
    for(ja3=ia3;ja3<ia4;ja3++)
      y[i+2] += a[ja3]*x[ja[ja3]];
/*...................................................................*/
linhai3:
    if(ia5 == ia4) goto linhai4;
/*linha i+3*/
    for(ja4=ia4;ja4<ia5;ja4++)
      y[i+3] += a[ja4]*x[ja[ja4]];
/*...................................................................*/
linhai4:
    continue;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRO6:produto matriz vetor para uma matriz generica no      *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrO6(INT neq           
                ,INT *restrict ia  ,INT *restrict ja
                ,double *restrict a,double *restrict x
                ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ia4,ia5,ia6,ia7,ja1,ja2,ja3,ja4,ja5,ja6;
  int resto;
  
  resto = neq % 6;
  
  if(resto == 5){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    y[2] = 0.0e0; 
    y[3] = 0.0e0; 
    y[4] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[2];ja1<ia[3];ja1++)
      y[2] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[3];ja1<ia[4];ja1++)
      y[3] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[4];ja1<ia[5];ja1++)
      y[4] += a[ja1]*x[ja[ja1]];
  }
  
  else if(resto == 4){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    y[2] = 0.0e0; 
    y[3] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[2];ja1<ia[3];ja1++)
      y[2] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[3];ja1<ia[4];ja1++)
      y[3] += a[ja1]*x[ja[ja1]];
  }

  else if(resto == 3){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    y[2] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[2];ja1<ia[3];ja1++)
      y[2] += a[ja1]*x[ja[ja1]];
  }
  
  else if(resto == 2){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]]; 
  }
  
  else if(resto == 1){
    y[0] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
  }
  

  for(i=resto;i<neq;i+=6){
    y[i]   = 0.0e0;
    y[i+1] = 0.0e0;
    y[i+2] = 0.0e0;
    y[i+3] = 0.0e0;
    y[i+4] = 0.0e0;
    y[i+5] = 0.0e0;
    ia1    = ia[  i];
    ia2    = ia[i+1];
    ia3    = ia[i+2];
    ia4    = ia[i+3];
    ia5    = ia[i+4];
    ia6    = ia[i+5];
    ia7    = ia[i+6];
    if(ia2 == ia1) goto linhai1;
/*linha i*/
    for(ja1=ia1;ja1<ia2;ja1++)
      y[i] += a[ja1]*x[ja[ja1]];
/*...................................................................*/
linhai1:
    if(ia2 == ia3) goto linhai2;
/*linha i+1*/
    for(ja2=ia2;ja2<ia3;ja2++)
      y[i+1] += a[ja2]*x[ja[ja2]];
/*...................................................................*/
linhai2:
    if(ia4 == ia3) goto linhai3;
/*linha i+2*/
    for(ja3=ia3;ja3<ia4;ja3++)
      y[i+2] += a[ja3]*x[ja[ja3]];
/*...................................................................*/
linhai3:
    if(ia5 == ia4) goto linhai4;
/*linha i+3*/
    for(ja4=ia4;ja4<ia5;ja4++)
      y[i+3] += a[ja4]*x[ja[ja4]];
/*...................................................................*/
linhai4:
    if(ia6 == ia5) goto linhai5;
/*linha i+4*/
    for(ja5=ia5;ja5<ia6;ja5++)
      y[i+4] += a[ja5]*x[ja[ja5]];
/*...................................................................*/
linhai5:
    if(ia7 == ia6) goto linhai6;
/*linha i+5*/
    for(ja6=ia6;ja6<ia7;ja6++)
      y[i+5] += a[ja6]*x[ja[ja6]];
/*...................................................................*/
linhai6:
    continue;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRO2I2:produto matriz vetor para uma matriz generica no    *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrO2I2(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict x
                   ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ja1,ja2;
  int resto1,resto2,resto3,n;
  
  resto1 = neq % 2;
  
  if(resto1){
    y[0] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
  }
  
  for(i=resto1;i<neq;i+=2){
    y[i]   = 0.0e0;
    y[i+1] = 0.0e0;
    ia1    = ia[  i];
    ia2    = ia[i+1];
    ia3    = ia[i+2];
    n      = ia2 - ia1;
    if(!n) goto linhai1;
    resto2 = n%2;
/*linha i*/
    if(resto2){
      y[i] = a[ia1]*x[ja[ia1]];
    }
    for(ja1=ia1+resto2;ja1<ia2;ja1+=2)
      y[i] += a[  ja1]*x[ja[ja1  ]]
            + a[ja1+1]*x[ja[ja1+1]];
/*...................................................................*/
linhai1:    
/*linha i+1*/
    n     = ia3 - ia2;
    if(!n) goto linhai2;
    resto3 = n%2;
    if(resto3){
      y[i+1] = a[ia2]*x[ja[ia2]];
    }
    for(ja2=ia2+resto3;ja2<ia3;ja2+=2)
      y[i+1] += a[ja2  ]*x[ja[ja2  ]]
              + a[ja2+1]*x[ja[ja2+1]];
/*...................................................................*/
linhai2:
    continue;    
  }
} 
/*********************************************************************/ 
#ifdef _OPENMP 
/********************************************************************* 
 * DOTOMP: produto interno entre dois vetores                           * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmp(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  #pragma omp single
    tmpDotOmp=0.0e0;

  #pragma omp for private(i) reduction(+:tmpDotOmp)
    for(i=0;i<n;i++){
      tmpDotOmp += x1[i]*x2[i];
    }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPO2L2: produto interno entre dois vetores                    * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpO2L2(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%4;
  #pragma omp single
  {
    tmpDotOmp1=0.0e0;
    tmpDotOmp2=0.0e0;
    if(resto==1)
      tmpDotOmp1= x1[0]*x2[0];
    if(resto==2)
      tmpDotOmp1= x1[0]*x2[0] + x1[1]*x2[1];
    if(resto==3)
      tmpDotOmp1= x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2];
  }

  #pragma omp for private(i) reduction(+:tmpDotOmp1,tmpDotOmp2)
    for(i=resto;i<n;i+=4){
      tmpDotOmp1 += x1[i  ]*x2[i  ] + x1[i+1]*x2[i+1]; 
      tmpDotOmp2 += x1[i+2]*x2[i+2] + x1[i+3]*x2[i+3];
    }
  
  #pragma omp single
  {
    tmpDotOmp=tmpDotOmp1+tmpDotOmp2;
  }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPO2L4: produto interno entre dois vetores                    * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpO2L4(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%8;

  #pragma omp single
  {
    tmpDotOmp1=0.0e0;
    tmpDotOmp2=0.0e0;
    tmpDotOmp3=0.0e0;
    tmpDotOmp4=0.0e0;
    tmpDotOmp5=0.0e0;
    tmpDotOmp6=0.0e0;
    tmpDotOmp7=0.0e0;
    tmpDotOmp8=0.0e0;
    if(resto==1)
      tmpDotOmp1 = x1[0]*x2[0];
    else if(resto==2)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1];
    else if(resto==3)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2];
    else if(resto==4)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3];
    else if(resto==5)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3] + x1[4]*x2[4];
    else if(resto==6)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3] + x1[4]*x2[4] + x1[5]*x2[5];
    else if(resto==7)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3] + x1[4]*x2[4] + x1[5]*x2[5]
                 + x1[6]*x2[6];
  }

  #pragma omp for private(i)\
  reduction(+:tmpDotOmp1,tmpDotOmp2\
             ,tmpDotOmp3,tmpDotOmp4)
    for(i=resto;i<n;i+=8){
      tmpDotOmp1 += x1[i  ]*x2[i  ] + x1[i+1]*x2[i+1];
      tmpDotOmp2 += x1[i+2]*x2[i+2] + x1[i+3]*x2[i+3];
      tmpDotOmp3 += x1[i+4]*x2[i+4] + x1[i+5]*x2[i+5];
      tmpDotOmp4 += x1[i+6]*x2[i+6] + x1[i+7]*x2[i+7];
    }
  
  #pragma omp single
  {
    tmpDotOmp = tmpDotOmp1 + tmpDotOmp2 
              + tmpDotOmp3 + tmpDotOmp4;
  }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPL2: produto interno entre dois vetores                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpL2(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%2;
  #pragma omp single
  {
    tmpDotOmp1=0.0e0;
    tmpDotOmp2=0.0e0;
    if(resto)
      tmpDotOmp1= x1[0]*x2[0];
  }

  #pragma omp for private(i) reduction(+:tmpDotOmp1,tmpDotOmp2)
    for(i=resto;i<n;i+=2){
      tmpDotOmp1 += x1[i  ]*x2[i  ]; 
      tmpDotOmp2 += x1[i+1]*x2[i+1];
    }
  
  #pragma omp single
  {
    tmpDotOmp=tmpDotOmp1+tmpDotOmp2;
  }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPL4: produto interno entre dois vetores                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpL4(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%4;
  #pragma omp single
  {
    tmpDotOmp1=0.0e0;
    tmpDotOmp2=0.0e0;
    tmpDotOmp3=0.0e0;
    tmpDotOmp4=0.0e0;
    if(resto==1)
      tmpDotOmp1 = x1[0]*x2[0];
    else if(resto==2)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1];
    else if(resto==3)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2];
  }

  #pragma omp for private(i)\
   reduction(+:tmpDotOmp1,tmpDotOmp2,tmpDotOmp3,tmpDotOmp4)
    for(i=resto;i<n;i+=4){
      tmpDotOmp1 += x1[i  ]*x2[i  ]; 
      tmpDotOmp2 += x1[i+1]*x2[i+1];
      tmpDotOmp3 += x1[i+2]*x2[i+2];
      tmpDotOmp4 += x1[i+3]*x2[i+3];
    }
  
  #pragma omp single
  {
    tmpDotOmp=tmpDotOmp1+tmpDotOmp2+tmpDotOmp3+tmpDotOmp4;
  }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPL6: produto interno entre dois vetores                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpL6(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%6;
  #pragma omp single
  {
    tmpDotOmp1=0.0e0;
    tmpDotOmp2=0.0e0;
    tmpDotOmp3=0.0e0;
    tmpDotOmp4=0.0e0;
    tmpDotOmp5=0.0e0;
    tmpDotOmp6=0.0e0;
    if(resto==1)
      tmpDotOmp1 = x1[0]*x2[0];
    else if(resto==2)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1];
    else if(resto==3)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2];
    else if(resto==4)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3];
    else if(resto==5)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3] + x1[4]*x2[4];
  }

  #pragma omp for private(i)\
  reduction(+:tmpDotOmp1,tmpDotOmp2,tmpDotOmp3\
             ,tmpDotOmp4,tmpDotOmp5,tmpDotOmp6)
    for(i=resto;i<n;i+=6){
      tmpDotOmp1 += x1[i  ]*x2[i  ]; 
      tmpDotOmp2 += x1[i+1]*x2[i+1];
      tmpDotOmp3 += x1[i+2]*x2[i+2];
      tmpDotOmp4 += x1[i+3]*x2[i+3];
      tmpDotOmp5 += x1[i+4]*x2[i+4];
      tmpDotOmp6 += x1[i+5]*x2[i+5];
    }
  
  #pragma omp single
  {
    tmpDotOmp = tmpDotOmp1 + tmpDotOmp2
              + tmpDotOmp3 + tmpDotOmp4
              + tmpDotOmp5 + tmpDotOmp6;
  }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPL6: produto interno entre dois vetores                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpL8(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%8;
  #pragma omp single
  {
    tmpDotOmp1=0.0e0;
    tmpDotOmp2=0.0e0;
    tmpDotOmp3=0.0e0;
    tmpDotOmp4=0.0e0;
    tmpDotOmp5=0.0e0;
    tmpDotOmp6=0.0e0;
    tmpDotOmp7=0.0e0;
    tmpDotOmp8=0.0e0;
    if(resto==1)
      tmpDotOmp1 = x1[0]*x2[0];
    else if(resto==2)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1];
    else if(resto==3)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2];
    else if(resto==4)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3];
    else if(resto==5)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3] + x1[4]*x2[4];
    else if(resto==6)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3] + x1[4]*x2[4] + x1[5]*x2[5];
    else if(resto==7)
      tmpDotOmp1 = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2]
                 + x1[3]*x2[3] + x1[4]*x2[4] + x1[5]*x2[5]
                 + x1[6]*x2[6];
    }

  #pragma omp for private(i)\
   reduction(+:tmpDotOmp1,tmpDotOmp2,tmpDotOmp3\
              ,tmpDotOmp4,tmpDotOmp5,tmpDotOmp6\
              ,tmpDotOmp7,tmpDotOmp8)
    for(i=resto;i<n;i+=8){
      tmpDotOmp1 += x1[i  ]*x2[i  ]; 
      tmpDotOmp2 += x1[i+1]*x2[i+1];
      tmpDotOmp3 += x1[i+2]*x2[i+2];
      tmpDotOmp4 += x1[i+3]*x2[i+3];
      tmpDotOmp5 += x1[i+4]*x2[i+4];
      tmpDotOmp6 += x1[i+5]*x2[i+5];
      tmpDotOmp7 += x1[i+6]*x2[i+6];
      tmpDotOmp8 += x1[i+7]*x2[i+7];
    }
  
  #pragma omp single
  {
    tmpDotOmp = tmpDotOmp1 + tmpDotOmp2
              + tmpDotOmp3 + tmpDotOmp4
              + tmpDotOmp5 + tmpDotOmp6
              + tmpDotOmp7 + tmpDotOmp8;
  }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPO2: produto interno entre dois vetores                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpO2(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%2;
  #pragma omp single
  {
    tmpDotOmp=0.0e0;
    if(resto)
      tmpDotOmp = x1[0]*x2[0];
  }

  #pragma omp for private(i) reduction(+:tmpDotOmp)
    for(i=resto;i<n;i+=2){
      tmpDotOmp += x1[i]*x2[i] + x1[i+1]*x2[i+1];
    }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPO4: produto interno entre dois vetores                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpO4(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%4;
  #pragma omp single
  {
    tmpDotOmp=0.0e0;
    if(resto==1)
      tmpDotOmp = x1[0]*x2[0];
    else if(resto==2)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1];
    else if(resto==3)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2];
  }

  #pragma omp for private(i) reduction(+:tmpDotOmp)
    for(i=resto;i<n;i+=4){
      tmpDotOmp += x1[  i]*x2[  i] 
                 + x1[i+1]*x2[i+1] 
                 + x1[i+2]*x2[i+2]
                 + x1[i+3]*x2[i+3];
    }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPO6: produto interno entre dois vetores                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpO6(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%6;
  #pragma omp single
  {
    tmpDotOmp=0.0e0;
    if(resto==1)
      tmpDotOmp = x1[0]*x2[0];
    else if(resto==2)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1];
    else if(resto==3)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2];
    else if(resto==4)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] 
                + x1[3]*x2[3];
    else if(resto==5)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] 
                + x1[3]*x2[3] + x1[4]*x2[4];              
  }

  #pragma omp for private(i) reduction(+:tmpDotOmp)
    for(i=resto;i<n;i+=6){
      tmpDotOmp += x1[  i]*x2[  i] 
                 + x1[i+1]*x2[i+1] 
                 + x1[i+2]*x2[i+2]
                 + x1[i+3]*x2[i+3] 
                 + x1[i+4]*x2[i+4] 
                 + x1[i+5]*x2[i+5];
    }
  
  return tmpDotOmp;
} 
/*********************************************************************/ 

/********************************************************************* 
 * DOTOMPO8: produto interno entre dois vetores                      * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * x1 - vetor x1                                                     * 
 * x2 - vetor x2                                                     * 
 * n  - numero de dimensoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
double dotOmpO8(double *restrict x1,double *restrict x2,INT n)
{
  INT i;
  int resto = n%8;
  #pragma omp single
  {
    tmpDotOmp=0.0e0;
    if(resto==1)
      tmpDotOmp = x1[0]*x2[0];
    else if(resto==2)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1];
    else if(resto==3)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2];
    else if(resto==4)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] 
                + x1[3]*x2[3];
    else if(resto==5)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] 
                + x1[3]*x2[3] + x1[4]*x2[4];              
    else if(resto==6)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] 
                + x1[3]*x2[3] + x1[4]*x2[4] + x1[5]*x2[5];
    else if(resto==7)
      tmpDotOmp = x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2] 
                + x1[3]*x2[3] + x1[4]*x2[4] + x1[5]*x2[5]
                + x1[6]*x2[6];
  }

  #pragma omp for 
    for(i=resto;i<n;i+=8){
      tmpDotOmp += x1[  i]*x2[  i] 
                 + x1[i+1]*x2[i+1] 
                 + x1[i+2]*x2[i+2]
                 + x1[i+3]*x2[i+3] 
                 + x1[i+4]*x2[i+4] 
                 + x1[i+5]*x2[i+5] 
                 + x1[i+6]*x2[i+6] 
                 + x1[i+7]*x2[i+7];
    }
  
  return tmpDotOmp;
} 
/*********************************************************************/

/********************************************************************* 
 * MATVECCSRDOMP :produto matriz vetor para uma matriz generica no   *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmp(INT neq           
                  ,INT *restrict ia  ,INT *restrict ja
                  ,double *restrict a,double *restrict ad
                  ,double *restrict x,double *restrict y)
{
  INT i,j;
  double tmp;
#pragma omp for
  for(i=0;i<neq;i++){
    tmp = ad[i]*x[i];
    for(j=ia[i];j<ia[i+1];j++)
      tmp += a[j]*x[ja[j]];
    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPI2:produto matriz vetor para uma matriz generica no  *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpI2(INT neq           
                  ,INT *restrict ia  ,INT *restrict ja
                  ,double *restrict a,double *restrict ad
                  ,double *restrict x,double *restrict y)
{
  INT i,j;
  INT ia1,ia2;
  int resto,n;
  double tmp;
#pragma omp for 
  for(i=0;i<neq;i++){
    tmp = ad[i]*x[i];
    ia1 = ia[  i];
    ia2 = ia[i+1];

    n     = ia2 - ia1;
    resto = n%2;
    
    if(resto)
      tmp += a[ia1]*x[ja[ia1]];
    
    for(j=ia1+resto;j<ia2;j+=2)
      tmp += a[  j]*x[ja[  j]]
           + a[j+1]*x[ja[j+1]];

    y[i] = tmp;

  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPI4:produto matriz vetor para uma matriz generica no  *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpI4(INT neq           
                  ,INT *restrict ia  ,INT *restrict ja
                  ,double *restrict a,double *restrict ad
                  ,double *restrict x,double *restrict y)
{
  INT i,j;
  INT ia1,ia2;
  int resto,n;
  double tmp;
#pragma omp for 
  for(i=0;i<neq;i++){
    tmp = ad[i]*x[i];
    ia1 = ia[  i];
    ia2 = ia[i+1];

    n     = ia2 - ia1;
    resto = n%4;
    
    if(resto == 3)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      tmp += a[ia1]*x[ja[ia1]]; 

    for(j=ia1+resto;j<ia2;j+=4)
      tmp += a[  j]*x[ja[  j]]
           + a[j+1]*x[ja[j+1]] 
           + a[j+2]*x[ja[j+2]] 
           + a[j+3]*x[ja[j+3]];
    
    y[i] = tmp;

  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPI6:produto matriz vetor para uma matriz generica no  *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpI6(INT neq           
                  ,INT *restrict ia  ,INT *restrict ja
                  ,double *restrict a,double *restrict ad
                  ,double *restrict x,double *restrict y)
{
  INT i,j;
  INT ia1,ia2;
  int resto,n;
  double tmp;

#pragma omp for 
  for(i=0;i<neq;i++){
    tmp = ad[i]*x[i];
    ia1 = ia[  i];
    ia2 = ia[i+1];

    n     = ia2 - ia1;
    resto = n%6;
    
    if(resto == 5)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]  
           + a[ia1+4]*x[ja[ia1+4]]; 
    else if(resto == 4)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]; 
    else if(resto == 3)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      tmp += a[ia1]*x[ja[ia1]]; 

    for(j=ia1+resto;j<ia2;j+=6)
      tmp += a[  j]*x[ja[  j]]
           + a[j+1]*x[ja[j+1]] 
           + a[j+2]*x[ja[j+2]] 
           + a[j+3]*x[ja[j+3]] 
           + a[j+4]*x[ja[j+4]] 
           + a[j+5]*x[ja[j+5]];

    y[i] = tmp;

  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPO2: produto matriz vetor para uma matriz generica no *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpO2(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,j;
  int resto;
  double tmp1,tmp2;

  resto = neq %2;
  
#pragma omp single
{
  if(resto){
    tmp1  = ad[0]*x[0];
    for(j=0;j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    y[0] = tmp1;
  }
}  

#pragma omp for 
  for(i=resto;i<neq;i+=2){
    tmp1  = ad[i  ]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    if(ia1 == ia2) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
/*...................................................................*/

/*linha i+1*/
linhai1:
    y[i] = tmp1;
    if(ia2 == ia3) goto linhai2;

    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
/*...................................................................*/
linhai2:
    y[i+1] = tmp2;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPO4: produto matriz vetor para uma matriz generica no *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpO4(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ia4,ia5,j;
  int resto;
  double tmp1,tmp2,tmp3,tmp4;

  resto = neq %4;
  
#pragma omp single
{
  if(resto == 3){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    tmp3  = ad[2]*x[2];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    for(j=ia[2];j<ia[3];j++)
      tmp3 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
    y[2] = tmp3;
  }

  else if(resto == 2){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
  }
  
  else if(resto == 1){
    tmp1  = ad[0]*x[0];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    y[0] = tmp1;
  }
}  

#pragma omp for 
  for(i=resto;i<neq;i+=4){
    tmp1  = ad[i  ]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    tmp3  = ad[i+2]*x[i+2];
    tmp4  = ad[i+3]*x[i+3];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    ia4   = ia[i+3];
    ia5   = ia[i+4];
    if(ia1 == ia2) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
/*...................................................................*/

/*linha i+1*/
linhai1:
    y[i] = tmp1;
    if(ia2 == ia3) goto linhai2;

    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
/*...................................................................*/

/*linha i+2*/
linhai2:
    y[i+1] = tmp2;
    if(ia4 == ia3) goto linhai3;

    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
/*...................................................................*/

/*linha i+3*/
linhai3:
    y[i+2] = tmp3;
    if(ia5 == ia4) goto linhai4;

    for(j=ia4;j<ia5;j++)
      tmp4 += a[j]*x[ja[j]];
/*...................................................................*/
linhai4:
    y[i+3] = tmp4;

  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPO6: produto matriz vetor para uma matriz generica no *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpO6(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ia4,ia5,ia6,ia7,j;
  int resto;
  double tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;

  resto = neq%6;
  
#pragma omp single
{
  if(resto == 5){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    tmp3  = ad[2]*x[2];
    tmp4  = ad[3]*x[3];
    tmp5  = ad[4]*x[4];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    for(j=ia[2];j<ia[3];j++)
      tmp3 += a[j]*x[ja[j]];
    for(j=ia[3];j<ia[4];j++)
      tmp4 += a[j]*x[ja[j]];
    for(j=ia[4];j<ia[5];j++)
      tmp5 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
    y[2] = tmp3;
    y[3] = tmp4;
    y[4] = tmp5;
  }
  if(resto == 4){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    tmp3  = ad[2]*x[2];
    tmp4  = ad[3]*x[3];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    for(j=ia[2];j<ia[3];j++)
      tmp3 += a[j]*x[ja[j]];
    for(j=ia[3];j<ia[4];j++)
      tmp4 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
    y[2] = tmp3;
    y[3] = tmp4;
  }
  if(resto == 3){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    tmp3  = ad[2]*x[2];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    for(j=ia[2];j<ia[3];j++)
      tmp3 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
    y[2] = tmp3;
  }
  
  else if(resto == 2){
    tmp1  = ad[0]*x[0];
    tmp2  = ad[1]*x[1];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    for(j=ia[1];j<ia[2];j++)
      tmp2 += a[j]*x[ja[j]];
    y[0] = tmp1;
    y[1] = tmp2;
  }
  
  else if(resto == 1){
    tmp1  = ad[0]*x[0];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    y[0] = tmp1;
  }
}  

#pragma omp for
  for(i=resto;i<neq;i+=6){
    tmp1  = ad[i  ]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    tmp3  = ad[i+2]*x[i+2];
    tmp4  = ad[i+3]*x[i+3];
    tmp5  = ad[i+4]*x[i+4];
    tmp6  = ad[i+5]*x[i+5];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    ia4   = ia[i+3];
    ia5   = ia[i+4];
    ia6   = ia[i+5];
    ia7   = ia[i+6];
    if(ia1 == ia2) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
/*...................................................................*/

linhai1:
    y[i] = tmp1;

/*linha i+1*/
    if(ia2 == ia3) goto linhai2;

    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
/*...................................................................*/

linhai2:
    y[i+1] = tmp2;
/*linha i+2*/
    if(ia4 == ia3) goto linhai3;

    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
/*...................................................................*/

linhai3:
    y[i+2] = tmp3;
/*linha i+3*/
    if(ia5 == ia4) goto linhai4;

    for(j=ia4;j<ia5;j++)
      tmp4 += a[j]*x[ja[j]];
/*...................................................................*/

linhai4:
    y[i+3] = tmp4;
/*linha i+4*/
    if(ia6 == ia5) goto linhai5;

    for(j=ia5;j<ia6;j++)
      tmp5 += a[j]*x[ja[j]];
/*...................................................................*/

linhai5:
    y[i+4] = tmp5;
/*linha i+5*/
    if(ia7 == ia6) goto linhai6;

    for(j=ia6;j<ia7;j++)
      tmp6 += a[j]*x[ja[j]];
/*...................................................................*/
linhai6:
    y[i+5] = tmp6;

  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPO2I2: produto matriz vetor para uma matriz generica  *
 * no formato csr com a diagonal principal retirada                  *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpO2I2(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict ad
                   ,double *restrict x,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,j;
  int resto1,resto2,n;
  double tmp1,tmp2;
  
  resto1 = neq % 2;

#pragma omp single
{  
  if(resto1){
    tmp1  = ad[0]*x[0];
    for(j=ia[0];j<ia[1];j++)
      tmp1 += a[j]*x[ja[j]];
    y[0] = tmp1;
  }
}  
  
#pragma omp for
  for(i=resto1;i<neq;i+=2){
    tmp1  = ad[i  ]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    if(ia1 == ia2) goto linhai1;
    n      = ia2 - ia1;
    resto2 = n%2;
    if(resto2)
      tmp1 += a[ia1]*x[ja[ia1]];

/*linha i*/
    for(j=ia1+resto2;j<ia2;j+=2)
      tmp1 +=a[  j]*x[ja[  j]] 
           + a[j+1]*x[ja[j+1]];
/*...................................................................*/

linhai1:
    y[i] = tmp1;
/*linha i+1*/
    if(ia2 == ia3) goto linhai2;
    n      = ia3 - ia2;
    resto2 = n%2;
    if(resto2)
      tmp2 += a[ia2]*x[ja[ia2]];

    for(j=ia2+resto2;j<ia3;j+=2)
      tmp2 += a[  j]*x[  ja[j]]
            + a[j+1]*x[ja[j+1]];
/*...................................................................*/
linhai2:
    y[i+1] = tmp2;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPBAL:produto matriz vetor para uma matriz generica no *
 * formato csr com a diagonal principal retirada                     *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq        -> numero de equacoes                                  * 
 * ia         -> vetor csr                                            * 
 * ja         -> vetor csr                                           * 
 * a          -> vetor com os valores da matriz                      * 
 * ad         -> vetor com os valores da diagonal principal da matriz* 
 * x          -> vetor a ser multiplicado                            * 
 * y          -> indefinido                                          * 
 * threadBegin - primeira linha do sistema do thread i               * 
 * threadEnd   - ultima linha do sistema do thread i                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: Balanciamento do trabalho entre as threads feito manualmente * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpBal(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict ad
                     ,double *restrict x,double *restrict y
                     ,INT  *thBegin     ,INT *thEnd  )
{
  INT i,j;
  int id;
  double tmp;
  id = omp_get_thread_num();
  for(i=thBegin[id];i<thEnd[id];i++){
    tmp = ad[i]*x[i];
    for(j=ia[i];j<ia[i+1];j++)
      tmp += a[j]*x[ja[j]];
    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPBALI2:produto matriz vetor para uma matriz generica  *
 * no formato csr com a diagonal principal retirada                  *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq        -> numero de equacoes                                  * 
 * ia         -> vetor csr                                           * 
 * ja         -> vetor csr                                           * 
 * a          -> vetor com os valores da matriz                      * 
 * ad         -> vetor com os valores da diagonal principal da matriz* 
 * x          -> vetor a ser multiplicado                            * 
 * y          -> indefinido                                          * 
 * threadBegin - primeira linha do sistema do thread i               * 
 * threadEnd   - ultima linha do sistema do thread i                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: Balanciamento do trabalho entre as threads feito manualmente * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpBalI2(INT neq           
                       ,INT *restrict ia  ,INT *restrict ja
                       ,double *restrict a,double *restrict ad
                       ,double *restrict x,double *restrict y
                       ,INT  *thBegin     ,INT *thEnd  )
{
  INT i,j;
  INT ia1,ia2;
  int resto,n;
  int id;
  double tmp;
  id = omp_get_thread_num();
  for(i=thBegin[id];i<thEnd[id];i++){
    tmp   = ad[i]*x[i];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    n     = ia2 -ia1;
    resto = n%2;

    if(resto) 
      tmp += a[ia1]*x[ja[ia1]];
    
    for(j=ia1+resto;j<ia2;j+=2)
      tmp += a[  j]*x[ja[  j]] 
           + a[j+1]*x[ja[j+1]];
    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPBALI4:produto matriz vetor para uma matriz generica  *
 * no formato csr com a diagonal principal retirada                  *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq        -> numero de equacoes                                  * 
 * ia         -> vetor csr                                           * 
 * ja         -> vetor csr                                           * 
 * a          -> vetor com os valores da matriz                      * 
 * ad         -> vetor com os valores da diagonal principal da matriz* 
 * x          -> vetor a ser multiplicado                            * 
 * y          -> indefinido                                          * 
 * threadBegin - primeira linha do sistema do thread i               * 
 * threadEnd   - ultima linha do sistema do thread i                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: Balanciamento do trabalho entre as threads feito manualmente * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpBalI4(INT neq           
                       ,INT *restrict ia  ,INT *restrict ja
                       ,double *restrict a,double *restrict ad
                       ,double *restrict x,double *restrict y
                       ,INT  *thBegin     ,INT *thEnd  )
{
  INT i,j;
  INT ia1,ia2;
  int resto,n;
  int id;
  double tmp;
  id = omp_get_thread_num();
  for(i=thBegin[id];i<thEnd[id];i++){
    tmp   = ad[i]*x[i];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    n     = ia2 -ia1;
    resto = n%4;
    if(resto == 3)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      tmp += a[ia1]*x[ja[ia1]]; 
    
    for(j=ia1+resto;j<ia2;j+=4)
      tmp += a[  j]*x[ja[  j]]
           + a[j+1]*x[ja[j+1]] 
           + a[j+2]*x[ja[j+2]] 
           + a[j+3]*x[ja[j+3]];
    y[i] = tmp;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPBALI6:produto matriz vetor para uma matriz generica  *
 * no formato csr com a diagonal principal retirada                  *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq        -> numero de equacoes                                  * 
 * ia         -> vetor csr                                           * 
 * ja         -> vetor csr                                           * 
 * a          -> vetor com os valores da matriz                      * 
 * ad         -> vetor com os valores da diagonal principal da matriz* 
 * x          -> vetor a ser multiplicado                            * 
 * y          -> indefinido                                          * 
 * threadBegin - primeira linha do sistema do thread i               * 
 * threadEnd   - ultima linha do sistema do thread i                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: Balanciamento do trabalho entre as threads feito manualmente * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpBalI6(INT neq           
                       ,INT *restrict ia  ,INT *restrict ja
                       ,double *restrict a,double *restrict ad
                       ,double *restrict x,double *restrict y
                       ,INT  *thBegin     ,INT *thEnd  )
{
  INT i,j;
  INT ia1,ia2;
  int resto,n;
  int id;
  double tmp;
  id = omp_get_thread_num();
  for(i=thBegin[id];i<thEnd[id];i++){
    tmp = ad[i]*x[i];
    ia1 = ia[  i];
    ia2 = ia[i+1];

    n     = ia2 - ia1;
    resto = n%6;
    
    if(resto == 5)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]  
           + a[ia1+4]*x[ja[ia1+4]]; 
    else if(resto == 4)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]; 
    else if(resto == 3)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      tmp +=   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      tmp += a[ia1]*x[ja[ia1]]; 

    for(j=ia1+resto;j<ia2;j+=6)
      tmp += a[  j]*x[ja[  j]]
           + a[j+1]*x[ja[j+1]] 
           + a[j+2]*x[ja[j+2]] 
           + a[j+3]*x[ja[j+3]] 
           + a[j+4]*x[ja[j+4]] 
           + a[j+5]*x[ja[j+5]];

    y[i] = tmp;

  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPBALO2:produto matriz vetor para uma matriz generica  *
 * no formato csr com a diagonal principal retirada                  *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 * threadBegin - primeira linha do sistema do thread i               * 
 * threadEnd   - ultima linha do sistema do thread i                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: Balanciamento do trabalho entre as threads feito manualmente * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpBalO2(INT neq           
                       ,INT *restrict ia  ,INT *restrict ja
                       ,double *restrict a,double *restrict ad
                       ,double *restrict x,double *restrict y
                       ,INT  *thBegin     ,INT *thEnd  )
{ 

  INT i,j,ia1,ia2,ia3;
  int id,resto;
  double tmp1,tmp2;
 
  id = omp_get_thread_num();
  resto = (thEnd[id]-thBegin[id])%2; 
  
  if(resto){
    i     = thBegin[id];
    tmp1  = ad[i]*x[i];
    for(j=ia[i];j<ia[i+1];j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
  }
  
  for(i=thBegin[id]+resto;i<thEnd[id];i+=2){
    tmp1 = ad[  i]*x[  i];
    tmp2 = ad[i+1]*x[i+1];
    ia1  = ia[  i];
    ia2  = ia[i+1];
    ia3  = ia[i+2];
    if( ia1 == ia2 ) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
/*...................................................................*/

/*linha i+1*/
linhai1:
    y[i] = tmp1;
    if(ia2 == ia3) goto linhai2;

    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
/*...................................................................*/
linhai2:
    y[i+1] = tmp2;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPBALO4:produto matriz vetor par5.uma matriz generica  *
 * no formato csr com a diagonal principal retirada                  *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 * threadBegin - primeira linha do sistema do thread i               * 
 * threadEnd   - ultima linha do sistema do thread i                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: Balanciamento do trabalho entre as threads feito manualmente * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpBalO4(INT neq           
                       ,INT *restrict ia  ,INT *restrict ja
                       ,double *restrict a,double *restrict ad
                       ,double *restrict x,double *restrict y
                       ,INT  *thBegin     ,INT *thEnd  )
{
  INT i,j,ia1,ia2,ia3,ia4,ia5;
  int id,resto;
  double tmp1,tmp2,tmp3,tmp4;
 
  id = omp_get_thread_num();
  resto = (thEnd[id]-thBegin[id])%4; 
  
  if(resto == 3){
    i     = thBegin[id];
    tmp1  =   ad[i]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    tmp3  = ad[i+2]*x[i+2];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    ia4   = ia[i+3];
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
    y[i+1] = tmp2;
    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
    y[i+2] = tmp3;
  }
  else if(resto == 2){
    i     = thBegin[id];
    tmp1  =   ad[i]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
    y[i+1] = tmp2;
  }
  else if(resto == 1){
    i     = thBegin[id];
    tmp1  =   ad[i]*x[i];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
  }
  
  for(i=thBegin[id]+resto;i<thEnd[id];i+=4){
    tmp1 = ad[  i]*x[  i];
    tmp2 = ad[i+1]*x[i+1];
    tmp3 = ad[i+2]*x[i+2];
    tmp4 = ad[i+3]*x[i+3];
    ia1  = ia[  i];
    ia2  = ia[i+1];
    ia3  = ia[i+2];
    ia4  = ia[i+3];
    ia5  = ia[i+4];
    if( ia1 == ia2 ) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
/*...................................................................*/

linhai1:
    y[i] = tmp1;
/*linha i+1*/
    if(ia2 == ia3) goto linhai2;

    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
/*...................................................................*/
linhai2:
    y[i+1] = tmp2;
/*linha i+2*/
    if(ia3 == ia4) goto linhai3;

    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
/*...................................................................*/
linhai3:
    y[i+2] = tmp3;
/*linha i+3*/
    if(ia4 == ia5) goto linhai4;

    for(j=ia4;j<ia5;j++)
      tmp4 += a[j]*x[ja[j]];
/*...................................................................*/
linhai4:
    y[i+3] = tmp4;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPBALO6:produto matriz vetor para uma matriz generica  *
 * no formato csr com a diagonal principal retirada                  *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 * threadBegin - primeira linha do sistema do thread i               * 
 * threadEnd   - ultima linha do sistema do thread i                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: Balanciamento do trabalho entre as threads feito manualmente * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpBalO6(INT neq           
                       ,INT *restrict ia  ,INT *restrict ja
                       ,double *restrict a,double *restrict ad
                       ,double *restrict x,double *restrict y
                       ,INT  *thBegin     ,INT *thEnd  )
{
  INT i,j,ia1,ia2,ia3,ia4,ia5,ia6,ia7;
  int id,resto;
  double tmp1,tmp2,tmp3,tmp4,tmp5,tmp6;
 
  id = omp_get_thread_num();
  resto = (thEnd[id]-thBegin[id])%6; 
  
  if(resto == 5){
    i     = thBegin[id];
    tmp1  = ad[  i]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    tmp3  = ad[i+2]*x[i+2];
    tmp4  = ad[i+3]*x[i+3];
    tmp5  = ad[i+4]*x[i+4];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    ia4   = ia[i+3];
    ia5   = ia[i+4];
    ia6   = ia[i+5];
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
    y[i+1] = tmp2;
    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
    y[i+2] = tmp3;
    for(j=ia4;j<ia5;j++)
      tmp4 += a[j]*x[ja[j]];
    y[i+3] = tmp4;
    for(j=ia5;j<ia6;j++)
      tmp5 += a[j]*x[ja[j]];
    y[i+4] = tmp5;
  }
  else if(resto == 4){
    i     = thBegin[id];
    tmp1  = ad[  i]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    tmp3  = ad[i+2]*x[i+2];
    tmp4  = ad[i+3]*x[i+3];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    ia4   = ia[i+3];
    ia5   = ia[i+4];
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
    y[i+1] = tmp2;
    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
    y[i+2] = tmp3;
    for(j=ia4;j<ia5;j++)
      tmp4 += a[j]*x[ja[j]];
    y[i+3] = tmp4;
  }
  else if(resto == 3){
    i     = thBegin[id];
    tmp1  = ad[  i]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    tmp3  = ad[i+2]*x[i+2];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    ia4   = ia[i+3];
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
    y[i+1] = tmp2;
    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
    y[i+2] = tmp3;
  }
  else if(resto == 2){
    i     = thBegin[id];
    tmp1  = ad[  i]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
    y[i+1] = tmp2;
  }
  else if(resto == 1){
    i     = thBegin[id];
    tmp1  = ad[  i]*x[i];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
  }
  
  for(i=thBegin[id]+resto;i<thEnd[id];i+=6){
    tmp1 = ad[  i]*x[  i];
    tmp2 = ad[i+1]*x[i+1];
    tmp3 = ad[i+2]*x[i+2];
    tmp4 = ad[i+3]*x[i+3];
    tmp5 = ad[i+4]*x[i+4];
    tmp6 = ad[i+5]*x[i+5];
    ia1  = ia[  i];
    ia2  = ia[i+1];
    ia3  = ia[i+2];
    ia4  = ia[i+3];
    ia5  = ia[i+4];
    ia6  = ia[i+5];
    ia7  = ia[i+6];
    if( ia1 == ia2 ) goto linhai1;

/*linha i*/
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
/*...................................................................*/

linhai1:
    y[i] = tmp1;
/*linha i+1*/
    if(ia2 == ia3) goto linhai2;

    for(j=ia2;j<ia3;j++)
      tmp2 += a[j]*x[ja[j]];
/*...................................................................*/
linhai2:
    y[i+1] = tmp2;
/*linha i+2*/
    if(ia3 == ia4) goto linhai3;

    for(j=ia3;j<ia4;j++)
      tmp3 += a[j]*x[ja[j]];
/*...................................................................*/
linhai3:
    y[i+2] = tmp3;
/*linha i+3*/
    if(ia4 == ia5) goto linhai4;

    for(j=ia4;j<ia5;j++)
      tmp4 += a[j]*x[ja[j]];
/*...................................................................*/
linhai4:
    y[i+3] = tmp4;
/*linha i+4*/
    if(ia5 == ia6) goto linhai5;

    for(j=ia5;j<ia6;j++)
      tmp5 += a[j]*x[ja[j]];
/*...................................................................*/
linhai5:
    y[i+4] = tmp5;
/*linha i+5*/
    if(ia6 == ia7) goto linhai6;

    for(j=ia6;j<ia7;j++)
      tmp6 += a[j]*x[ja[j]];
/*...................................................................*/
linhai6:
    y[i+5] = tmp6;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRDOMPBALO2I2:produto matriz vetor para uma matriz generica*
 * no formato csr com a diagonal principal retirada                  *
 * (y=Ax, A uma matriz geral)                                        * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq        -> numero de equacoes                                  * 
 * ia         -> vetor csr                                           * 
 * ja         -> vetor csr                                           * 
 * a          -> vetor com os valores da matriz                      * 
 * ad         -> vetor com os valores da diagonal principal da matriz* 
 * x          -> vetor a ser multiplicado                            * 
 * y          -> indefinido                                          * 
 * threadBegin - primeira linha do sistema do thread i               * 
 * threadEnd   - ultima linha do sistema do thread i                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS: Balanciamento do trabalho entre as threads feito manualmente * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrDOmpBalO2I2(INT neq           
                       ,INT *restrict ia  ,INT *restrict ja
                       ,double *restrict a,double *restrict ad
                       ,double *restrict x,double *restrict y
                       ,INT  *thBegin     ,INT *thEnd  )
{
  INT i,j;
  INT ia1,ia2,ia3;
  int resto1,resto2,n;
  int id;
  double tmp1,tmp2;
  
  id = omp_get_thread_num();
  
  resto1 = (thEnd[id]-thBegin[id])%2; 
  
  if(resto1){
    i     = thBegin[id];
    tmp1  = ad[  i]*x[i];
    ia1   = ia[  i];
    ia2   = ia[i+1];
    for(j=ia1;j<ia2;j++)
      tmp1 += a[j]*x[ja[j]];
    y[i] = tmp1;
  }


  for(i=thBegin[id]+resto1;i<thEnd[id];i+=2){
    tmp1  = ad[i  ]*x[  i];
    tmp2  = ad[i+1]*x[i+1];
    ia1   = ia[i  ];
    ia2   = ia[i+1];
    ia3   = ia[i+2];
    if(ia1 == ia2) goto linhai1;
    n      = ia2 - ia1;
    resto2 = n%2;
    if(resto2)
      tmp1 += a[ia1]*x[ja[ia1]];

/*linha i*/
    for(j=ia1+resto2;j<ia2;j+=2)
      tmp1 +=a[  j]*x[ja[  j]] 
           + a[j+1]*x[ja[j+1]];
/*...................................................................*/

linhai1:
    y[i] = tmp1;
/*linha i+1*/
    if(ia2 == ia3) goto linhai2;
    n      = ia3 - ia2;
    resto2 = n%2;
    if(resto2)
      tmp2 += a[ia2]*x[ja[ia2]];
    
    for(j=ia2+resto2;j<ia3;j+=2)
      tmp2 += a[  j]*x[  ja[j]]
            + a[j+1]*x[ja[j+1]];
/*...................................................................*/
linhai2:
    y[i+1] = tmp2;
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCDOMP : produto matriz vetor para uma matriz no formato  * 
 * csrc (y=Ax, A uma matriz estruturalmente simentrica)              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq      -> numero de equacoes                                    * 
 * ia       -> vetor csr                                             * 
 * ja       -> vetor csr                                             * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x        -> vetor a ser multiplicado                              * 
 * y        -> indefinido                                            * 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thHeight -> altura efetiva do thread                              * 
 * nThreads -> numero de threads                                     * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCOmp(INT neq           
                  ,INT *restrict ia       ,INT *restrict ja
                  ,double *restrict au    ,double *restrict ad
                  ,double *restrict al                          
                  ,double *restrict x     ,double *restrict y
                  ,INT  *thBegin          ,INT *thEnd  
                  ,INT  *thHeight    
                  ,double *restrict thY   ,int nThreads)                     
{
  INT i,k,jak;
  int inc,id;
  double tmp,xi;

  id  = omp_get_thread_num();
  
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i]+inc;k<thBegin[i]+inc;k++){
      thY[k] = 0.0e0;
    }
  } 
  inc = id*neq; 

  for(i=thBegin[id];i<thEnd[id];i++){
    y[i]= 0.0e0; 
    xi  = x[i];
    tmp = ad[i]*xi;
    for(k=ia[i];k<ia[i+1];k++){
      jak = ja[k];
/*... produto da linha i pelo vetor x*/
      tmp     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak       = jak + inc;
      thY[jak] += au[k]*xi;
    }
    thY[i+inc] = tmp;
  }
#pragma omp barrier
 
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i];k<thEnd[i];k++){
      y[k] += thY[k+inc];
    }
  } 
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCDOMPO2:produto matriz vetor para uma matriz no formato  * 
 * csrc (y=Ax, A uma matriz estruturalmente simentrica)              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq      -> numero de equacoes                                    * 
 * ia       -> vetor csr                                             * 
 * ja       -> vetor csr                                             * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x        -> vetor a ser multiplicado                              * 
 * y        -> indefinido                                            * 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thHeight -> altura efetiva do thread                              * 
 * nThreads -> numero de threads                                     * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCOmpO2(INT neq           
                    ,INT *restrict ia       ,INT *restrict ja
                    ,double *restrict au    ,double *restrict ad
                    ,double *restrict al                          
                    ,double *restrict x     ,double *restrict y
                    ,INT  *thBegin          ,INT *thEnd  
                    ,INT  *thHeight    
                    ,double *restrict thY   ,int nThreads)                     
{
  INT i,k,jak;
  INT ia1,ia2,ia3;
  int inc,id;
  int resto;
  double tmp1,tmp2,xi1,xi2;

  id  = omp_get_thread_num();
  
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i]+inc;k<thBegin[i]+inc;k++){
      thY[k] = 0.0e0;
    }
  } 
  
  inc = id*neq;
  resto = (thEnd[id]-thBegin[id])%2; 
  if(resto){
    i    = thBegin[id];
    y[i] = 0.0e0; 
    xi1  = x[i];
    tmp1  = ad[i]*xi1;
    for(k=ia[i];k<ia[i+1];k++){
      jak = ja[k];
/*... produto da linha i pelo vetor x*/
      tmp1     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak       = jak + inc;
      thY[jak] += au[k]*xi1;
    }
    thY[i+inc] = tmp1;
  }
   

  for(i=thBegin[id]+resto;i<thEnd[id];i+=2){
    y[  i] = 0.0e0; 
    y[i+1] = 0.0e0; 
    xi1    =  x[  i];
    xi2    =  x[i+1];
    tmp1   = ad[  i]*xi1;
    tmp2   = ad[i+1]*xi2;
    ia1    = ia[  i]; 
    ia2    = ia[i+1]; 
    ia3    = ia[i+2];

    if( ia2 == ia1 ) goto linhai1; 

/*... linha i*/ 
    for(k=ia1;k<ia2;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp1      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi1;
    }
/*...................................................................*/

linhai1:
    thY[i+inc] = tmp1;
    
    if( ia3 == ia2 ) goto linhai2; 

/*... linha i+1*/ 
    for(k=ia2;k<ia3;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp2      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi2;
    }
/*...................................................................*/
  
linhai2:
    thY[i+inc+1] = tmp2;

  }

#pragma omp barrier
 
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i];k<thEnd[i];k++){
      y[k] += thY[k+inc];
    }
  } 
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCDOMPO4:produto matriz vetor para uma matriz no formato  * 
 * csrc (y=Ax, A uma matriz estruturalmente simentrica)              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq      -> numero de equacoes                                    * 
 * ia       -> vetor csr                                             * 
 * ja       -> vetor csr                                             * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x        -> vetor a ser multiplicado                              * 
 * y        -> indefinido                                            * 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thHeight -> altura efetiva do thread                              * 
 * nThreads -> numero de threads                                     * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCOmpO4(INT neq           
                    ,INT *restrict ia       ,INT *restrict ja
                    ,double *restrict au    ,double *restrict ad
                    ,double *restrict al                          
                    ,double *restrict x     ,double *restrict y
                    ,INT  *thBegin          ,INT *thEnd  
                    ,INT  *thHeight    
                    ,double *restrict thY   ,int nThreads)                     
{
  INT i,k,jak;
  INT ia1,ia2,ia3,ia4,ia5;
  int inc,id;
  int resto;
  double tmp1,tmp2,tmp3,tmp4,xi1,xi2,xi3,xi4;

  id  = omp_get_thread_num();
  
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i]+inc;k<thBegin[i]+inc;k++){
      thY[k] = 0.0e0;
    }
  } 
  
  inc   = id*neq;
  resto = (thEnd[id]-thBegin[id])%4; 

  if(resto == 3){
      i    = thBegin[id];
      y[  i] = 0.0e0; 
      y[i+1] = 0.0e0; 
      y[i+2] = 0.0e0; 
      xi1    = x[  i];
      xi2    = x[i+1];
      xi3    = x[i+2];
      tmp1   = ad[  i]*xi1;
      tmp2   = ad[i+1]*xi2;
      tmp3   = ad[i+2]*xi3;
      ia1    = ia[  i]; 
      ia2    = ia[i+1]; 
      ia3    = ia[i+2];
      ia4    = ia[i+3];
      for(k=ia1;k<ia2;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp1     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi1;
      }
      thY[i+inc] = tmp1;
  
      for(k=ia2;k<ia3;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp2     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+1)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi2;
      }
      thY[i+inc+1] = tmp2;
      
      for(k=ia3;k<ia4;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp3     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+3)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi3;
      }
      thY[i+inc+2] = tmp3;
  }

  else if(resto == 2){
      i      = thBegin[id];
      y[  i] = 0.0e0; 
      y[i+1] = 0.0e0; 
      xi1    = x[  i];
      xi2    = x[i+1];
      tmp1   = ad[  i]*xi1;
      tmp2   = ad[i+1]*xi2;
      ia1    = ia[  i]; 
      ia2    = ia[i+1]; 
      ia3    = ia[i+2];
      for(k=ia1;k<ia2;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp1     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi1;
      }
      thY[i+inc] = tmp1;
  
      for(k=ia2;k<ia3;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp2     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+1)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi2;
      }
      thY[i+inc+1] = tmp2;
      
  }
  
  else if(resto == 1){
      i    = thBegin[id];
      y[  i] = 0.0e0; 
      xi1    = x[  i];
      tmp1   = ad[  i]*xi1;
      ia1    = ia[  i]; 
      ia2    = ia[i+1]; 
      for(k=ia1;k<ia2;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp1     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi1;
      }
      thY[i+inc] = tmp1;
  }
   

  for(i=thBegin[id]+resto;i<thEnd[id];i+=4){
    y[  i] = 0.0e0; 
    y[i+1] = 0.0e0; 
    y[i+2] = 0.0e0; 
    y[i+3] = 0.0e0; 
    xi1    =  x[  i];
    xi2    =  x[i+1];
    xi3    =  x[i+2];
    xi4    =  x[i+3];
    tmp1   = ad[  i]*xi1;
    tmp2   = ad[i+1]*xi2;
    tmp3   = ad[i+2]*xi3;
    tmp4   = ad[i+3]*xi4;
    ia1    = ia[  i]; 
    ia2    = ia[i+1]; 
    ia3    = ia[i+2];
    ia4    = ia[i+3];
    ia5    = ia[i+4];

    if( ia2 == ia1 ) goto linhai1; 

/*... linha i*/ 
    for(k=ia1;k<ia2;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp1      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi1;
    }
/*...................................................................*/

linhai1:
    thY[i+inc] = tmp1;
    
    if( ia3 == ia2 ) goto linhai2; 

/*... linha i+1*/ 
    for(k=ia2;k<ia3;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp2      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+1)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi2;
    }
/*...................................................................*/
  
linhai2:
    thY[i+inc+1] = tmp2;
    
    if( ia4 == ia3 ) goto linhai3; 

/*... linha i+2*/ 
    for(k=ia3;k<ia4;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp3      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+2)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi3;
    }
/*...................................................................*/

linhai3:
    thY[i+inc+2] = tmp3;
    
    if( ia5 == ia4 ) goto linhai4; 

/*... linha i+3*/ 
    for(k=ia4;k<ia5;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp4      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+3)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi4;
    }
/*...................................................................*/

linhai4:
    thY[i+inc+3] = tmp4;

  }

#pragma omp barrier
 
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i];k<thEnd[i];k++){
      y[k] += thY[k+inc];
    }
  } 
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCDOMPO6:produto matriz vetor para uma matriz no formato  * 
 * csrc (y=Ax, A uma matriz estruturalmente simentrica)              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq      -> numero de equacoes                                    * 
 * ia       -> vetor csr                                             * 
 * ja       -> vetor csr                                             * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x        -> vetor a ser multiplicado                              * 
 * y        -> indefinido                                            * 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thHeight -> altura efetiva do thread                              * 
 * nThreads -> numero de threads                                     * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCOmpO6(INT neq           
                    ,INT *restrict ia       ,INT *restrict ja
                    ,double *restrict au    ,double *restrict ad
                    ,double *restrict al                          
                    ,double *restrict x     ,double *restrict y
                    ,INT  *thBegin          ,INT *thEnd  
                    ,INT  *thHeight    
                    ,double *restrict thY   ,int nThreads)                     
{
  INT i,k,jak;
  INT ia1,ia2,ia3,ia4,ia5,ia6,ia7;
  int inc,id;
  int resto;
  double tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,xi1,xi2,xi3,xi4,xi5,xi6;

  id  = omp_get_thread_num();
  
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i]+inc;k<thBegin[i]+inc;k++){
      thY[k] = 0.0e0;
    }
  } 
  
  inc   = id*neq;
  resto = (thEnd[id]-thBegin[id])%6; 
  
  if(resto == 5){
      i    = thBegin[id];
      y[  i] = 0.0e0; 
      y[i+1] = 0.0e0; 
      y[i+2] = 0.0e0; 
      y[i+3] = 0.0e0; 
      y[i+4] = 0.0e0; 
      xi1    = x[  i];
      xi2    = x[i+1];
      xi3    = x[i+2];
      xi4    = x[i+3];
      xi5    = x[i+4];
      tmp1   = ad[  i]*xi1;
      tmp2   = ad[i+1]*xi2;
      tmp3   = ad[i+2]*xi3;
      tmp4   = ad[i+3]*xi4;
      tmp5   = ad[i+4]*xi5;
      ia1    = ia[  i]; 
      ia2    = ia[i+1]; 
      ia3    = ia[i+2];
      ia4    = ia[i+3];
      ia5    = ia[i+4];
      ia6    = ia[i+5];
      for(k=ia1;k<ia2;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp1     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi1;
      }
      thY[i+inc] = tmp1;
  
      for(k=ia2;k<ia3;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp2     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+1)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi2;
      }
      thY[i+inc+1] = tmp2;
      
      for(k=ia3;k<ia4;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp3     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+3)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi3;
      }
      thY[i+inc+2] = tmp3;
      
      for(k=ia4;k<ia5;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp4     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+4)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi4;
      }
      thY[i+inc+3] = tmp4;
      
      for(k=ia5;k<ia6;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp5     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+5)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi5;
      }
      thY[i+inc+4] = tmp5;
  }
  else if(resto == 4){
      i    = thBegin[id];
      y[  i] = 0.0e0; 
      y[i+1] = 0.0e0; 
      y[i+2] = 0.0e0; 
      y[i+3] = 0.0e0; 
      xi1    = x[  i];
      xi2    = x[i+1];
      xi3    = x[i+2];
      xi4    = x[i+3];
      tmp1   = ad[  i]*xi1;
      tmp2   = ad[i+1]*xi2;
      tmp3   = ad[i+2]*xi3;
      tmp4   = ad[i+3]*xi4;
      ia1    = ia[  i]; 
      ia2    = ia[i+1]; 
      ia3    = ia[i+2];
      ia4    = ia[i+3];
      ia5    = ia[i+4];
      for(k=ia1;k<ia2;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp1     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi1;
      }
      thY[i+inc] = tmp1;
  
      for(k=ia2;k<ia3;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp2     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+1)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi2;
      }
      thY[i+inc+1] = tmp2;
      
      for(k=ia3;k<ia4;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp3     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+3)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi3;
      }
      thY[i+inc+2] = tmp3;
      
      for(k=ia4;k<ia5;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp4     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+3)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi4;
      }
      thY[i+inc+3] = tmp4;
  }

  else if(resto == 3){
      i    = thBegin[id];
      y[  i] = 0.0e0; 
      y[i+1] = 0.0e0; 
      y[i+2] = 0.0e0; 
      xi1    = x[  i];
      xi2    = x[i+1];
      xi3    = x[i+2];
      tmp1   = ad[  i]*xi1;
      tmp2   = ad[i+1]*xi2;
      tmp3   = ad[i+2]*xi3;
      ia1    = ia[  i]; 
      ia2    = ia[i+1]; 
      ia3    = ia[i+2];
      ia4    = ia[i+3];
      for(k=ia1;k<ia2;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp1     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi1;
      }
      thY[i+inc] = tmp1;
  
      for(k=ia2;k<ia3;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp2     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+1)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi2;
      }
      thY[i+inc+1] = tmp2;
      
      for(k=ia3;k<ia4;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp3     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+3)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi3;
      }
      thY[i+inc+2] = tmp3;
  }

  else if(resto == 2){
      i      = thBegin[id];
      y[  i] = 0.0e0; 
      y[i+1] = 0.0e0; 
      xi1    = x[  i];
      xi2    = x[i+1];
      tmp1   = ad[  i]*xi1;
      tmp2   = ad[i+1]*xi2;
      ia1    = ia[  i]; 
      ia2    = ia[i+1]; 
      ia3    = ia[i+2];
      for(k=ia1;k<ia2;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp1     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi1;
      }
      thY[i+inc] = tmp1;
  
      for(k=ia2;k<ia3;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp2     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+1)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi2;
      }
      thY[i+inc+1] = tmp2;
      
  }
  
  else if(resto == 1){
      i    = thBegin[id];
      y[  i] = 0.0e0; 
      xi1    = x[  i];
      tmp1   = ad[  i]*xi1;
      ia1    = ia[  i]; 
      ia2    = ia[i+1]; 
      for(k=ia1;k<ia2;k++){
        jak = ja[k];
/*... produto da linha i pelo vetor x*/
        tmp1     += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
        jak       = jak + inc;
        thY[jak] += au[k]*xi1;
      }
      thY[i+inc] = tmp1;
  }
   

  for(i=thBegin[id]+resto;i<thEnd[id];i+=6){
    y[  i] = 0.0e0; 
    y[i+1] = 0.0e0; 
    y[i+2] = 0.0e0; 
    y[i+3] = 0.0e0; 
    y[i+4] = 0.0e0; 
    y[i+5] = 0.0e0; 
    xi1    =  x[  i];
    xi2    =  x[i+1];
    xi3    =  x[i+2];
    xi4    =  x[i+3];
    xi5    =  x[i+4];
    xi6    =  x[i+5];
    tmp1   = ad[  i]*xi1;
    tmp2   = ad[i+1]*xi2;
    tmp3   = ad[i+2]*xi3;
    tmp4   = ad[i+3]*xi4;
    tmp5   = ad[i+4]*xi5;
    tmp6   = ad[i+5]*xi6;
    ia1    = ia[  i]; 
    ia2    = ia[i+1]; 
    ia3    = ia[i+2];
    ia4    = ia[i+3];
    ia5    = ia[i+4];
    ia6    = ia[i+5];
    ia7    = ia[i+6];

    if( ia2 == ia1 ) goto linhai1; 

/*... linha i*/ 
    for(k=ia1;k<ia2;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp1      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi1;
    }
/*...................................................................*/

linhai1:
    thY[i+inc] = tmp1;
    
    if( ia3 == ia2 ) goto linhai2; 

/*... linha i+1*/ 
    for(k=ia2;k<ia3;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp2      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+1)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi2;
    }
/*...................................................................*/
  
linhai2:
    thY[i+inc+1] = tmp2;
    
    if( ia4 == ia3 ) goto linhai3; 

/*... linha i+2*/ 
    for(k=ia3;k<ia4;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp3      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+2)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi3;
    }
/*...................................................................*/

linhai3:
    thY[i+inc+2] = tmp3;
    
    if( ia5 == ia4 ) goto linhai4; 

/*... linha i+3*/ 
    for(k=ia4;k<ia5;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp4      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+3)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi4;
    }
/*...................................................................*/

linhai4:
    thY[i+inc+3] = tmp4;
    
    if( ia6 == ia5 ) goto linhai5; 

/*... linha i+4*/ 
    for(k=ia5;k<ia6;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp5      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+4)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi5;
    }
/*...................................................................*/

linhai5:
    thY[i+inc+4] = tmp5;
    
    if( ia7 == ia6 ) goto linhai6; 

/*... linha i+5*/ 
    for(k=ia6;k<ia7;k++){
      jak        = ja[  k];
/*... produto da linha i pelo vetor x*/
      tmp6      += al[k]*x[jak];
/*... produto dos coef. da parte superior da matriz por x(i+5)*/
      jak        = jak  + inc;
      thY[jak]  += au[  k]*xi6;
    }
/*...................................................................*/

linhai6:
    thY[i+inc+5] = tmp6;

  }

#pragma omp barrier
 
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i];k<thEnd[i];k++){
      y[k] += thY[k+inc];
    }
  } 
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCDOMPI2:produto matriz vetor para uma matriz no formato  * 
 * csrc (y=Ax, A uma matriz estruturalmente simentrica)              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq      -> numero de equacoes                                    * 
 * ia       -> vetor csr                                             * 
 * ja       -> vetor csr                                             * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x        -> vetor a ser multiplicado                              * 
 * y        -> indefinido                                            * 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thHeight -> altura efetiva do thread                              * 
 * nThreads -> numero de threads                                     * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCOmpI2(INT neq           
                    ,INT *restrict ia       ,INT *restrict ja
                    ,double *restrict au    ,double *restrict ad
                    ,double *restrict al                          
                    ,double *restrict x     ,double *restrict y
                    ,INT  *thBegin          ,INT *thEnd  
                    ,INT  *thHeight    
                    ,double *restrict thY   ,int nThreads)                     
{
  INT i,k,jak1,jak2;
  INT ia1,ia2; 
  int inc,id;
  int resto,n;
  double tmp,xi;

  id  = omp_get_thread_num();
  
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i]+inc;k<thBegin[i]+inc;k++){
      thY[k] = 0.0e0;
    }
  } 
  
  inc = id*neq; 

  for(i=thBegin[id];i<thEnd[id];i++){
    y[i]  = 0.0e0; 
    xi    = x[i];
    tmp   = ad[i]*xi;
    ia1   = ia[  i];
    ia2   = ia[i+1];
    
    n     = ia2 - ia1;
    resto = n%2;

    if(resto){
      jak1       = ja[ia1];
      tmp       += al[ia1]*x[jak1];
      jak1       = jak1 + inc;
      thY[jak1] += au[ia1]*xi;
    }

    for(k=ia1+resto;k<ia2;k+=2){
      jak1       = ja[  k];
      jak2       = ja[k+1];
/*... produto da linha i pelo vetor x*/
      tmp       += al[k]*x[jak1] + al[k+1]*x[jak2];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      thY[jak1] += au[  k]*xi;
      thY[jak2] += au[k+1]*xi;
    }
    thY[i+inc] = tmp;
  }
#pragma omp barrier
 
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i];k<thEnd[i];k++){
      y[k] += thY[k+inc];
    }
  } 
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCDOMPI4:produto matriz vetor para uma matriz no formato  * 
 * csrc (y=Ax, A uma matriz estruturalmente simentrica)              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq      -> numero de equacoes                                    * 
 * ia       -> vetor csr                                             * 
 * ja       -> vetor csr                                             * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x        -> vetor a ser multiplicado                              * 
 * y        -> indefinido                                            * 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thHeight -> altura efetiva do thread                              * 
 * nThreads -> numero de threads                                     * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCOmpI4(INT neq           
                    ,INT *restrict ia       ,INT *restrict ja
                    ,double *restrict au    ,double *restrict ad
                    ,double *restrict al                          
                    ,double *restrict x     ,double *restrict y
                    ,INT  *thBegin          ,INT *thEnd  
                    ,INT  *thHeight    
                    ,double *restrict thY   ,int nThreads)                     
{
  INT i,k,jak1,jak2,jak3,jak4;
  INT ia1,ia2; 
  int inc,id;
  int resto,n;
  double tmp,xi;

  id  = omp_get_thread_num();
  
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i]+inc;k<thBegin[i]+inc;k++){
      thY[k] = 0.0e0;
    }
  } 
  
  inc = id*neq; 

  for(i=thBegin[id];i<thEnd[id];i++){
    y[i]  = 0.0e0; 
    xi    = x[i];
    tmp   = ad[i]*xi;
    ia1   = ia[  i];
    ia2   = ia[i+1];
    
    n     = ia2 - ia1;
    resto = n%4;

    if(resto == 3){
      jak1       = ja[  ia1];
      jak2       = ja[ia1+1];
      jak3       = ja[ia1+2];
      tmp       += al[ia1]*x[jak1] + al[ia1+1]*x[jak2] + al[ia1+2]*x[jak3];
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      jak3       = jak3 + inc;
      thY[jak1] += au[  ia1]*xi;
      thY[jak2] += au[ia1+1]*xi;
      thY[jak3] += au[ia1+2]*xi;
    }

    else if(resto == 2){
      jak1       = ja[  ia1];
      jak2       = ja[ia1+1];
      tmp       += al[ia1]*x[jak1] + al[ia1+1]*x[jak2]; 
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      thY[jak1] += au[  ia1]*xi;
      thY[jak2] += au[ia1+1]*xi;
    }

    else if(resto == 1){
      jak1       = ja[  ia1];
      tmp       += al[ia1]*x[jak1];
      jak1       = jak1 + inc;
      thY[jak1] += au[ia1]*xi;
    }

    for(k=ia1+resto;k<ia2;k+=4){
      jak1       = ja[  k];
      jak2       = ja[k+1];
      jak3       = ja[k+2];
      jak4       = ja[k+3];
/*... produto da linha i pelo vetor x*/
      tmp       += al[  k]*x[jak1] + al[k+1]*x[jak2] 
                 + al[k+2]*x[jak3] + al[k+3]*x[jak4];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      jak3       = jak3 + inc;
      jak4       = jak4 + inc;
      thY[jak1] += au[  k]*xi;
      thY[jak2] += au[k+1]*xi;
      thY[jak3] += au[k+2]*xi;
      thY[jak4] += au[k+3]*xi;
    }
    thY[i+inc] = tmp;
  }
#pragma omp barrier
 
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i];k<thEnd[i];k++){
      y[k] += thY[k+inc];
    }
  } 
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCDOMPI6:produto matriz vetor para uma matriz no formato  * 
 * csrc (y=Ax, A uma matriz estruturalmente simentrica)              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq      -> numero de equacoes                                    * 
 * ia       -> vetor csr                                             * 
 * ja       -> vetor csr                                             * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x        -> vetor a ser multiplicado                              * 
 * y        -> indefinido                                            * 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thHeight -> altura efetiva do thread                              * 
 * nThreads -> numero de threads                                     * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCOmpI6(INT neq           
                    ,INT *restrict ia       ,INT *restrict ja
                    ,double *restrict au    ,double *restrict ad
                    ,double *restrict al                          
                    ,double *restrict x     ,double *restrict y
                    ,INT  *thBegin          ,INT *thEnd  
                    ,INT  *thHeight    
                    ,double *restrict thY   ,int nThreads)                     
{
  INT i,k,jak1,jak2,jak3,jak4,jak5,jak6;
  INT ia1,ia2; 
  int inc,id;
  int resto,n;
  double tmp,xi;

  id  = omp_get_thread_num();
  
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i]+inc;k<thBegin[i]+inc;k++){
      thY[k] = 0.0e0;
    }
  } 
  
  inc = id*neq; 

  for(i=thBegin[id];i<thEnd[id];i++){
    y[i]  = 0.0e0; 
    xi    = x[i];
    tmp   = ad[i]*xi;
    ia1   = ia[  i];
    ia2   = ia[i+1];
    
    n     = ia2 - ia1;
    resto = n%6;
    if(resto == 5){
      jak1       = ja[  ia1];
      jak2       = ja[ia1+1];
      jak3       = ja[ia1+2];
      jak4       = ja[ia1+3];
      jak5       = ja[ia1+4];
      tmp       += al[  ia1]*x[jak1] + al[ia1+1]*x[jak2] 
                 + al[ia1+2]*x[jak3] + al[ia1+3]*x[jak4]
                 + al[ia1+4]*x[jak5];
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      jak3       = jak3 + inc;
      jak4       = jak4 + inc;
      jak5       = jak5 + inc;
      thY[jak1] += au[  ia1]*xi;
      thY[jak2] += au[ia1+1]*xi;
      thY[jak3] += au[ia1+2]*xi;
      thY[jak4] += au[ia1+3]*xi;
      thY[jak5] += au[ia1+4]*xi;
    }
    else if(resto == 4){
      jak1       = ja[  ia1];
      jak2       = ja[ia1+1];
      jak3       = ja[ia1+2];
      jak4       = ja[ia1+3];
      tmp       +=   al[ia1]*x[jak1] + al[ia1+1]*x[jak2] 
                 + al[ia1+2]*x[jak3] + al[ia1+3]*x[jak4];
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      jak3       = jak3 + inc;
      jak4       = jak4 + inc;
      thY[jak1] += au[  ia1]*xi;
      thY[jak2] += au[ia1+1]*xi;
      thY[jak3] += au[ia1+2]*xi;
      thY[jak4] += au[ia1+3]*xi;
    }
    else if(resto == 3){
      jak1       = ja[  ia1];
      jak2       = ja[ia1+1];
      jak3       = ja[ia1+2];
      tmp       +=   al[ia1]*x[jak1] + al[ia1+1]*x[jak2] 
                 + al[ia1+2]*x[jak3];
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      jak3       = jak3 + inc;
      thY[jak1] += au[  ia1]*xi;
      thY[jak2] += au[ia1+1]*xi;
      thY[jak3] += au[ia1+2]*xi;
     }

    else if(resto == 2){
      jak1       = ja[  ia1];
      jak2       = ja[ia1+1];
      tmp       += al[ia1]*x[jak1] + al[ia1+1]*x[jak2]; 
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      thY[jak1] += au[  ia1]*xi;
      thY[jak2] += au[ia1+1]*xi;
    }

    else if(resto == 1){
      jak1       = ja[  ia1];
      tmp       += al[ia1]*x[jak1];
      jak1       = jak1 + inc;
      thY[jak1] += au[ia1]*xi;
    }
  
    for(k=ia1+resto;k<ia2;k+=6){
      jak1       = ja[  k];
      jak2       = ja[k+1];
      jak3       = ja[k+2];
      jak4       = ja[k+3];
      jak5       = ja[k+4];
      jak6       = ja[k+5];
/*... produto da linha i pelo vetor x*/
      tmp       += al[  k]*x[jak1] + al[k+1]*x[jak2]
                 + al[k+2]*x[jak3] + al[k+3]*x[jak4]
                 + al[k+4]*x[jak5] + al[k+5]*x[jak6];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      jak3       = jak3 + inc;
      jak4       = jak4 + inc;
      jak5       = jak5 + inc;
      jak6       = jak6 + inc;
      thY[jak1] += au[  k]*xi;
      thY[jak2] += au[k+1]*xi;
      thY[jak3] += au[k+2]*xi;
      thY[jak4] += au[k+3]*xi;
      thY[jak5] += au[k+4]*xi;
      thY[jak6] += au[k+5]*xi;
    }
    thY[i+inc] = tmp;
  }
#pragma omp barrier
 
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i];k<thEnd[i];k++){
      y[k] += thY[k+inc];
    }
  } 
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSRCDOMPO2I2:produto matriz vetor para uma matriz no formato* 
 * csrc (y=Ax, A uma matriz estruturalmente simentrica)              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq      -> numero de equacoes                                    * 
 * ia       -> vetor csr                                             * 
 * ja       -> vetor csr                                             * 
 * au  -> vetor com os valores superiores da matriz                  * 
 * ad  -> vetor com os valores da diagonal principal da matriz       * 
 * al  -> vetor com os valores inferiores da matriz                  * 
 * x        -> vetor a ser multiplicado                              * 
 * y        -> indefinido                                            * 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thHeight -> altura efetiva do thread                              * 
 * nThreads -> numero de threads                                     * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrCOmpO2I2(INT neq           
                    ,INT *restrict ia       ,INT *restrict ja
                    ,double *restrict au    ,double *restrict ad
                    ,double *restrict al                          
                    ,double *restrict x     ,double *restrict y
                    ,INT  *thBegin          ,INT *thEnd  
                    ,INT  *thHeight    
                    ,double *restrict thY   ,int nThreads)                     
{
  INT i,k,jak1,jak2;
  INT ia1,ia2,ia3; 
  int inc,id;
  int resto1,resto2,n;
  double tmp1,tmp2,xi1,xi2;

  id  = omp_get_thread_num();
  
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i]+inc;k<thBegin[i]+inc;k++){
      thY[k] = 0.0e0;
    }
  } 
  
  inc    = id*neq; 
  resto1 = (thEnd[id]-thBegin[id])%2; 
  if(resto1){
    i    = thBegin[id];
    y[i] = 0.0e0; 
    xi1  = x[i];
    tmp1 = ad[i]*xi1;
    for(k=ia[i];k<ia[i+1];k++){
      jak1 = ja[k];
/*... produto da linha i pelo vetor x*/
      tmp1     += al[k]*x[jak1];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak1      = jak1 + inc;
      thY[jak1] += au[k]*xi1;
    }
    thY[i+inc] = tmp1;
  }

  for(i=thBegin[id]+resto1;i<thEnd[id];i+=2){
    y[  i]  = 0.0e0; 
    y[i+1]  = 0.0e0; 
    xi1     = x[  i];
    xi2     = x[i+1];
    tmp1    = ad[  i]*xi1;
    tmp2    = ad[i+1]*xi2;
    ia1     = ia[  i];
    ia2     = ia[i+1];
    ia3     = ia[i+2];
    
/*... i*/    
    if( ia2 == ia1 ) goto linhai1; 
    n     = ia2 - ia1;
    resto2 = n%2;

    if(resto2){
      jak1       = ja[ia1];
      tmp1      += al[ia1]*x[jak1];
      jak1       = jak1 + inc;
      thY[jak1] += au[ia1]*xi1;
    }

    for(k=ia1+resto2;k<ia2;k+=2){
      jak1       = ja[  k];
      jak2       = ja[k+1];
/*... produto da linha i pelo vetor x*/
      tmp1      += al[k]*x[jak1] + al[k+1]*x[jak2];
/*... produto dos coef. da parte superior da matriz por x(i)*/
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      thY[jak1] += au[  k]*xi1;
      thY[jak2] += au[k+1]*xi1;
    }
/*...................................................................*/

linhai1:
    thY[i+inc] = tmp1;

/*... i + 1*/    
    if( ia3 == ia2 ) goto linhai2; 
    n     = ia3 - ia2;
    resto2 = n%2;

    if(resto2){
      jak1       = ja[ia2];
      tmp2      += al[ia2]*x[jak1];
      jak1       = jak1 + inc;
      thY[jak1] += au[ia2]*xi2;
    }

    for(k=ia2+resto2;k<ia3;k+=2){
      jak1       = ja[  k];
      jak2       = ja[k+1];
/*... produto da linha i pelo vetor x*/
      tmp2      += al[k]*x[jak1] + al[k+1]*x[jak2];
/*... produto dos coef. da parte superior da matriz por x(i+1)*/
      jak1       = jak1 + inc;
      jak2       = jak2 + inc;
      thY[jak1] += au[  k]*xi2;
      thY[jak2] += au[k+1]*xi2;
    }
/*...................................................................*/

linhai2:
    thY[i+inc+1] = tmp2;
  }
#pragma omp barrier
 
  for(i=0;i<nThreads;i++){
    inc = i*neq;
#pragma omp for 
    for(k=thHeight[i];k<thEnd[i];k++){
      y[k] += thY[k+inc];
    }
  } 
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSROMP :produto matriz vetor para uma matriz generica no    *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrOmp(INT neq           
                 ,INT *restrict ia  ,INT *restrict ja
                 ,double *restrict a,double *restrict x
                 ,double *restrict y)
{
  INT i,j;
#pragma omp for private(i,j)
  for(i=0;i<neq;i++){
    y[i] = 0.0e0;
    for(j=ia[i];j<ia[i+1];j++)
      y[i] += a[j]*x[ja[j]];
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSROMPI2 :produto matriz vetor para uma matriz generica no  *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrOmpI2(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict x
                   ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ja1;
  int resto,n;
#pragma omp for private(i,ja1,ia1,ia2,resto,n)
  for(i=0;i<neq;i++){
    y[i] = 0.0e0;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    n    = ia2 - ia1;
    resto = n%2;
    if(resto)
      y[i] = a[ia1]*x[ja[ia1]];
    
    for(ja1=ia1+resto;ja1<ia2;ja1+=2)
      y[i] += a[  ja1]*x[  ja[ja1]] 
            + a[ja1+1]*x[ja[ja1+1]];
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSROMPI4 :produto matriz vetor para uma matriz generica no  *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrOmpI4(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict x
                   ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ja1;
  int resto,n;
#pragma omp for private(i,ja1,ia1,ia2,resto,n)
  for(i=0;i<neq;i++){
    y[i] = 0.0e0;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    n    = ia2 - ia1;
    resto = n%4;

    if(resto == 3)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      y[i] = a[ia1]*x[ja[ia1]]; 

    
    for(ja1=ia1+resto;ja1<ia2;ja1+=4)
      y[i] += a[  ja1]*x[  ja[ja1]] 
            + a[ja1+1]*x[ja[ja1+1]] 
            + a[ja1+2]*x[ja[ja1+2]] 
            + a[ja1+3]*x[ja[ja1+3]];
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSROMPI6 :produto matriz vetor para uma matriz generica no  *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrOmpI6(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict x
                   ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ja1;
  int resto,n;
#pragma omp for private(i,ja1,ia1,ia2,resto,n)
  for(i=0;i<neq;i++){
    y[i] = 0.0e0;
    ia1  = ia[  i];
    ia2  = ia[i+1];
    n    = ia2 - ia1;
    resto = n%6;

    if(resto == 5)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]  
           + a[ia1+4]*x[ja[ia1+4]]; 
    else if(resto == 4)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]  
           + a[ia1+3]*x[ja[ia1+3]]; 
    else if(resto == 3)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]] 
           + a[ia1+2]*x[ja[ia1+2]]; 
    else if(resto == 2)
      y[i] =   a[ia1]*x[  ja[ia1]] 
           + a[ia1+1]*x[ja[ia1+1]];
    else if(resto == 1)
      y[i] = a[ia1]*x[ja[ia1]]; 

    
    for(ja1=ia1+resto;ja1<ia2;ja1+=6)
      y[i] += a[  ja1]*x[  ja[ja1]] 
            + a[ja1+1]*x[ja[ja1+1]] 
            + a[ja1+2]*x[ja[ja1+2]] 
            + a[ja1+3]*x[ja[ja1+3]] 
            + a[ja1+4]*x[ja[ja1+4]] 
            + a[ja1+5]*x[ja[ja1+5]];
  }
} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSROMPO2 :produto matriz vetor para uma matriz generica no  *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrOmpO2(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict x
                   ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ja1,ja2;
  int resto;

  resto = neq%2;
#pragma omp single
{  
  if(resto){
    y[0] =0.0e0;
    for(ja1=0;ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
  }
}

#pragma omp for private(i,ja1,ja2,ia1,ia2,ia3)
  for(i=resto;i<neq;i+=2){
    y[i]   = 0.0e0;
    y[i+1] = 0.0e0;
    ia1    = ia[  i];
    ia2    = ia[i+1];
    ia3    = ia[i+2];
    if(ia1 == ia2) goto linhai1;
/*linha i*/
    for(ja1=ia1;ja1<ia2;ja1++)
      y[i] += a[ja1]*x[ja[ja1]];
/*...................................................................*/

/*linha i+1*/
linhai1:
    if(ia2 == ia3) goto linhai2;
    
    for(ja2=ia2;ja2<ia3;ja2++)
      y[i+1] += a[ja2]*x[ja[ja2]];
/*...................................................................*/
linhai2:
    continue;
  }

} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSROMPO4 :produto matriz vetor para uma matriz generica no  *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrOmpO4(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict x
                   ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ia4,ia5,ja1,ja2,ja3,ja4;
  int resto;

  resto = neq%4;

#pragma omp single
{  
  if(resto == 3){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    y[2] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[2];ja1<ia[3];ja1++)
      y[2] += a[ja1]*x[ja[ja1]];
  }
  
  else if(resto == 2){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]]; 
  }
  
  else if(resto == 1){
    y[0] = 0.0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
  }
}

#pragma omp for private(i,ja1,ja2,ja3,ja4,ia1,ia2,ia3,ia4,ia5)
  for(i=resto;i<neq;i+=4){
    y[i]   = 0.0e0;
    y[i+1] = 0.0e0;
    y[i+2] = 0.0e0;
    y[i+3] = 0.0e0;
    ia1    = ia[  i];
    ia2    = ia[i+1];
    ia3    = ia[i+2];
    ia4    = ia[i+3];
    ia5    = ia[i+4];
    if(ia2 == ia1) goto linhai1;
/*linha i*/
    for(ja1=ia1;ja1<ia2;ja1++)
      y[i] += a[ja1]*x[ja[ja1]];
/*...................................................................*/
linhai1:
    if(ia2 == ia3) goto linhai2;
/*linha i+1*/
    for(ja2=ia2;ja2<ia3;ja2++)
      y[i+1] += a[ja2]*x[ja[ja2]];
/*...................................................................*/
linhai2:
    if(ia4 == ia3) goto linhai3;
/*linha i+2*/
    for(ja3=ia3;ja3<ia4;ja3++)
      y[i+2] += a[ja3]*x[ja[ja3]];
/*...................................................................*/
linhai3:
    if(ia5 == ia4) goto linhai4;
/*linha i+3*/
    for(ja4=ia4;ja4<ia5;ja4++)
      y[i+3] += a[ja4]*x[ja[ja4]];
/*...................................................................*/
linhai4:
    continue;
  }

} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSROMPO6 :produto matriz vetor para uma matriz generica no  *
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrOmpO6(INT neq           
                   ,INT *restrict ia  ,INT *restrict ja
                   ,double *restrict a,double *restrict x
                   ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ia4,ia5,ia6,ia7,ja1,ja2,ja3,ja4,ja5,ja6;
  int resto;

  resto = neq%6;

#pragma omp single
{  
  if(resto == 5){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    y[2] = 0.0e0; 
    y[3] = 0.0e0; 
    y[4] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[2];ja1<ia[3];ja1++)
      y[2] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[3];ja1<ia[4];ja1++)
      y[3] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[4];ja1<ia[5];ja1++)
      y[4] += a[ja1]*x[ja[ja1]];
  }
  if(resto == 4){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    y[2] = 0.0e0; 
    y[3] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[2];ja1<ia[3];ja1++)
      y[2] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[3];ja1<ia[4];ja1++)
      y[3] += a[ja1]*x[ja[ja1]];
  }
  if(resto == 3){
    y[0] = 0.0; 
    y[1] = 0.0; 
    y[2] = 0.0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]];
    for(ja1=ia[2];ja1<ia[3];ja1++)
      y[2] += a[ja1]*x[ja[ja1]];
  }
  
  else if(resto == 2){
    y[0] = 0.0e0; 
    y[1] = 0.0e0; 
    
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
    
    for(ja1=ia[1];ja1<ia[2];ja1++)
      y[1] += a[ja1]*x[ja[ja1]]; 
  }
  
  else if(resto == 1){
    y[0] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
  }
}

#pragma omp for private(i,ja1,ja2,ja3,ja4,ja5,ja6,ia1,ia2,ia3,ia4,ia5)
  for(i=resto;i<neq;i+=6){
    y[i]   = 0.0e0;
    y[i+1] = 0.0e0;
    y[i+2] = 0.0e0;
    y[i+3] = 0.0e0;
    y[i+4] = 0.0e0;
    y[i+5] = 0.0e0;
    ia1    = ia[  i];
    ia2    = ia[i+1];
    ia3    = ia[i+2];
    ia4    = ia[i+3];
    ia5    = ia[i+4];
    ia6    = ia[i+5];
    ia7    = ia[i+6];
    if(ia2 == ia1) goto linhai1;
/*linha i*/
    for(ja1=ia1;ja1<ia2;ja1++)
      y[i] += a[ja1]*x[ja[ja1]];
/*...................................................................*/
linhai1:
    if(ia2 == ia3) goto linhai2;
/*linha i+1*/
    for(ja2=ia2;ja2<ia3;ja2++)
      y[i+1] += a[ja2]*x[ja[ja2]];
/*...................................................................*/
linhai2:
    if(ia4 == ia3) goto linhai3;
/*linha i+2*/
    for(ja3=ia3;ja3<ia4;ja3++)
      y[i+2] += a[ja3]*x[ja[ja3]];
/*...................................................................*/
linhai3:
    if(ia5 == ia4) goto linhai4;
/*linha i+3*/
    for(ja4=ia4;ja4<ia5;ja4++)
      y[i+3] += a[ja4]*x[ja[ja4]];
/*...................................................................*/
linhai4:
    if(ia6 == ia5) goto linhai5;
/*linha i+4*/
    for(ja5=ia5;ja5<ia6;ja5++)
      y[i+4] += a[ja5]*x[ja[ja5]];
/*...................................................................*/
linhai5:
    if(ia7 == ia6) goto linhai6;
/*linha i+5*/
    for(ja6=ia6;ja6<ia7;ja6++)
      y[i+5] += a[ja6]*x[ja[ja6]];
/*...................................................................*/
linhai6:
    continue;
  }

} 
/*********************************************************************/ 

/********************************************************************* 
 * MATVECCSROMPO2O2 :produto matriz vetor para uma matriz generica no*
 * formato csr padrao (y=Ax, A uma matriz geral)                     * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * neq -> numero de equacoes                                         * 
 * ia  -> vetor csr                                                  * 
 * ja  -> vetor csr                                                  * 
 * a   -> vetor com os valores da matriz                             * 
 * x   -> vetor a ser multiplicado                                   * 
 * y   -> indefinido                                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * y   -> vetor com o resultado da multiplicacao                     * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matVecCsrOmpO2I2(INT neq           
                     ,INT *restrict ia  ,INT *restrict ja
                     ,double *restrict a,double *restrict x
                     ,double *restrict y)
{
  INT i;
  INT ia1,ia2,ia3,ja1,ja2;
  int resto1,resto2,resto3,n1,n2;

  resto1 = neq%2;

#pragma omp single
{  
  if(resto1){
    y[0] = 0.0e0; 
    for(ja1=ia[0];ja1<ia[1];ja1++)
      y[0] += a[ja1]*x[ja[ja1]];
  }
}

#pragma omp for private(i,ja1,ja2,ia1,ia2,ia3,resto2,resto3,n1,n2)
  for(i=resto1;i<neq;i+=2){
    y[i]   = 0.0e0;
    y[i+1] = 0.0e0;
    ia1    = ia[  i];
    ia2    = ia[i+1];
    ia3    = ia[i+2];
    n1     = ia2 - ia1;
    if(!n1) goto linhai1;
    resto2 = n1%2;
/*linha i*/
    if(resto2){
      y[i] = a[ia1]*x[ja[ia1]];
    }
    for(ja1=ia1+resto2;ja1<ia2;ja1+=2)
      y[i] += a[  ja1]*x[ja[ja1  ]]
            + a[ja1+1]*x[ja[ja1+1]];
/*...................................................................*/
linhai1:    
/*linha i+1*/
    n2     = ia3 - ia2;
    if(!n2) goto linhai2;
    resto3 = n2%2;
    if(resto3){
      y[i+1] = a[ia2]*x[ja[ia2]];
    }
    for(ja2=ia2+resto3;ja2<ia3;ja2+=2)
      y[i+1] += a[ja2  ]*x[ja[ja2  ]]
              + a[ja2+1]*x[ja[ja2+1]];
/*...................................................................*/
linhai2:
    continue;    
  }

} 
/*********************************************************************/ 
#endif
