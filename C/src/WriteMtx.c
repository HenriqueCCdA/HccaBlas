#include<Coo.h>
/********************************************************************* 
 * WRITECOO : escreve o grafo da matrix no formato MM (MATRIX MARKET)* 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * m     -> memoria principal                                        * 
 * ia    -> vetor da matrix                                          * 
 * ja    -> vetor da matrix                                          * 
 * neq   -> numero de equacoes                                       * 
 * nad   -> numero de termos nao nulos fora da diagonal principal    * 
 * type  -> tipo de armazenamento ( 1 - CSR/CSRC)                    * 
 * unsym -> simetria da matrix                                       * 
 * name  -> nume do arquivo de saida                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void writeCoo(Memoria *m,INT *ia   ,INT *ja,INT neq
             ,INT nad  ,short type
             ,bool unsym,char *name){
#ifdef _MMIO_  
  MM_typecode matcode;
  int  *lin,*col;
  double *val;
  INT nTotal; 

/*...*/    
  mm_initialize_typecode(&matcode);
  mm_set_matrix(&matcode);
  mm_set_coordinate(&matcode);
  mm_set_real(&matcode);
  if(unsym) 
    mm_set_general(&matcode);
  else
    mm_set_symmetric(&matcode);
/*...................................................................*/

/*...*/
  nTotal = neq + nad;
  hccaAlloc(int   ,m,lin,nTotal,"lin",_AD_);
  hccaAlloc(int   ,m,col,nTotal,"col",_AD_);
  hccaAlloc(double,m,val,nTotal,"val",_AD_);
  zero(lin,nTotal,"int");
  zero(col,nTotal,"int");
  zero(val,nTotal,"double");
/*...................................................................*/
  
  switch(type){
/*... CSR/CSRC*/
    case 1:
      csrToCoo(lin,col,val,ia,ja,neq,nad);
      mm_write_mtx_crd(name,neq,neq,nTotal,lin,col,val,matcode);
    break;
/*...................................................................*/

/*...*/
    default:
      printf("\n opcao invalida\n"
           "funcao fname(*,*,*)\narquivo = %s\n",__FILE__);
      exit(EXIT_FAILURE);
    break;
/*...................................................................*/
    hccaDealloc(m,val,"val",false);
    hccaDealloc(m,col,"col",false);
    hccaDealloc(m,lin,"lin",false);
  }
#else
      printf("\nEscrita da matriz no formato MM não disponivel.\n"
           "funcao %s\narquivo = %s\n",__func__,__FILE__);
      exit(EXIT_FAILURE);
#endif
}
/*********************************************************************/ 

/********************************************************************* 
 * CSRTOCOO : conveter do formato CSR para COO                       * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * lin -> indefinido                                                 * 
 * col -> indefinido                                                 * 
 * val -> indefinido                                                 * 
 * ia  -> vetor CSR                                                  * 
 * ja  -> vetor CSR                                                  * 
 * neq -> numero de equacoes                                         * 
 * nad -> numero de termos nao nulos                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * lin -> numero da linha                                            * 
 * col -> numero da coluna                                           * 
 * val -> valor                                                      * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void csrToCoo(int *lin, int *col,double *val,INT *ia,INT *ja
             ,INT neq,INT nad){
  
  INT i,j,kk,nl1,n,ipoint;

  kk = 0;
  for(i=0;i<neq;i++){
/*... diagonal principal*/
    nl1     = i + 1; 
    lin[kk] = col[kk] = nl1;
    val[kk] = 1.0;
    kk++;
/*...................................................................*/
    n  = ia[i+1] - ia[i];
    ipoint = ia[i];
    for(j=0;j<n ;j++){
      lin[kk] = nl1;
      col[kk]  = ja[ipoint+j]+1;
      val[kk]  = 1.0;
      kk++;
    }
  }
}
/*********************************************************************/ 

/*********************************************************************
 * COOTOCSR : conveter do formato COO para CSR                       *
 *-------------------------------------------------------------------*
 * Parametros de entrada:                                            *
 *-------------------------------------------------------------------*
 * lin -> linhas do COO                                              *
 * col -> colunad do COO                                             *
 * val -> valores do COO                                             *
 * ia  ->                                                            *
 * ja  ->                                                            *
 * au  ->                                                            *
 * ad  ->                                                            *
 * al  ->                                                            *
 * type-> tt                                                         *
 * neq -> tipo do Csr                                                *
 * nad -> numero de termos nao nulos                                 *
 * upper -> parte superior da matriz                                 *
 * diag  -> diagonal da matriz                                       *
 * lower -> parte inferior da matriz                                 *
 *-------------------------------------------------------------------*
 * Parametros de saida:                                              *
 *-------------------------------------------------------------------*
 * ia  -> vetor CSR                                                  *
 * ja  -> vetor CSR                                                  *
 * au  -> vetor CSR                                                  *
 * ad  -> vetor CSR                                                  *
 * al  -> vetor CSR                                                  *
 *-------------------------------------------------------------------*
 * OBS:                                                              *
 *-------------------------------------------------------------------*
 *********************************************************************/
void cooToCsr(int *lin  ,int *col  ,double *val
             ,INT *ia   ,INT *ja
             ,double *au,double *ad,double *al
             ,INT neq   ,INT nad   ,short type
             ,int *aux  ,bool upper,bool diag,bool lower){
  
  INT i,j,k,ipont,nJa;
  INT nLin,nCol;

  for(i=0;i<neq;i++){
    aux[i] = 0.0;
  }
/*... vetor ia*/
  for(i=0;i<nad;i++){
    nLin = lin[i]-1;
    nCol = col[i]-1;
    if(upper && nCol > nLin) 
      aux[nLin]++;
    else if( diag && nCol == nLin)
      aux[nLin]++;
    else if( lower && nCol < nLin)
      aux[nLin]++;
  }
  
  ia[0] = 0;
  for(i=1;i<neq+1;i++){
    ia[i] = ia[i-1] + aux[i-1];
  }
/*...................................................................*/

/*... vetor ja*/
  for(i=0;i<neq;i++){
    aux[i] = 0.0;
  }
  
  for(i=0;i<nad;i++){
    nLin  = lin[i]-1;
    nCol  = col[i]-1;
    ipont = ia[nLin]; 
/*    printf("ipont %d aux %d\n",ipont,aux[nLin]);
    printf("l     %d c   %d\n",nLin,nCol);*/
    if(upper && nCol > nLin){
      ja[ipont+aux[nLin]] = nCol;
      aux[nLin]++;
    }
    else if( diag && nCol == nLin){
      ja[ipont+aux[nLin]] = nCol;
      aux[nLin]++;
    }
    else if( lower && nCol < nLin){
      ja[ipont+aux[nLin]] = nCol;
      aux[nLin]++;
    }
  }
  
/*...................................................................*/

/*...*/  
  sortGraphCsr(ia,ja,neq);
/*...................................................................*/

  switch(type){
/*... csr padrao*/
    case 1:
/*... vetor a*/
      for(k=0;k<nad;k++){
        nLin  = lin[k]-1;
        nCol  = col[k]-1;
        ipont = ia[nLin];
        for(j=ipont;j<ia[nLin+1];j++){
          nJa           = ja[j];
          if( nCol == nJa){
            ad[j] = val[k];
            break;
          }
        }  
      }
    break;
/*... csrMdw*/
    case 2:
/*... vetor ad a*/
      for(k=0;k<nad;k++){
        nLin  = lin[k]-1;
        nCol  = col[k]-1;
        ipont = ia[nLin];
        if(nLin == nCol)
          ad[nLin] = val[k];
        else
          for(j=ipont;j<ia[nLin+1];j++){
            nJa           = ja[j];
            if( nCol == nJa){
              al[j] = val[k];   
              break;
            }
          }  
        }
    break;
/*... csrC*/
    case 3:
      for(k=0;k<nad;k++){
        nLin  = lin[k]-1;
        nCol  = col[k]-1;
        if(nLin == nCol)
          ad[nLin] = val[k];
        else if(nLin > nCol){
          ipont = ia[nLin];
          for(j=ipont;j<ia[nLin+1];j++){
            nJa           = ja[j];
            if( nCol == nJa){
              al[j] = val[k];   
              break;
            }
          }
        }    
        else if(nLin < nCol){
          ipont = ia[nCol];
          for(j=ipont;j<ia[nCol+1];j++){
            nJa           = ja[j];
            if( nLin == nJa){
              au[j] = val[k];   
              break;
            }
          }
        }      
      }
    break;
/*...*/
    default:
      ERRO_OP(__FILE__,__func__,type);
      break;
  }
/*...................................................................*/

/*...*/
//  printf("neq %d nad %d nnz %d\n",neq,ia[neq],nad);
//  for(i=0;i<neq+1;i++)
//    printf("ia %d %d\n",i+1,ia[i]);
//  for(i=0;i<ia[neq];i++)
//    printf("ja %d %d\n",i+1,ja[i]);
//  if(type ==1) {
//    for(i=0;i<ia[neq];i++)
//      printf("a   %d %lf\n",i+1,ad[i]);
//  }
//  else if(type ==2) {
//    for(i=0;i<ia[neq];i++)
//      printf("a  %d %lf\n",i+1,al[i]);
//    for(i=0;i<neq;i++)
//      printf("ad %d %lf\n",i+1,ad[i]);
//  }
//  else if(type ==3) {
//    for(i=0;i<ia[neq];i++)
//      printf("al %d %lf\n",i+1,al[i]);
//    for(i=0;i<ia[neq];i++)
//      printf("au %d %lf\n",i+1,au[i]);
//    for(i=0;i<neq;i++)
//      printf("ad %d %lf\n",i+1,ad[i]);
//  }
/*...................................................................*/

/*...................................................................*/ 
}
/*********************************************************************/ 

/********************************************************************* 
 * COOTOFULL: conveter do formato COO para matriz cheia              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * lin -> linhas                                                     * 
 * col -> colunas                                                    * 
 * val -> valores                                                    * 
 * neq -> numero de equacoes                                         * 
 * nad -> numero de termos nao nulos                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * a   -> matriz cheia                                               * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void cooToFull(int *lin, int *col,double *val,double *a,INT neq,INT nad)
{
  INT nLin,nCol,i;
  
  for(i=0;i<neq*neq;i++)
    a[i] = 0.0;
  
  for(i=0;i<nad;i++){
    nLin = lin[i] -1;
    nCol = col[i] -1;
    MAT2D(nLin,nCol,a,neq) = val[i];
  }
}
/*********************************************************************/ 

/********************************************************************* 
 * MATRIXCHECK: nveter do formato COO para matriz cheia              * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * lin -> linhas                                                     * 
 * col -> colunas                                                    * 
 * val -> valores                                                    * 
 * nl  -> numero de equacoes                                         * 
 * nnz -> numero de termos nao nulos                                 * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void matrixCheck(double *val,int *lin,int *col, int nl,int nnz){

  int i,j,nLin1,nCol1,nLin2,nCol2,aux;
  bool flag;

/*... termos nulos na diagonal principal*/ 
  aux = 0;
  for(i=0;i<nnz;i++){
    nLin1 = lin[i];
    nCol1 = col[i];
    if( nLin1 == nCol1 && val[i] != 0.0)
      aux++;
  }
  if( aux == nl){
    printf("Todos os termos diagonal principal são nao nulos!!\n");
  }
/*...................................................................*/

/*... checa se a matrix e estruturalemente simetrica*/
  aux = 0;
  for(i=0;i<nnz;i++){
    nLin1 = lin[i];
    nCol1 = col[i];
    if( nLin1 != nCol1){
      flag = true;
      for(j=0;j<nnz;j++){
        nLin2 = lin[j];
        nCol2 = col[j];
        if( (nLin1 == nCol2) && (nLin2 == nCol1)){
          flag=false;
          break;
        }
      }
      if(flag)
        break;
    }
  }
  if(flag){
    printf("A Matriz nao e estruturalmente simetricamente!!\n");
    exit(EXIT_FAILURE);
  }
/*...................................................................*/

} 
/*********************************************************************/ 
