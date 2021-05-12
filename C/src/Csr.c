#include<Csr.h>
/********************************************************************* 
 * CSRIA:                                                            * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * ia     -> indefinido                                              * 
 * id     -> numeracao das equacoes por elemento                     * 
 * num    -> renumeracao dos elementos                               * 
 * adj    -> adjacencia dos elementos                                * 
 * nViz   -> numero de vizinhos por elemento                         * 
 * numel  -> numero de elementos                                     * 
 * neq    -> numero de equacoes                                      * 
 * ndf    -> numero de graus de liberade                             * 
 * maxViz -> numero maximo de vizinho da malha                       * 
 * upper  -> armazenamento da parte superior da matriz (CSR/CSRC)    * 
 * diag   -> armazenamento da diagonal (CSR/CSRC)                    * 
 * lower  -> armazenamenro da parte inferior (CSR/CSRC)              * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * ia     -> ponteiro do CSR                                         * 
 *-------------------------------------------------------------------* 
 * OBS: a funcao retorna o numero do termos nao nulor no CSR/CSRC    * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
INT csrIa(INT *ia  ,INT *id    ,INT *num   ,INT  *adj, short *nViz
          ,INT numel,INT neq    ,short maxViz,short  ndf, bool upper
          ,bool diag , bool lower){
  
  INT  i,nel1,neq1,neq2,viz1,col,aux;
  short jNdf,kNdf,j;
/*... gerando arranjo ia*/  
  ia[0] = 0;
  for(i=0;i<numel;i++){
    nel1= num[i]-1;
    for(jNdf=0;jNdf<ndf;jNdf++){
      aux = 0;
      neq1 = MAT2D(nel1,jNdf,id,ndf)-1;
      if(neq1 != -2){
/*... conectividade no proprio elemento*/
        for(kNdf=0;kNdf<ndf;kNdf++){
          neq2 = MAT2D(nel1,kNdf,id,ndf)-1;
          if(neq2 != -2){
/*... parte superior*/
            if(lower && neq1 > neq2) 
              aux++;
/*... parte inferior*/            
            else if(upper && neq1 < neq2)
              aux++;
/*... diagonal princial*/      
            else if(diag && neq1 == neq2 ) 
              aux++;
          }
        }
/*...................................................................*/
  
/*... conecitivada nos vizinhos*/
        for(j=0;j<nViz[nel1];j++){
          viz1 = MAT2D(nel1,j,adj,maxViz) - 1;
          if( viz1 != -2) {
            for(kNdf=0;kNdf<ndf;kNdf++){
              col   = MAT2D(viz1,kNdf,id,ndf)-1;
              if( col != -2){
/*... parte superior*/
                if(lower && col < neq1) 
                  aux++;
/*...................................................................*/

/*... parte inferior*/            
                else if(upper && col > neq1)
                  aux++;
              }
/*...................................................................*/
            }    
/*...................................................................*/
          }
/*...................................................................*/
        }
/*...................................................................*/
        ia[neq1+1] = ia[neq1] + aux;
      }
/*...................................................................*/
    }
  }
/*...................................................................*/
  return ia[neq] - ia[0];
}
/*********************************************************************/ 


/********************************************************************* 
 * CSRJA:                                                            * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * ia     -> arranjo CSR/CSRC                                        * 
 * ja     -> indefinido                                              * 
 * id     -> numeracao das equacoes por elemento                     * 
 * num    -> renumeracao dos elementos                               * 
 * adj    -> adjacencia dos elementos                                * 
 * nViz   -> numero de vizinhos por elemento                         * 
 * numel  -> numero de elementos                                     * 
 * neq    -> numero de equacoes                                      * 
 * maxViz -> numero maximo de vizinho da malha                       * 
 * ndf    -> numero de graus de liberade                             * 
 * upper  -> armazenamento da parte superior da matriz (CSR/CSRC)    * 
 * diag   -> armazenamento da diagonal (CSR/CSRC)                    * 
 * lower  -> armazenamenro da parte inferior (CSR/CSRC)              * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * ja     -> ponteiro do CSR                                         * 
 *-------------------------------------------------------------------* 
 * OBS:                                                              * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void csrJa(INT *ia    ,INT *ja 
          ,INT *id    ,INT *num   ,INT  *adj  ,short *nViz
          ,INT numel,INT neq      ,short maxViz,short ndf
          ,bool upper,bool diag     ,bool lower){
  
  INT  i,nel1,neq1,neq2,viz1,col,aux,ipont;
  short j,jNdf,kNdf;

/*... gerando arranjo ja*/  
  for(i=0;i<numel;i++){
    nel1= num[i]-1;
    for(jNdf=0;jNdf<ndf;jNdf++){
      aux = 0;
      neq1= MAT2D(nel1,jNdf,id,ndf)-1;
      ipont = ia[neq1];
      if(neq1 != -2){
/*... conectividade no proprio elemento*/
        for(kNdf=0;kNdf<ndf;kNdf++){
          neq2 = MAT2D(nel1,kNdf,id,ndf)-1;
          if(neq2 != -2){
/*... parte superior*/
            if(lower && neq1 > neq2){ 
              ja[ipont+aux] = neq2; 
              aux++;
            }
/*... parte inferior*/            
            else if(upper && neq1 < neq2){
              ja[ipont+aux] = neq2; 
              aux++;
            }
/*... diagonal princial*/      
            else if(diag && neq1 == neq2 ){ 
              ja[ipont+aux] = neq1;  
              aux++;
            }
/*...................................................................*/
          }
/*...................................................................*/
        }
/*...................................................................*/
/*...*/
        for(j=0;j<nViz[nel1];j++){
          viz1 = MAT2D(nel1,j,adj,maxViz) - 1;
          if( viz1 != -2) {
            for(kNdf=0;kNdf<ndf;kNdf++){
              col= MAT2D(viz1,kNdf,id,ndf)-1;
              if( col != -2){
/*... parte superior*/
                if(lower && col < neq1){
                  ja[ipont+aux] = col; 
                  aux++;
                }
/*...................................................................*/

/*... parte inferior*/            
                else if(upper && col > neq1){
                  ja[ipont+aux] = col; 
                  aux++;
                }
/*...................................................................*/
              }
/*...................................................................*/
            }
/*...................................................................*/
          }    
/*...................................................................*/
        }
/*...................................................................*/
      }
/*...................................................................*/
    }
/*...................................................................*/
  }
/*...................................................................*/
}
/*********************************************************************/ 


/********************************************************************* 
 * BANDCSR: banda da matriz no formato CSR                           * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * ia  - ponteiro CSR                                                * 
 * ja  - ponteiro CSR                                                * 
 * neq - numero de equacoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS: retorna a banda da matrix                                    * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
INT bandCsr(INT *ia,INT *ja,INT  neq,short type){

  INT i,j,bandL=0,aux;
  
  switch(type){
/*... banda maxima da matriz*/
    case 1:
      for(i=0;i<neq;i++){
        for(j=ia[i];j<ia[i+1];j++){
          bandL = max(bandL,abs(i-ja[j]));
        }
      }
    break;
/*...................................................................*/ 

/*... banda media da matriz*/
    case 2: 
      for(i=0;i<neq;i++){
        aux = 0;
        for(j=ia[i];j<ia[i+1];j++){
          aux = max(aux,abs(i-ja[j]));
        }
        bandL += aux;
      }
      bandL = bandL/neq;
    break;
/*...................................................................*/ 

/*... banda minima da matriz*/
    case 3: 
      for(i=0;i<neq;i++){
        for(j=ia[i];j<ia[i+1];j++){
          bandL = min(bandL,abs(i-ja[j]));
        }
      }
    break;
/*...................................................................*/ 
  }
  return bandL;

}
/*********************************************************************/ 

/********************************************************************* 
 * NLCSR: numero de elementos nao nulos por linha                    * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * ia  - ponteiro CSR                                                * 
 * ja  - ponteiro CSR                                                * 
 * neq - numero de equacoes                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 *                                                                   * 
 *-------------------------------------------------------------------* 
 * OBS: retorna o numero de elmentos nao nulos                       * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
INT nlCsr(INT *ia,INT  neq,short type){

  INT i,nl=0,n;
  
  switch(type){
/*... numero maximo de elementos nao nulos por linha*/
    case 1:
      for(i=0;i<neq;i++){
        n  = ia[i+1] - ia[i];
        nl = max(nl,n);
      }
    break;
/*...................................................................*/ 

/*... numero medios de elementos nao nulos por linha*/
    case 2: 
      n  = ia[neq] - ia[0];
      nl = n/neq;
    break;
/*...................................................................*/ 

/*... numero minimo de elementos nao nulos por linha*/
    case 3: 
      for(i=0;i<neq;i++){
        n  = ia[i+1] - ia[i];
        nl = min(nl,n);
      }
    break;
/*...................................................................*/ 
  }
  return nl;

}
/*********************************************************************/ 

#ifdef _OPENMP
/********************************************************************* 
 * PARTITIOBCSRBYNOZEROS: divisa do trabalho por threads para matriz * 
 * no formato CSR                                                    * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * ia       -> ponteiro CSR                                          * 
 * ja       -> ponteiro CSR                                          * 
 * neq      -> numero de equacoes                                    * 
 * thBegin  -> nao definido                                          * 
 * thEnd    -> nao definido                                          * 
 * thSize   -> nao definido                                          * 
 * thHeigth -> nao definido                                          * 
 * type     -> CSR,CSRD,CSRC                                         * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thSize   -> numero de termo nao nulos no thread i                 * 
 * thHeight -> altura efetiva do thread                              * 
 *-------------------------------------------------------------------* 
 * OBS: retorna o numero de elmentos nao nulos                       * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void partitionCsrByNonzeros(INT *ia         ,INT *ja         
                           ,INT neq
                           ,int nThreads  ,INT *thBegin
                           ,INT *thEnd    ,INT *thSize
                           ,INT *thHeight ,short type){
  
  INT nad,meanVariables,line;
  INT tam;
  int i,nTh;
  
  for(i = 0;i<nThreads;i++){
    thBegin[i]  = 0;
    thEnd[i]    = 0;
    thSize[i]   = 0;
    thHeight[i] = 0;
  }
/*se o numero de threads for maior que o numero de equacoes*/
  if(neq < nThreads){
   nThreads = neq;
  }
  nad           = ia[neq];
  switch(type){
/*... CSR padrao*/
    case CSR:
      meanVariables = nad / nThreads;
      line = 1;
      thBegin[0] = 0;
      for(i = 0;i<nThreads-1;i++){
        thSize[i] = 0;
        tam           = 0;
        for(;;){
          tam              = ia[line] - ia[line-1];
          thSize[i]   += tam;
          thEnd[i]     = line-1;
          thBegin[i+1] = line;
          line++;
          if( (thSize[i] + tam) > meanVariables) break;
          if(line > neq ) {
            ERRO_GERAL(__FILE__,__func__,"numero de linhas excedido");
          }
        } 
      }
      nTh = nThreads - 1;
      thSize[nTh] = ia[neq] - ia[thBegin[nTh]];  
      thEnd [nTh] = neq-1;   
      break;
/*...................................................................*/

/*... CSRD - csr sem a diagonal principal*/
    case CSRD:
      meanVariables = (nad + neq)/ nThreads;
      line = 1;
      thBegin[0] = 0;
      for(i = 0;i<nThreads-1;i++){
        thSize[i] = 0;
        tam           = 0;
        for(;;){
          tam              = ia[line] - ia[line-1] + 1;
          thSize[i]   += tam;
          thEnd[i]     = line;
          thBegin[i+1] = line;
          line++;
          if( (thSize[i] + tam) > meanVariables) break;
          if(line > neq ) {
            ERRO_GERAL(__FILE__,__func__,"numero de linhas excedido");
          }
        } 
      }
      nTh = nThreads - 1;
      thEnd [nTh]=neq;   
      thSize[nTh]=ia[neq] - ia[thBegin[nTh]] +
                      (thEnd[nTh] - thBegin[nTh]+1);   
      break;
/*...................................................................*/

/*... CSRC*/
    case CSRC:
      meanVariables = (2*nad+neq) / nThreads;
      line = 1;
      thBegin[0] = 0;
      for(i = 0;i<nThreads-1;i++){
        thSize[i]     = 0;
        tam           = 0;
        for(;;){
          tam              = 2*(ia[line] - ia[line-1])+1;
          thSize[i]   += tam;
          thEnd[i]     = line;
          thBegin[i+1] = line;
          line++;
          if( (thSize[i] + tam) > meanVariables) break;
          if(line > neq ) {
            ERRO_GERAL(__FILE__,__func__,"numero de linhas excedido");
          }
        } 
      }
      nTh = nThreads - 1;
      thEnd [nTh]=neq;   
      thSize[nTh]=2*(ia[neq] - ia[thBegin[nTh]]) + 1;

/*... calcula o tamanho efetivo do buffer*/
      computeEffectiveWork(ia         ,ja         
                          ,neq
                          ,thBegin    ,thEnd  
                          ,thSize     ,thHeight   );
   
      break;
/*...................................................................*/

/*...*/
    default:
      ERRO_OP(__FILE__,__func__,type);
      break;
  }
/*...................................................................*/
  
}
/*********************************************************************/ 
 
/********************************************************************* 
 * COMPUTEEFFECTIVEWORK: calcula o trabalho efetivo por thread       * 
 *-------------------------------------------------------------------* 
 * Parametros de entrada:                                            * 
 *-------------------------------------------------------------------* 
 * ia      -> ponteiro CSR                                           * 
 * ja      -> ponteiro CSR                                           * 
 * neq     -> numero de equacoes                                     * 
 * thBegin -> nao definido                                           * 
 * thEnd   -> nao definido                                           * 
 * thSize  -> nao definido                                           * 
 * thHeight-> nao definido                                           * 
 * type    -> CSR,CSRD,CSRC                                          * 
 *-------------------------------------------------------------------* 
 * Parametros de saida:                                              * 
 *-------------------------------------------------------------------* 
 * thBegin  -> primeira linha do sistema do thread i                 * 
 * thEnd    -> ultima linha do sistema do thread i                   * 
 * thSize   -> numero de termo nao nulos no thread i                 * 
 * thHeight -> altura efetiva do thread                              * 
 *-------------------------------------------------------------------* 
 * OBS: retorna o numero de elmentos nao nulos                       * 
 *-------------------------------------------------------------------* 
 *********************************************************************/
void computeEffectiveWork(INT *ia         ,INT *ja         
                         ,INT neq
                         ,INT *thBegin    ,INT *thEnd  
                         ,INT *thSize     ,INT *thHeight   ){
  int i,id=0,h=0;

  #pragma omp parallel private(id,h)
  {      
    id = omp_get_thread_num();
    h  = thBegin[id];
    for(i=thBegin[id];i<thEnd[id];i++) 
      h = min( h , ja[ia[i]]);  
    thHeight[id] = h;
  }

} 
#endif
