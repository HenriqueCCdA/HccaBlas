#ifndef _GRAPH_H  
  #define _GRAPH_H
   #ifdef _AD_
    #undef  _AD_
  #endif  
  #define _AD_ false
  #include<HccaStdBool.h>
  #include<Define.h>
/*...*/
  void sortGraphCsr(INT *ia,INT *ja,INT n);
/*...*/  
  void convGraph(INT *xAdj    ,INT *adjncy,INT const *adj,short const *nViz
                ,short maxViz  ,INT numel  ,bool xAdjFlag  ,bool adjFlag );
/*...*/
  void bubblesort(INT *ja,INT n);

#endif/*_CSR_H*/
