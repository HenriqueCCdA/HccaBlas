#ifndef _DEFINE_
  #include<stdio.h>
  #include<stdlib.h>
  #define _DEFINE_
/*numero maximo de propriedade*/
  #define MAXPROP      15 /*numero maximo de propriedades*/
  #define MAXMAT      200 /*numero maximo de materias*/
  #define MAX_TRANS_EQ 3 /*numero maximo de equacoes de transporte*/ 
/*...................................................................*/

/*... GPU*/
  #define MAX_THREADS_PER_BLOCK 1024
/*...................................................................*/


/*... CSR*/
  #define CSR  1
  #define CSRD 2
  #define CSRC 3
/*...................................................................*/

/*...cellLoop*/
  #define  MAX_NUM_RES   28
  #define  MAX_NUM_PONT  168
  #define  MAX_NUM_FACE  6
  #define  MAX_SN        24 
/*...................................................................*/

/*... vtk elmentos*/
  #define VTK_TRIA      5
  #define VTK_QUAD      9
  #define VTK_TETR     10
  #define VTK_HEXA     12
/*...................................................................*/

/*... definicao do tipo de inteiros usados*/
  #define INT        int 
  #define INTC      "int"
  #define DOUBLE   double
  #define DOUBLEC "double"
/*...................................................................*/

/*... macro para acesso matricial em vetores*/
  #define   MAT2D(i,j,vector,col)           vector[i*col+j]
  #define   MAT3D(i,j,k,vector,col1,col2)   vector[i*col1*col2+col2*j+k]
/*...................................................................*/

/*... definicao de funcoes*/
  #define min(a, b)  (((a) < (b)) ? (a) : (b))
  #define max(a, b)  (((a) > (b)) ? (a) : (b))
  #define vectorPlusOne(v,n,i)  for(i=0;i<n;i++)  v[i] = v[i] + 1;  
  #define vectorMinusOne(v,n,i) for(i=0;i<n;i++)  v[i] = v[i] - 1;  
/*...................................................................*/

/*... Saida de Erro*/                                                  
  #define ERRO_RCM fprintf(stderr,"\nrcm - fatal error!\n")
  #define ERRO_MEM(a,str)\
  if(a ==NULL){\
       fprintf(stderr,"Erro na alocacao do vetor %s\n",str);\
       exit(EXIT_FAILURE);\
  }
  #define ERRO_ARGS(argv,nargs,nameArgs)\
    fprintf(stderr,"Usage:\n");for(i=0;i<nargs;i++)\
    fprintf(stderr,"%s %s\n",argv[0],nameArgs[i]);\
    return EXIT_SUCCESS;
  
  #define ERRO_OP(file,func,op)\
    fprintf(stderr,"Opecao %d e invalida!!\n",op);\
    fprintf(stderr,"Arquivo:%s\nFonte:  %s\n",file,func);\
    exit(EXIT_FAILURE);

  #define ERRO_GERAL(file,func,str)\
    fprintf(stderr,"Erro: %s!!\n",str);\
    fprintf(stderr,"Arquivo:%s\nFonte:  %s\n",file,func);\
    exit(EXIT_FAILURE);
/*...................................................................*/
#endif/*_DEFINE_*/
