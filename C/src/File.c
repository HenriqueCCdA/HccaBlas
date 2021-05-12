#include<File.h>
/*********************************************************************
 * openFile : abre um arquivo                                        *
 * ------------------------------------------------------------------*
 * Parametros de entrada:                                            *
 * ------------------------------------------------------------------* 
 * name  -> nome do arquivo                                          * 
 * mod   -> mode de abertura                                         *
 * ------------------------------------------------------------------*
 * Paramanetros de saida:                                            *
 * ------------------------------------------------------------------*
 * FILE * ponterio par o arquivo de saida                            * 
 * ------------------------------------------------------------------*
 * *******************************************************************/
FILE* openFile(char *name,char *mod){
FILE *aux;
  if(( aux =fopen(name,mod))==NULL){
      fprintf(stderr,"Erro na abertura do arquivo %s.\n",name);
      exit(EXIT_FAILURE);
    }
  return aux;  
}
/********************************************************************/

/*********************************************************************
 * fname: add as extencoes dos arquivos de saida                     *
 * ------------------------------------------------------------------*
 * Parametros de entrada:                                            *
 * ------------------------------------------------------------------* 
 * out   -> nome do arquivo de saida                                 * 
 * num   -> passo no tempo                                           *
 * cod   -> codigo                                                   *
 *          1 -> particianomento vtk novlp                           *
 *          2 -> particianomento vtk  ovlp                           *
 *          3 -> malha por paricao vtk                               *
 *          4 -> arquivo de dados mefpar                             *      
 *          5 -> resultados do mefc      formato vtk (wave)          *      
 *          6 -> carrgamentos formato vtk                            *      
 *          7 -> arquivo de log de excucao                           *      
 *          8 -> resultados do mefc      formato vtk (temp)          *      
 *          9 -> derivdas  formato vtk (tflux)                       *      
 *         10 -> resultados do mefc vtk(concentracoes)               *      
 * ------------------------------------------------------------------*
 * Paramanetros de saida:                                            *
 * ------------------------------------------------------------------*
 * out -> aruivo de saida com a extencao                             * 
 * ------------------------------------------------------------------*
 * *******************************************************************/
void fName(char *name,INT num,int cod ,char **out ){
/*===*/
  char st[MAX_STR_NUMBER];
  char ext[MAX_EXT];
  int size1,size2;
/*===================================================================*/
/**/
/*===*/
  strcpy(ext,"");
  strcpy(*out,"");
  switch( cod ) {
/*.... vtk no-overlanping*/
    case 1:
      
      iota(num,st);
      strcpy(ext,"_n_");
      strcat(ext,st);
      strcat(ext,".vtk");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;     
/*...................................................................*/

/*...*/
    case 3:
      iota(num,st);
      strcpy(ext,"_n_");
      strcat(ext,st);
      strcat(ext,".part.dat");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*...*/
    case 4:
      iota(num,st);
      strcpy(ext,"_n_");
      strcat(ext,st);
      strcat(ext,".part.vtk");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*...*/
    case 5:
      iota(num,st);
      strcpy(ext,"_step_");
      strcat(ext,st);
      strcat(ext,".wave.vtk");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*...*/
    case 6:
      strcat(ext,"_pgeo.vtk");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*... arquivo com o log de excucao*/
    case 7:
      strcat(ext,"_log.txt");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*...*/
    case 8:
      iota(num,st);
      strcpy(ext,"_step_");
      strcat(ext,st);
      strcat(ext,".temp.vtk");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*...*/
    case 9:
      iota(num,st);
      strcpy(ext,"_step_");
      strcat(ext,st);
      strcat(ext,".flux.vtk");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*...*/
    case 10:
      iota(num,st);
      strcpy(ext,"_step_");
      strcat(ext,st);
      strcat(ext,".conc.vtk");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*... arquivo com o log do solv*/
    case 11:
      strcat(ext,"_solv_log.txt");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*... arquivo com o log do solv*/
    case 12:
      strcat(ext,".mtx");
      size1 = strlen(name);
      size2 = strlen(ext);
      if( (size1+size2)  > SIZEMAX){
        fprintf(stderr,"Nome do arquivo muito extenso.\n"
	               "name : \"%s\"\n"
		       "Name maximo : %d\n"
		       "Funcao %s, arquivo fonte \"%s\"\n" 
		       ,name,SIZEMAX,__func__,__FILE__);
	exit(EXIT_FAILURE);      
      }
      strcpy(*out,name);
      strcat(*out,ext);
      break;
/*...................................................................*/

/*...*/
    default:
        printf("\n opcao invalida\n"
             "funcao %s(*,*,*)\narquivo = %s\n",__func__,__FILE__);
        exit(EXIT_FAILURE);
      break;
/*...................................................................*/
  }
/*===================================================================*/
}
/*********************************************************************/

/*********************************************************************
 * ito:pegar o valor de um interio e trasnforma em uma cadeia de     *
 * caracter.                                                         *
 *********************************************************************
 * ------------------------------------------------------------------*
 * Parametros de entrada:                                            *
 * ------------------------------------------------------------------*
 * istep -> posso no tempo                                           *
 * ------------------------------------------------------------------*
 * Paramanetros de saida:                                            *
 * ------------------------------------------------------------------*
 * st -> string com a volar de istep                                 *
 * ------------------------------------------------------------------*
 *********************************************************************/
void iota( INT t , char* st ){
/*...*/
  char c;
  int n , dec , j , k;
/*...................................................................*/
/**/
/*...*/
  n = ( t > 0 )? (int)( log10((double)t) + 1 ): 1;
  st[n] = '\0'; 
/*...................................................................*/
/**/
/*...*/
  dec = 10 ;   
  for( j = n - 1 ; j >= 0 ; --j ){
    k = ( t % dec ) / ( dec / 10 );
    c = (char)(k + '0');
    st[j] = c;
    dec = dec * 10;
  }
/*...................................................................*/
}
/*********************************************************************/ 


/*********************************************************************
 * READMACRO: le uma carcteres de um arquivo                         *
 *********************************************************************
 * ------------------------------------------------------------------*
 * Parametros de entrada:                                            *
 * ------------------------------------------------------------------*
 * mc     -> string de caracter                                      *
 * allline-> leitura da linha toda                                   *
 * ------------------------------------------------------------------*
 * Paramanetros de saida:                                            *
 * ------------------------------------------------------------------*
 * mc -> string com lida                                             *
 * ------------------------------------------------------------------*
 *********************************************************************/
void readMacro(FILE* file,char *mc,bool allline)
{
 static char line[LINE_SIZE];
 static char word[WORD_SIZE];
 char c;
 int z;

#ifdef _DEBUG_ 
 printf("DEGUB INFO: Antes funcao readmacro.\n"
        "linha  : \"%s\"\n",line);
#endif

 /*lendo um linha inteira*/

 if(allline){
  z = 0;
  while((c=getc(file))!='\n' && z<WORD_SIZE-1 && c!=EOF){
    line[z++] = c;
  }
  line[z]='\0';
  strcpy(mc,line);
 }
 
 /*lendo apenas um paralvra*/
 else{
   fscanf(file,"%s",word);
   sscanf(word,"%s",mc);
 }   

#ifdef _DEBUG_ 
 printf("DEGUB INFO: Depois funcao readmacro.\n"
        "linha  : \"%s\"\n"
        "palavra: \"%s\"\n",line,mc);
#endif
}
/*********************************************************************/

int rl(FILE *f,char *st){
  
  long posicao;
  char s[LINE_SIZE];
  
  posicao=ftell(f);
  clearLine(s);
  readMacro(f,s,false);

#ifdef _DEBUG_ 
   fprintf(stderr,"\nLinha lida\nlinha=!%s!",s);
   fprintf(stderr,"\nMACRO procurada   !%s!",st);
#endif
/*...*/  
  if(!strcmp(s,st)){
/*-------------------------------------------------------------------*/
#ifdef _DEBUG_ 
      fprintf(stderr,"\nMacro !%s! achada\n",st); 
#endif
/*-------------------------------------------------------------------*/
    return 0;
  }
/*====== =============================================================*/
 /*...*/ 
  else{
/*-------------------------------------------------------------------*/
#ifdef _DEBUG_ 
     fprintf(stderr,"\nRetonando linha");
#endif
/*-------------------------------------------------------------------*/
     fseek(f,posicao,0);
/*-------------------------------------------------------------------*/
#ifdef _DEBUG_ 
     fprintf(stderr,"\nLinha retrocedida");
#endif
/*-------------------------------------------------------------------*/
  return 1;
  }
/*===================================================================*/
}
/*********************************************************************/
void clearLine(char *s){
   unsigned int i;
   i=0;
/*-------------------------------------------------------------------*/
#ifdef _DEBUG_ 
           fprintf(stderr,"Funcao clear_line\n");
#endif
/*-------------------------------------------------------------------*/
   while(i<WORD_SIZE-1)
       s[i++]='\0';         
}

