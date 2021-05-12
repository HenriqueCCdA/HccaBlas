#!/bin/sh
PATH_INCLUDE="./include"    
PATH_LIB="./lib"    
PATH_SCR="src"    
PRENAME=MyBlas
CC=icc
OPENMP=yes
OT=O3
OTBLAS=O3
DEBUG=no
BLAS=MKL
CUDA=yes
#------------------gerando o nome do excutavel-------------
ifeq ($(CC),icc)
  ifeq ($(OPENMP),yes)
# ... GSL
    ifeq ($(BLAS),GSL)
      COMPILER_NAME=intel_omp_gslBlas_$(OT)
    endif
# ... ATLAS
    ifeq ($(BLAS),ATLAS)
      COMPILER_NAME=intel_omp_atlasBlas_$(OT)
    endif
# ... OPENBLAS
    ifeq ($(BLAS),OPENBLAS)
      COMPILER_NAME=intel_omp_openBlas_$(OT)
    endif
# ... MKL      
    ifeq ($(BLAS),MKL)
      COMPILER_NAME=intel_omp_mkl_$(OT)
    endif
  else
# ... GSL
    ifeq ($(BLAS),GSL)
      COMPILER_NAME=intel_gslBlas_$(OT)
    endif
# ... ATLAS
    ifeq ($(BLAS),ATLAS)
      COMPILER_NAME=intel_atlasBlas_$(OT)
    endif
# ... OPENBLAS
    ifeq ($(BLAS),OPENBLAS)
      COMPILER_NAME=intel_openBlas_$(OT)
    endif
  endif  
endif

ifeq ($(CC),gcc)
  ifeq ($(OPENMP),yes)
# ... GSL
    ifeq ($(BLAS),GSL)
      COMPILER_NAME=gnu_omp_gslBlas_$(OT)
# ... ATLAS
    else ifeq ($(BLAS),ATLAS)
      COMPILER_NAME=gnu_omp_atlasBlas_$(OT)
# ... OPENBLAS
    else ifeq ($(BLAS),OPENBLAS)
      COMPILER_NAME=gnu_omp_openBlas_$(OT)
    else ifeq ($(BLAS),MKL)
      COMPILER_NAME=gnu_omp_mkl_$(OT)
    else
      COMPILER_NAME=gnu_omp_$(OT)
    endif
  else
# ... GSL
    ifeq ($(BLAS),GSL)
      COMPILER_NAME=gnu_gslBlas_$(OT)
    endif
# ... ATLAS
    ifeq ($(BLAS),ATLAS)
      COMPILER_NAME=gnu_atlasBlas_$(OT)
    endif
# ... OPENBLAS
    ifeq ($(BLAS),OPENBLAS)
      COMPILER_NAME=gnu_openBlas_$(OT)
    endif
  endif  
endif
  
#--------------------------GLS-----------------------------

NAME+=$(PRENAME)_$(COMPILER_NAME)
#-------------------Fontes--------------------------------
fontes = \
src/Csr.c\
src/File.c\
src/Graph.c\
src/Main.c\
src/Memoria.c\
src/Random.c\
src/Time.c\
src/WriteMtx.c

fontesNvcc=\
src/InterfaceCuSparse.c

fontesNvccK=\
src/CudaHccaBlas.cu

fontesBlas=\
src/HccaBlas.c 

#-------------------Flags necessarios--------------------------------
NFLAGS+=-I$(PATH_INCLUDE) -L$(PATH_LIB)\
       -D_MMIO_  
LDFLAGS=-lmmio

NFLAGSBLAS= -I$(PATH_INCLUDE) 

#--------------------compiladores------------------------------------

ifeq ($(CUDA),yes)
  NFLAGS   +=-D_CUDABLAS_ 
  LDFLAGS  +=-L/usr/local/cuda-6.0/lib64/ -lcudart -lcusparse
  NFLAGSNVCC=-D_CUDABLAS_ -I$(PATH_INCLUDE) -gencode arch=compute_30,code=sm_30\
						-use_fast_math --ptxas-options=-v 
  LDFLAGSNVCC=
endif

# intel icc  
ifeq ($(CC),icc)
  LDFLAGS +=                           
  OFLAGS  += -Werror-all -std=c99 -ansi 
  OFLAGSBLAS  +=  -Werror -ansi -std=c99 -funroll-all-loops\
              -debug inline-debug-info -vec-report2 -opt-report-file=LogOt_intel.txt
  ifeq ($(OPENMP),yes)
    OFLAGS      += -openmp
    OFLAGSBLAS  += -openmp 
  endif
#mkl      
  ifeq ($(BLAS),MKL)
    OFLAGS  += -D_MKL_ -I/opt/intel/mkl/include
    LDFLAGS +=  -mkl
  endif
#GSL_BLAS
  ifeq ($(BLAS),GSL)
    OFLAGS  += -D_GSL_BLAS_
    LDFLAGS +=  -lgslclbas
  endif
#ATLAS
  ifeq ($(BLAS),ATLAS)
    OFLAGS  += -D_ATLAS_ -I/opt/atlas/include -L/opt/atlas/lib
    LDFLAGS +=  -lcblas -latlas
  endif
#OPENBLAS
  ifeq ($(BLAS),OPENBLAS)
    OFLAGS  += -D_OPENBLAS_ -I/opt/openBlas/gnu/include\
							-L/opt/openBlas/gnu/lib
    LDFLAGS += -lopenblas
  endif
endif
# gnu gcc
ifeq ($(CC),gcc)
  LDFLAGS     +=  -lm 
  OFLAGS      +=  -Wall -ansi -std=c99 -pedantic-errors\
                  -fopt-info-optimized-missed=logOpt.txt
  OFLAGSBLAS  +=  -Wall -ansi -std=c99 -funroll-all-loops\
							-fopt-info-optimized-missed=logOpt.txt
  ifeq ($(OPENMP),yes)
    OFLAGS      += -fopenmp
    OFLAGSBLAS  += -fopenmp 
  endif
#mkl      
  ifeq ($(BLAS),MKL)
    OFLAGS  += -D_MKL_ -I/opt/intel/mkl/include\
							 -L/opt/intel/mkl/lib/intel64
    LDFLAGS += -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core\
							 -liomp5  
  endif
#GSL_BLAS
  ifeq ($(BLAS),GSL)
    OFLAGS  += -D_GSL_BLAS_
    LDFLAGS +=  -lgslcblas
  endif
#ATLAS
  ifeq ($(BLAS),ATLAS)
    OFLAGS  += -D_ATLAS_ -I/opt/atlas/include -L/opt/atlas/lib
    LDFLAGS +=  -lcblas -latlas
  endif
#OPENBLAS
  ifeq ($(BLAS),OPENBLAS)
    OFLAGS  += -D_OPENBLAS_ -I/opt/openBlas/gnu/include\
						-L/opt/openBlas/gnu/lib
    LDFLAGS += -l:libopenblas.a
  endif
endif
#--------------------------------------------------------------------
#---------------------------Debug------------------------------------
ifeq ($(DEBUG),yes)
  OFLAGS     += -g -$(OT)
  OFLAGSBLAS += -g -$(OTBLAS)
else
  OFLAGS     += -$(OT) 
  OFLAGSBLAS += -$(OTBLAS) 
endif
#--------------------------------------------------------------------

.SUFFIXES: 
.SUFFIXES: .c .cu .h .o
CFLAGS    = $(NFLAGS) $(OFLAGS) 
CFLAGSBLAS= $(NFLAGSBLAS) $(OFLAGSBLAS) 
CFLAGSNVCC= $(NFLAGSNVCC) $(OFLAGSNVCC) 

OBJS     = $(patsubst %.c,%.o ,$(fontes))
OBJSBLAS = $(patsubst %.c,%.o ,$(fontesBlas))
OBJSNVCC = $(patsubst %.c,%.o ,$(fontesNvcc))
OBJSNVCCK= $(patsubst %.cu,%.o,$(fontesNvccK))

ifeq ($(CUDA),yes)
build:	$(OBJS) $(OBJSBLAS) $(OBJSNVCC) $(OBJSNVCCK) 
	  ls bin || mkdir -p bin
	  $(CC) $(CFLAGS) $(OBJS) $(OBJSBLAS) $(OBJSNVCC) $(OBJSNVCCK) -o bin/$(NAME)  $(LDFLAGS)
else
build:	$(OBJS) $(OBJSBLAS) 
	  ls bin || mkdir -p bin
	  $(CC) $(CFLAGS) $(OBJS) $(OBJSBLAS) -o bin/$(NAME)  $(LDFLAGS)
endif

$(OBJS): %.o:%.c
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJSNVCC): %.o:%.c 
	nvcc $(CFLAGSNVCC) -x c -c $< -o $@

$(OBJSNVCCK): %.o:%.cu 
	nvcc $(CFLAGSNVCC) -x cu -c $< -o $@


$(OBJSBLAS): %.o:%.c
	$(CC) $(CFLAGSBLAS) -c $< -o $@

tags:
	ctags -R src/*.c include/*.h

.PHONY: cleantags
cleantags:
	@rm -fv tags
	
.PHONY: clean
clean:  
	@rm -fv src/*.o
	@rm -fv ../src/*.o
	@rm -fv bin/$(NAME)

.PHONY: cleanall
cleanall:  
	@rm -fv tags
	@rm -fv src/*.o
	@rm -fv ../src/*.o
	@rm -fv bin/$(NAME)


.PHONY: help
help:
	@echo "Autor :$(AUTHOR)                              "
	@echo "Makefile para prepar para sitemas linux.      "
	@echo -e "\E[7;32mOpcoes:\E[1;0m                      "
	@echo "build         - compila o prepar              "
	@echo "build_modules - gera os modulos               "
	@echo "tags          - gera os tags                  "
	@echo "cleantags     - limpa os tags                 "
	@echo "clean         - limpa os obj, bin e mod       "
	@echo "cleaall       - limpa tudo obj,bin,mod e tags "

# DO NOT DELETE

# DO NOT DELETE
