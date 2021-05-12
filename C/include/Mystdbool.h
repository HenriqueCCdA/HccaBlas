#ifndef _MYSTDBOOL_H
  #if WIN32
    #define bool short
    #define true  1
    #define false 0 
  #else
    #include<stdbool.h>
  #endif
#endif