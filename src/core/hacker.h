#ifndef VARS_H 
#define VARS_H
#include <stdio.h>

extern int hacker;

#define INCREASE do { \
hacker ++; \
printf("increased at %s line %d : counter is now %d\n", __FILE__, __LINE__, hacker); \
} while(0)

#define DECREASE do { \
hacker --; \
printf("decreased at %s line %d : counter is now %d\n", __FILE__, __LINE__, hacker); \
} while(0)
#endif 
