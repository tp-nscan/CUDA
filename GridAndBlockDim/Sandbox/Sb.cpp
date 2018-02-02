// Sandbox.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


struct Rdb {
	int arrayLength;
	int reps;
	int seed;
	float totalTime;
	float frames;
};


char *RdbToStr(Rdb *t);

double getit(double in)
{
	return 2.0;
}

float getit(float in)
{
	return 0.5;
}

template <typename Real>
int TT()
{
	Real in = 0;
	Real x = getit(in);
	if (x > 1) return 1;
	return 0;
}

void PrintIntArray(int *aa, int width, int  length)
{
	for (int i = 0; i < length; i++) {
			printf("%d ", aa[i]);
			if (i % width == 0) printf("\n");
		}
	printf("\n");
}

//int main()
//{
//	Rdb Tom;
//	Tom.arrayLength = 55;
//	Tom.reps = 1000;
//	Tom.totalTime = 55.55533;
//	char *yow = RdbToStr(&Tom);
//	printf("Ta da: %s", yow);
//    return 0;
//}
//
//char *RdbToStr(Rdb *rdb)
//{
//	char buffer[256] = { '\0' };
//	strcat_s(buffer, "a");
//	strcat_s(buffer, "a and b");
//	sprintf_s(buffer,"%d %d %f", sizeof(buffer), rdb->arrayLength, rdb->reps, rdb->totalTime);
//	return buffer;
//}

// Example program to demonstrate sprintf()
int main()
{
	char buffer[50];
	int a = 10, b = 20, c;
	c = a + b;
	sprintf_s(buffer, "Sum of %d and %d is %d", a, b, c);

	// The string "sum of 10 and 20 is 30" is stored 
	// into buffer instead of printing on stdout
	printf("%s", buffer);

	return 0;
}

//int main(void)
//
//{
//	char character = 'A';
//
//	char string[] = "This is a string";
//
//	const char *stringPtr = "This is also a string";
//
//	printf("---------------------------------\n");
//
//	printf("---Character and String format---\n");
//
//	printf("---------------------------------\n\n");
//
//	printf("%c <--This one is character\n", character);
//
//	printf("\nLateral string\n");
//
//	printf("%s\n", "This is a string");
//
//	printf("\nUsing array name, the pointer to the first array's element\n");
//
//	printf("%s\n", string);
//
//	printf("\nUsing pointer, pointing to the first character of string\n");
//
//	printf("%s\n", stringPtr);
//	return 0;
//}