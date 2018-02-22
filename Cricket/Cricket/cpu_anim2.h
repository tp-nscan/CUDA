/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and
 * proprietary rights in and to this software and related documentation.
 * Any use, reproduction, disclosure, or distribution of this software
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA)
 * associated with this source code for terms and conditions that govern
 * your use of this NVIDIA software.
 *
 */


#ifndef __CPU_ANIM_H__
#define __CPU_ANIM_H__

#include "gl_helper.h"
#include <iostream>

struct CPUAnimBitmap;

struct GridPlot
{
	unsigned int   zoom;
	unsigned int   imageWidth;
	unsigned int   xStart;
	unsigned int   yStart;
};

struct ControlBlock {
	void    *appBlock;
	char command[20];
	unsigned char   *device_bitmap;
	CPUAnimBitmap   *cPUAnimBitmap;
	GridPlot        *gridPlot;
	int2             winSize;
};

struct CPUAnimBitmap {
    unsigned char    *pixels;
    int     width, height;
	ControlBlock    *controlBlock;
    void (*fAnim)(void*,int);
    void (*animExit)(void*);
    void (*clickDrag)(void*,int,int,int,int);
    int     dragStartX, dragStartY;
	char commandBuff[20];

    CPUAnimBitmap() {
    }

	void Init(int width, ControlBlock *cb)
	{
		controlBlock = cb;
		CPUAnimBitmap**   cpBm = get_bitmap_ptr();
		*cpBm = this;
		cb->cPUAnimBitmap = this;
		this->width = controlBlock->winSize.x;
		this->height = controlBlock->winSize.y;
		pixels = new unsigned char[width * height * 4];
		clickDrag = NULL;
		ClearCommandBuff();
	}

    ~CPUAnimBitmap() {
        delete [] pixels;
    }

	void SubmitCommand() {

		strcpy_s(controlBlock->command, 20, commandBuff);
		commandBuff[0] = '\0';
	}

	void ClearCommand() {
		controlBlock->command[0] = '\0';
	}

	void ClearCommandBuff() {
		commandBuff[0] = '\0';
	}

	void AppendCharToCommandBuff(char c) {
		int curLen = strlen(commandBuff);
		commandBuff[curLen] = c;
		commandBuff[curLen + 1] = '\0';
	}

    unsigned char* get_ptr( void ) const   { return pixels; }
    long image_size( void ) const { return width * height * 4; }

    void click_drag( void (*f)(void*,int,int,int,int)) {
        clickDrag = f;
    }

    void anim_and_exit( void (*f)(void*,int), void(*e)(void*) ) {

        fAnim = f;
        animExit = e;
        // a bug in the Windows GLUT implementation prevents us from
        // passing zero arguments to glutInit()
        int c=1;
        char* dummy = "";
        glutInit( &c, &dummy );
        glutInitDisplayMode( GLUT_DOUBLE | GLUT_RGBA );
        glutInitWindowSize( width, height );
        glutCreateWindow( "bitmap" );
        glutKeyboardFunc(Key);
        glutDisplayFunc(Draw);
        if (clickDrag != NULL)
            glutMouseFunc( mouse_func );
        glutIdleFunc( idle_func );
        glutMainLoop();
    }

    // static method used for glut callbacks
    static CPUAnimBitmap** get_bitmap_ptr( void ) {
        static CPUAnimBitmap*   gBitmap;
        return &gBitmap;
    }

    // static method used for glut callbacks
    static void mouse_func( int button, int state,
                            int mx, int my ) {
        if (button == GLUT_LEFT_BUTTON) {
            CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
            if (state == GLUT_DOWN) {
                bitmap->dragStartX = mx;
                bitmap->dragStartY = my;
            } else if (state == GLUT_UP) {
                bitmap->clickDrag( bitmap->controlBlock,
                                   bitmap->dragStartX,
                                   bitmap->dragStartY,
                                   mx, my );
            }
        }
    }

    // static method used for glut callbacks
    static void idle_func( void ) {
        static int ticks = 1;
        CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
        bitmap->fAnim( bitmap->controlBlock, ticks++ );
        glutPostRedisplay();
    }

	static void Key(unsigned char key, int x, int y) {
		CPUAnimBitmap *cPUAnimBitmap = *(get_bitmap_ptr());
		ControlBlock *controlBlock = cPUAnimBitmap->controlBlock;
		if (key == 27) {
			cPUAnimBitmap->animExit(controlBlock);
			exit(0);
		}
		if (key == 13) {
			cPUAnimBitmap->SubmitCommand();
		}
		else
		{
			cPUAnimBitmap->AppendCharToCommandBuff(key);
		}
	}

    static void Draw( void ) {
        CPUAnimBitmap*   bitmap = *(get_bitmap_ptr());
        glClearColor( 0.0, 0.0, 0.0, 1.0 );
        glClear( GL_COLOR_BUFFER_BIT );
        glDrawPixels( bitmap->width, bitmap->height, GL_RGBA, GL_UNSIGNED_BYTE, bitmap->pixels );
        glutSwapBuffers();
    }
};


#endif  // __CPU_ANIM_H__

