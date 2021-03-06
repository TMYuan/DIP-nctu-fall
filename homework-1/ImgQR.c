/***************************************************************************/
/*  Digital Image Processing (2017)                                        */
/*  Homework 1                                                             */
/*  Author: Min Yuan Tseng                                                 */
/*  Student ID: 0310139                                                    */
/*                                                                         */
/*  There are three tasks to finish:                                       */
/*  1. Image Input/Output                                                  */
/*  2. Resolution                                                          */
/*  3. Scaling                                                             */
/*                                                                         */
/*  For this homework, I modify some code for BMP reader/writer from:      */
/*  http://capricorn-liver.blogspot.tw/2010/11/cbmp.html                   */
/*                                                                         */
/*  In this file, there are some same function and operation in            */
/*  "ImgRWbmp.c", so I would not add comment in this file for them.        */
/***************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define STEP_SIZE_1 32
#define STEP_SIZE_2 64
#define STEP_SIZE_3 128

typedef struct _bmpheader{
    unsigned short identifier;      // 0x0000
    unsigned int filesize;          // 0x0002
    unsigned int reserved;          // 0x0006
    unsigned int bitmap_dataoffset; // 0x000A
    unsigned int bitmap_headersize; // 0x000E
    unsigned int width;             // 0x0012
    unsigned int height;            // 0x0016
    unsigned short planes;          // 0x001A
    unsigned short bits_perpixel;   // 0x001C
    unsigned int compression;       // 0x001E
    unsigned int bitmap_datasize;   // 0x0022
    unsigned int hresolution;       // 0x0026
    unsigned int vresolution;       // 0x002A
    unsigned int usedcolors;        // 0x002E
    unsigned int importantcolors;   // 0x0032
} __attribute__((packed,aligned(1))) bmpheader;


int readbmp(unsigned char* filename, bmpheader* hbmp, int mode, unsigned char* palette, unsigned char* buffer){
    FILE* ifp;
    char c[128];
    unsigned char* ptr;

    sprintf(c, "./data/%s", filename);
    ifp = fopen(c, "rb");
    if(ifp==NULL){
        printf("readbmp: file open error\n");
        return -1;
    }
    
    ptr = (unsigned char *) hbmp;
    fread(ptr, sizeof(unsigned char), sizeof(bmpheader), ifp);

    if(mode==1){
        fread(palette, sizeof(unsigned char), hbmp->bitmap_dataoffset - sizeof(bmpheader), ifp);
        fread(buffer, sizeof(unsigned char), hbmp->bitmap_datasize, ifp);
    }
    else{
        fclose(ifp);
        return 1;
    }

    fclose(ifp);
    return 1;
}

int writebmp(unsigned char* filename, bmpheader* hbmp, unsigned char* palette, unsigned char* buffer){
    FILE* ofp;
    char c[128];
    unsigned char* ptr;

    sprintf(c, "./data/%s", filename);
    ofp = fopen(c, "wb");
    if(ofp==NULL){
        printf("writebmp: file open error\n");
        return -1;
    }

    ptr = (unsigned char *)hbmp;
    fwrite(ptr, sizeof(unsigned char), sizeof(bmpheader), ofp);
    fwrite(palette, sizeof(unsigned char), hbmp->bitmap_dataoffset - sizeof(bmpheader), ofp);
    fwrite(buffer, sizeof(unsigned char), hbmp->bitmap_datasize, ofp);

    fclose(ofp);
    return 1;
}

// This function is to perform resolution.
// The level depend on variable "bit", for bit=1, level=2,
// bit=3, level=8, bit=6, level=64 ...
// It calculate step_size first, then use it to operate resolution.
void resolution(unsigned char *buffer, unsigned char bit, unsigned int buffer_size, unsigned char *bmpimage_re){
    unsigned int i=0;
    double step_size;
    step_size = 255/(pow(2, bit)-1);
    for(i=0; i<buffer_size; i++){
        bmpimage_re[i] = round(buffer[i]/step_size) * step_size;
    }
}


int main(int argc, char* argv[]){
    unsigned char *bmpimage, *palette, *bmpimage_re;
    bmpheader hbmp;
    char* input_name, *output_name_1, *output_name_2, *output_name_3;
    input_name = argv[1];
    output_name_1 = argv[2];
    output_name_2 = argv[3];
    output_name_3 = argv[4];

    // Use "bmpimage_re" to store bmp array after resolution
    readbmp(input_name, &hbmp, 0, palette, bmpimage);
    bmpimage = malloc(sizeof(unsigned char)*hbmp.bitmap_datasize);
    bmpimage_re = malloc(sizeof(unsigned char)*hbmp.bitmap_datasize);
    palette = malloc(sizeof(unsigned char)*(hbmp.bitmap_dataoffset - sizeof(bmpheader)));
    readbmp(input_name, &hbmp, 1, palette, bmpimage);

    // 2 level(1 bit) resolution
    resolution(bmpimage, 1, hbmp.bitmap_datasize, bmpimage_re);
    writebmp(output_name_1, &hbmp, palette, bmpimage_re);

    // 8 level(3 bit) resolution
    resolution(bmpimage, 3, hbmp.bitmap_datasize, bmpimage_re);
    writebmp(output_name_2, &hbmp, palette, bmpimage_re);

    // 64 level(6 bit) resolution
    resolution(bmpimage, 6, hbmp.bitmap_datasize, bmpimage_re);
    writebmp(output_name_3, &hbmp, palette, bmpimage_re);

    return 0;
}
