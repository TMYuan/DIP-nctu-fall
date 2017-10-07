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
/*                                                                         */
/***************************************************************************/
#include <stdlib.h>
#include <stdio.h>

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

void headerinfo(bmpheader *hbmp){
    printf("identifier:          %hu\n", hbmp->identifier);
    printf("filesize:            %u\n", hbmp->filesize);
    printf("bitmap_dataoffset:   %u\n", hbmp->bitmap_dataoffset);
    printf("bitmap_headersize:   %u\n", hbmp->bitmap_headersize);
    printf("width:               %u\n", hbmp->width);
    printf("height:              %u\n", hbmp->height);
    printf("planes:              %hu\n", hbmp->planes);
    printf("bits_perpixel:       %hu\n", hbmp->bits_perpixel);
    printf("compression:         %u\n", hbmp->compression);
    printf("bitmap_datasize:     %u\n", hbmp->bitmap_datasize);
    printf("hresolution:         %u\n", hbmp->hresolution);
    printf("vresolution:         %u\n", hbmp->vresolution);
    printf("usedcolors:          %u\n", hbmp->usedcolors);
    printf("importantcolors:     %u\n", hbmp->importantcolors);
}


int main(int argc, char* argv[]){
    unsigned char* bmpimage, *palette;
    bmpheader hbmp;
    char* input_name, *output_name;
    input_name = argv[1];
    output_name = argv[2];

    readbmp(input_name, &hbmp, 0, palette, bmpimage);
    bmpimage = malloc(sizeof(unsigned char)*hbmp.bitmap_datasize);
    palette = malloc(sizeof(unsigned char)*(hbmp.bitmap_dataoffset - sizeof(bmpheader)));
    readbmp(input_name, &hbmp, 1, palette, bmpimage);
    headerinfo(&hbmp);
    writebmp(output_name, &hbmp, palette, bmpimage);

    return 0;
}
