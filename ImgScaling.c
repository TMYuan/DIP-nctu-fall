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
#include <math.h>

#define SCALING_RATE 0.67

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
} __attribute__((packed,aligned(2))) bmpheader;

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
        fread(buffer, sizeof(unsigned char), hbmp->height*hbmp->width*(hbmp->bits_perpixel/8), ifp);
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
    fwrite(buffer, sizeof(unsigned char), hbmp->height*hbmp->width*(hbmp->bits_perpixel/8), ofp);

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

unsigned int getindex(double x, double y, unsigned int width, unsigned int pixel_color, unsigned int color_num){
    return color_num*(x + y * width)+pixel_color;
}

unsigned int interpolation(double x, double y, double x_small, double x_large, double y_small, \
                        double y_large, unsigned char* buffer, unsigned int height, unsigned int width, \
                        unsigned int pixel_color, unsigned color_num){
    double up_left, up_right, down_left, down_right, f1, f2, f3;
    up_left = buffer[getindex(x_small, y_small, width, pixel_color, color_num)];
    up_right = buffer[getindex(x_large, y_small, width, pixel_color, color_num)];
    down_left = buffer[getindex(x_small, y_large, width, pixel_color, color_num)];
    down_right = buffer[getindex(x_large, y_large, width, pixel_color, color_num)];
    f1 = up_left + (x - x_small) * (up_right - up_left);
    f2 = down_left + (x - x_small) * (down_right - down_left);
    f3 = f1 + (y - y_small) * (f2 - f1);
}

void scale_up(bmpheader hbmp, unsigned char* buffer, unsigned char* image_up){
    unsigned int color_num, i=0, size_up, height, width, height_up, width_up;
    color_num = hbmp.bits_perpixel/8;
    height = hbmp.height;
    width = hbmp.width;
    height_up = height * SCALING_RATE;
    width_up = ((width * SCALING_RATE)/4)*4;
    size_up = height_up * width_up * color_num;

    for(i=0; i<size_up; i++){
        unsigned int pixel, pixel_height, pixel_width, pixel_color;
        double pixel_height_ori, pixel_width_ori, x_small, x_large, y_small, y_large;
        pixel = i / color_num;
        pixel_color = i % color_num;
        pixel_height = pixel / width_up;
        pixel_width = pixel % width_up;
        pixel_height_ori = pixel_height / SCALING_RATE;
        pixel_width_ori = pixel_width / SCALING_RATE;
        x_small = floor(pixel_width_ori);
        x_large = ceil(pixel_width_ori);
        y_small = floor(pixel_height_ori);
        y_large = ceil(pixel_height_ori);
        image_up[i] = interpolation(pixel_width_ori, pixel_height_ori, x_small, x_large, \
                                    y_small, y_large, buffer, height, width, pixel_color, color_num);
    }
}


int main(int argc, char* argv[]){
    unsigned char *bmpimage, *palette, *bmpimage_up, *bmpimage_down;
    bmpheader hbmp, hbmp_up;
    char* input_name, *output_name;
    input_name = argv[1];
    output_name = argv[2];

    readbmp(input_name, &hbmp, 0, palette, bmpimage);
    bmpimage = malloc(sizeof(unsigned char)*hbmp.width*hbmp.height*(hbmp.bits_perpixel/8));
    palette = malloc(sizeof(unsigned char)*(hbmp.bitmap_dataoffset - sizeof(bmpheader)));
    readbmp(input_name, &hbmp, 1, palette, bmpimage);
    headerinfo(&hbmp);

    bmpimage_up = malloc(sizeof(unsigned char)*\
        ((hbmp.width*SCALING_RATE)/4)*4*hbmp.height*(hbmp.bits_perpixel/8)*SCALING_RATE);
    // bmpimage_down = malloc(sizeof(unsigned char)*hbmp.bitmap_datasize/SCALING_RATE);

    scale_up(hbmp, bmpimage, bmpimage_up);
    hbmp_up = hbmp;
    hbmp_up.height *= SCALING_RATE;
    hbmp_up.width *= SCALING_RATE;
    hbmp_up.bitmap_datasize = hbmp_up.height * hbmp_up.width * (hbmp.bits_perpixel/8);
    hbmp_up.filesize = hbmp_up.bitmap_datasize + hbmp.bitmap_dataoffset;
    headerinfo(&hbmp_up);

    writebmp(output_name, &hbmp_up, palette, bmpimage_up);

    return 0;
}
