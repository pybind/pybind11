/*
    example/example0.cu -- CUDA acceleratred Mandelbrot example

    Copyright (c) 2012-2016 Axel Huebl <a.huebl@hzdr.de>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#include "example.h"

#include <cstdio>
#include <cmath>
#include <complex.h>

#include "cuda.h"

// simulation parameters
const int max_iterations = 255;
const int num_cols = 1000;
const int num_rows = 2000;

// cuda parameters
size_t blocksize = 256; // threads per block
const int maxRam = 250; // ION has approx. 256 MB global RAM


// Complex Numbers
struct cuComplex {
  float r;
  float i;
  __device__ cuComplex( float a, float b ) : r(a), i(b) {}
  __device__ float magnitude2( void ) {
    return r * r + i * i;
  }
  __device__ cuComplex operator*(const cuComplex& a) {
    return cuComplex(r*a.r - i*a.i, i*a.r + r*a.i);
  }
  __device__ cuComplex operator*(const float& a) {
    return cuComplex(r*a, i*a);
  }
  __device__ cuComplex operator+(const cuComplex& a) {
    return cuComplex(r+a.r, i+a.i);
  }
  __device__ cuComplex operator+(const float& a) {
    return cuComplex(r+a, i);
  }
};

__device__ int iterate(cuComplex c ) {
        cuComplex z(0., 0.);
        int iterations = 0;
        bool val = true;
        for( int i=0; i<max_iterations; i++ ) {
                if ( val )
                    z = z*z + c;
                const bool tmp = (sqrtf(z.magnitude2() ) > 2.0f);
                val = (val && !tmp);
                if( val )
                  ++iterations;
        }
        return iterations;
}

__global__ void calcMandelbrot( int* color_d, const int num_rows, const int num_cols ) {

    const int globalX = ( blockIdx.x * blockDim.x ) + threadIdx.x;
    const int globalY = blockIdx.y;
    const int offset  = globalY * num_cols + globalX;

    // parameters
    const float c_rmin = -2.0;
    const float c_rmax = +1.0;
    const float c_imin = -1.0;
    const float c_imax = +1.0;

    const float dx = (c_rmax - c_rmin) / float(num_cols);
    const float dy = (c_imax - c_imin) / float(num_rows);

    cuComplex imaginary( 0., 1.);

    if( globalY < num_rows && globalX < num_cols ) {
      cuComplex c = ( imaginary*( c_imin+(float(globalY)*dy) ) ) + (c_rmin+(float(globalX)*dx));
      color_d[offset] = iterate(c);
    }

}

int mandelbrot() {

    FILE *output = fopen("mandelbrot.ppm", "w+b");

        int *color_h, *color_d;
        const int nBytes = num_rows*num_cols*sizeof(int);

        const int globalMem = nBytes / 1024 / 1024; // in MiB
        printf( "Will use %d MiB of global Memory...\n", globalMem );

        if( globalMem > maxRam ) {
           printf( "Maximum RAM is %d ... exit now...\n", maxRam);
           return 1;
        }

        // allocate host memory
        color_h = (int*)malloc(nBytes);
        // allocate device memory
        cudaMalloc( (void**)&color_d, nBytes );

        // init host
        for( int i=0; i<num_cols*num_rows; i++ ) color_h[i] = 0;
        // copy to device
        cudaMemcpy(color_d, color_h, nBytes, cudaMemcpyHostToDevice);
        printf( "Copied Memory to Device...\n" );

        // call kernel
        // dimension and size of grid *in blocks* (2D)
        dim3 grid( ceil( double(num_cols)/double(blocksize) ), num_rows );
        printf( "Grid size in blocks: %d %d\n", grid.x, grid.y );
        // dimension and size of blocks *in threads* (3D)
        dim3 threads( blocksize );

        // asynchroner (!!) funktionsaufruf!
        calcMandelbrot<<<grid, threads>>>( color_d, num_rows, num_cols );
        printf( "%s\n", cudaGetErrorString( cudaGetLastError() ) );


        // copy to host
        cudaMemcpy(color_h, color_d, nBytes, cudaMemcpyDeviceToHost);
        printf( "Copied Memory back to Host...\n" );


    fprintf(output, "P3\n");
    fprintf(output, "%d %d\n%d\n\n", num_cols, num_rows, max_iterations);

    for (int x=0; x<num_cols; x++) {
        for (int y=0; y<num_rows; y++) {
            // float complex imaginary = 0+1.0i;
            // c = (c_rmin+(x*dx)) + ((c_imin+(y*dy))*imaginary);

            // color = iterate(c);
            fprintf(output, "%d\n", color_h[x*num_rows + y]);
            fprintf(output, "%d\n", color_h[x*num_rows + y]);
            fprintf(output, "%d\n\n", color_h[x*num_rows + y]);
        }
    }

    fclose(output);

        // free host memory
        free(color_h);
        // free devide memory
        cudaFree(color_d);

    return 0;
}


namespace py = pybind11;

void init_ex0(py::module &m) {
    m.def("mandelbrot", &mandelbrot, "Start mandelbrot calculation");
}
