/* ============================================================

Copyright (c) 2009 Advanced Micro Devices, Inc.  All rights reserved.
 
COPYRIGHT NOTICE TRUNCATED BY JONATHAN... TO SAVE MEMORY STORING IT!
PLEASE SEE ORIGINAL SOURCE:
http://developer.amd.com/resources/heterogeneous-computing/opencl-zone/programming-in-opencl/image-convolution-using-opencl/image-convolution-using-opencl-a-step-by-step-tutorial-2/

============================================================ */


//KERNEL_SIMPLE
__kernel void Convolve(const __global float * pInput,
                       __constant float * pFilter,
                       __global  float * pOutput,
                       const int nInWidth,
                       const int nFilterWidth) {
	/*
	// Work on CPU
	if (get_global_id(0) == 0 && get_global_id(1) == 0) {
		printf("hello");
	}
	*/

    const int nWidth = get_global_size(0);

    const int xOut = get_global_id(0);
    const int yOut = get_global_id(1);

    const int xInTopLeft = xOut;
    const int yInTopLeft = yOut;

    float sum = 0;
    for (int r = 0; r < nFilterWidth; r++)
    {
        const int idxFtmp = r * nFilterWidth;

        const int yIn = yInTopLeft + r;
        const int idxIntmp = yIn * nInWidth + xInTopLeft;

        for (int c = 0; c < nFilterWidth; c++)
        {
            const int idxF  = idxFtmp  + c;
            const int idxIn = idxIntmp + c;
            sum += pFilter[idxF]*pInput[idxIn];
        }
    }
    const int idxOut = yOut * nWidth + xOut;
    pOutput[idxOut] = sum;
}
//KERNEL_SIMPLE
