// CUDA: grid stride looping
//  #define CUDA_KERNEL_LOOP(i, n)                        \
//    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
//        i < (n);                                        \
//        i += blockDim.x * gridDim.x)

// For translation between CUDA and OpenCL indexing see:
// http://developer.amd.com/tools-and-sdks/opencl-zone/opencl-resources/programming-in-opencl/porting-cuda-applications-to-opencl/
#define CUDA_KERNEL_LOOP(i, n)                                        \
  for (int i = get_group_id(0) * get_local_size(0) + get_local_id(0); \
      i < (n);                                                        \
      i += get_local_size(0) * get_num_groups(0))

// Kernel for fast unfold+copy
// (borrowed from Caffe: https://github.com/BVLC/caffe/blob/master/src/caffe/layers/conv_layer.cu)
// And then I (Jonathan Tompson) took the Torch version
__kernel void im2col_kernel(const int n,                    // 0
                            const __global float* data_im,  // 1
                            const int height,               // 2
                            const int width,                // 3
                            const int ksize_h,              // 4
                            const int ksize_w,              // 5
                            const int pad_h,                // 6
                            const int pad_w,                // 7
                            const int stride_h,             // 8 
                            const int stride_w,             // 9
                            const int height_col,           // 10
                            const int width_col,            // 11
                            __global float* data_col) {     // 12
  CUDA_KERNEL_LOOP(index, n) {
    int w_out = index % width_col;
    index /= width_col;
    int h_out = index % height_col;
    int channel_in = index / height_col;
    int channel_out = channel_in * ksize_h * ksize_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    data_col += (channel_out * height_col + h_out) * width_col + w_out;
    data_im += (channel_in * height + h_in) * width + w_in;
    for (int i = 0; i < ksize_h; ++i) {
      for (int j = 0; j < ksize_w; ++j) {
        int h = h_in + i;
        int w = w_in + j;
        *data_col = (h >= 0 && w >= 0 && h < height && w < width) ?
          data_im[i * width + j] : 0;
        data_col += height_col * width_col;
      }
    }
  }
}

