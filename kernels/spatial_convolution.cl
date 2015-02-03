__kernel void SpatialConvolution(
  const __global  float* input,  // 0
  __global  float* output,       // 1 
  const __global float* weights,     // 2
  const __global float* biases,      // 3
  const int input_nfeats,        // 4
  const int input_height,        // 5
  const int input_width,         // 6
  const int filt_height,         // 7
  const int filt_width) {        // 8

  const int width = get_global_size(0);
  const int height = get_global_size(1);
  const int feats = get_global_size(2);

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);
  const int xInTopLeft = x_out;
  const int yInTopLeft = y_out;

  // Initilize the output to the bias
  float sum = biases[f_out];

  const int filt_size = filt_height * filt_width;
  const int filt_size_per_fout = input_nfeats * filt_size;
  const int in_size = input_width * input_height;
  for (int f = 0; f < input_nfeats; f++) {
    // Get a pointer to the current weight matrix and input feature
    // THIS COULD BE FASTER --> STRIPE WEIGHTS MATRIX FOR BETTER DATA ACCESS!
    const __global  float* pkernel = &weights[f_out * filt_size_per_fout + f * filt_size];
    const __global  float* pinput = &input[f * in_size];

    // Perform the convolution on this input feature
    for (int r = 0; r < filt_height; r++) {
      const int idxFtmp = r * filt_width;
      const int yIn = yInTopLeft + r;
      const int idxIntmp = yIn * input_width + xInTopLeft;
      for (int c = 0; c < filt_width; c++) {
        const int idxF  = idxFtmp  + c;
        const int idxIn = idxIntmp + c;
        sum += pkernel[idxF] * pinput[idxIn];
      }
    }
  }
  const int iout = x_out + width * (y_out + height * f_out);
  output[iout] = sum;
}

__kernel void SpatialConvolutionPadding(
  const __global  float* input,   // 0
  __global  float* output,        // 1 
  const __global float* weights,  // 2
  const __global float* biases,   // 3
  const int input_nfeats,         // 4
  const int input_height,         // 5
  const int input_width,          // 6
  const int filt_height,          // 7
  const int filt_width,           // 8
  const int padding) {            // 9

  const int width = get_global_size(0);
  const int height = get_global_size(1);
  const int feats = get_global_size(2);

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);
  const int xInTopLeft = x_out;
  const int yInTopLeft = y_out;

  const int pad_left_top = padding;

  // Initilize the output to the bias
  float sum = biases[f_out];

  const int filt_size = filt_height * filt_width;
  const int filt_size_per_fout = input_nfeats * filt_size;
  const int in_size = input_width * input_height;
  for (int f = 0; f < input_nfeats; f++) {
    // Get a pointer to the current weight matrix and input feature
    // THIS COULD BE FASTER --> STRIPE WEIGHTS MATRIX FOR BETTER DATA ACCESS!
    const __global  float* pkernel = &weights[f_out * filt_size_per_fout + f * filt_size];
    const __global  float* pinput = &input[f * in_size];

    // Perform the convolution on this input feature
    for (int r = 0; r < filt_height; r++) {
      const int idxFtmp = r * filt_width;
      const int yIn = yInTopLeft + r - pad_left_top;
      const int idxIntmp = yIn * input_width;

      if (yIn >= 0 && yIn < input_height) {
        for (int c = 0; c < filt_width; c++) {
          const int idxF  = idxFtmp  + c;
          const int xIn = xInTopLeft + c - pad_left_top;
          if (xIn >= 0 && xIn < input_width) {
            const int idxIn = idxIntmp + xIn;
            sum += pkernel[idxF] * pinput[idxIn];
          }
        }
      }
    }
  }
  const int iout = x_out + width * (y_out + height * f_out);
  output[iout] = sum;
}

/*
__kernel void SpatialConvolutionPadding(
  const __global  float* input,   // 0
  __global  float* output,        // 1 
  const __global float* weights,  // 2
  const __global float* biases,   // 3
  const int input_nfeats,         // 4
  const int input_height,         // 5
  const int input_width,          // 6
  const int filt_height,          // 7
  const int filt_width,           // 8
  const int padding) {            // 9

  const int width = get_global_size(0);
  const int height = get_global_size(1);
  const int feats = get_global_size(2);
  const int pad_left_top = padding / 2;

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);
  const int xInTopLeft = x_out - pad_left_top;
  const int yInTopLeft = y_out - pad_left_top;

  // Initilize the output to the bias
  float sum = biases[f_out];

  // SpatialConvolutionMM adds padding on the OUTPUT of the convolution stage
  // (which is probably incorrect but we need to repeat that here).
  if (xInTopLeft >= 0 && (xInTopLeft + filt_width) < input_width &&
      yInTopLeft >= 0 && (yInTopLeft + filt_height) < input_height) {
    const int filt_size = filt_height * filt_width;
    const int filt_size_per_fout = input_nfeats * filt_size;
    const int in_size = input_width * input_height;
    for (int f = 0; f < input_nfeats; f++) {
      // Get a pointer to the current weight matrix and input feature
      // THIS COULD BE FASTER --> STRIPE WEIGHTS MATRIX FOR BETTER DATA ACCESS!
      const __global  float* pkernel = &weights[f_out * filt_size_per_fout + f * filt_size];
      const __global  float* pinput = &input[f * in_size];

      // Perform the convolution on this input feature
      for (int r = 0; r < filt_height; r++) {
        const int idxFtmp = r * filt_width;
        const int yIn = yInTopLeft + r;
        const int idxIntmp = yIn * input_width + xInTopLeft;
        for (int c = 0; c < filt_width; c++) {
          const int idxF  = idxFtmp  + c;
          const int idxIn = idxIntmp + c;
          sum += pkernel[idxF] * pinput[idxIn];
        }
      }
    }
  }
  const int iout = x_out + width * (y_out + height * f_out);
  output[iout] = sum;
}

*/