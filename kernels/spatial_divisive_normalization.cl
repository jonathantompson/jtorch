__kernel void SpatialDivisiveNormalizationHoriz(
  const __global float* input,   // 0
  __global  float* output,       // 1 
  const __global float* kernel1d,    // 2
  const int filt_rad) {          // 3

  const int width = get_global_size(0);
  const int height = get_global_size(1);

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);

  // Initilize the output to zero and accumulate the input values
  float sum = 0;

  const int iout = x_out + width * (y_out + height * f_out);

  int i = 0;
  for (int u_offset = -filt_rad; u_offset <= filt_rad; u_offset++, i++) {
    int u = x_out + u_offset;
	if (u >= 0 && u < width) {
	  sum += kernel1d[i] * (input[iout + u_offset] * input[iout + u_offset]);  // Sum sqs
    }
  }

  output[iout] = sum;
}

__kernel void SpatialDivisiveNormalizationVert(
  const __global float* input,   // 0
  __global  float* output,       // 1 
  const __global float* kernel1d,    // 2
  const int filt_rad) {          // 3

  const int width = get_global_size(0);
  const int height = get_global_size(1);

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);

  // Initilize the output to zero and accumulate the input values
  float sum = 0;

  const int iout = x_out + width * (y_out + height * f_out);

  int i = 0;
  for (int v_offset = -filt_rad; v_offset <= filt_rad; v_offset++, i++) {
    int v = y_out + v_offset;
	if (v >= 0 && v < height) {
	  sum += kernel1d[i] * input[iout + v_offset * width];
    }
  }

  output[iout] = sum;
}

__kernel void SpatialDivisiveNormalizationAccumDiv(
  const __global float* input,   // 0
  __global  float* output,       // 1 
  const __global float* std_coef,    // 2
  const int input_nfeats,        // 3
  const float threshold) {       // 4

  const int width = get_global_size(0);
  const int height = get_global_size(1);

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);

  // Initilize the output to zero and accumulate the input values
  float sum = 0;

  const int uvout = x_out + width * y_out;  // index on each input image
  const int im_dim = width * height;
  for (int f = 0; f < input_nfeats; f++) {
    sum += input[f * im_dim + uvout];
  }

  output[uvout] = max(sqrt(sum) / ((float)input_nfeats * (float)input_nfeats * std_coef[uvout]),
                      threshold);
}

__kernel void SpatialDivisiveNormalization(
  const __global float* input,  // 0
  __global float* output,       // 1 
  const __global float* std) {     // 2

  const int width = get_global_size(0);
  const int height = get_global_size(1);

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);

  const int index = x_out + width * (y_out + height * f_out);
  output[index] = input[index] / std[y_out * width + x_out];
}
