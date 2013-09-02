__kernel void Threshold(
  const __global  float* input, 
  __global float* output,
  const float threshold, 
  const float val) {

  const int width = get_global_size(0);
  const int height = get_global_size(1);

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);

  const int index = x_out + width * (y_out + height * f_out);

  output[index] = input[index] > threshold ? input[index] : val;
}

__kernel void Threshold1D(
  const __global  float* input, 
  __global float* output,
  const float threshold, 
  const float val) {

  const int x_out = get_global_id(0);

  output[x_out] = input[x_out] > threshold ? input[x_out] : val;
}

