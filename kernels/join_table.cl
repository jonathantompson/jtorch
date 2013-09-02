__kernel void JoinTable1D(
  const __global  float* input,  // 0
  __global  float* output,       // 1 
  const int output_offset) {     // 2

  const int x_in = get_global_id(0);

  output[x_in + output_offset] = input[x_in];
}
