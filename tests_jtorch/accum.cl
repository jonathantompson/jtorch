__kernel void Accum1D(
  const __global  float* input,  // 0
  __global  float* output) {     // 1

  const int x_in = get_global_id(0);

  output[x_in] += input[x_in];
}
