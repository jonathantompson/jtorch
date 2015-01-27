// output += input1
__kernel void Accumulate(
  const __global  float* input1,  // 0
  __global  float* output) {      // 2

  const int x_out = get_global_id(0);

  output[x_out] += input1[x_out];
}
