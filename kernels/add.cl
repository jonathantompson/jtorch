// output = input1 + input2
__kernel void Add(
  const __global  float* input1,  // 0
  const __global  float* input2,  // 1
  __global  float* output) {      // 2

  const int x_out = get_global_id(0);

  output[x_out] = input1[x_out] + input2[x_out];
}
