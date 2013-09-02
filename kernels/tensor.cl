__kernel void Copy(
  const __global float* input,  // 0
  __global float* output) {     // 1

  const int x_out = get_global_id(0);

  output[x_out] = input[x_out];
}

__kernel void Zero(
  __global float* output) {     // 0

  const int x_out = get_global_id(0);

  output[x_out] = 0;
}