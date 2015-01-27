__kernel void Zero(
  __global float* output) {     // 0

  const int x_out = get_global_id(0);

  output[x_out] = 0;
}