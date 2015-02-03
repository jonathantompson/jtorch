__kernel void Fill(
  __global float* output,  // 0
  const float value) {     // 1

  const int x_out = get_global_id(0);

  output[x_out] = value;
}