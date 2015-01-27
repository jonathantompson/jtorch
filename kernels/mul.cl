// output = mul_val * output
__kernel void Mul(
  const  float mul_val,  // 0
  __global  float* output) {      // 1

  const int x_out = get_global_id(0);

  output[x_out] *= mul_val;
}
