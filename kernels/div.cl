// output = output / div_val
__kernel void Div(
  const  float div_val,  // 0
  __global  float* output) {      // 1

  const int x_out = get_global_id(0);

  output[x_out] /= div_val;
}
