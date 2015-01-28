__kernel void SpatialUpSamplingNearest(
  const __global  float* input,  // 0
  __global  float* output,       // 1 
  const int scale) {             // 2

  const int width_out = get_global_size(0);
  const int height_out = get_global_size(1);
  // const int feats_out = get_global_size(2);

  const int width_in = width_out / scale;
  const int height_in = height_out / scale;
  // const int feats_in = feats_in / scale;

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);

  const int x_in = x_out / scale;
  const int y_in = y_out / scale;
  const int f_in = f_out;

  const int iout = x_out + width_out * (y_out + height_out * f_out);
  const int iin = x_in + width_in * (y_in + height_in * f_in);

  output[iout] = input[iin];
}

