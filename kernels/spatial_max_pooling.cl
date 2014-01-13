__kernel void SpatialMaxPooling(const __global  float* input,  // 0
							    __global  float* output,       // 1 
								const int input_height,        // 2
								const int input_width,         // 3
								const int poolsize_v,		   // 4
								const int poolsize_u) {        // 5

  const int width = get_global_size(0);
  const int height = get_global_size(1);
  const int feats = get_global_size(2);

  const int x_out = get_global_id(0);
  const int y_out = get_global_id(1);
  const int f_out = get_global_id(2);

  // Initilize the output to the bias
  float out_val = - INFINITY;

  const int vstart = y_out * poolsize_v;
  const int vend = (y_out + 1) * poolsize_v - 1;

  // Get a pointer to the current input feature (that corresponds to this
  // output feature;
  const __global  float* input_f = &input[f_out * input_width * input_height];

  for (int v = vstart; v <= vend; v++) {
    const int istart = v * input_width + x_out * poolsize_u;
	  const int iend = v * input_width + (x_out + 1) * poolsize_u - 1;

    for (int i = istart; i <= iend; i++) {
	    out_val = max(out_val, input_f[i]);
	  }
  }

  const int index = x_out + width * (y_out + height * f_out);
  output[index] = out_val;
}
