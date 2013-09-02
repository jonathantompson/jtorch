// From here: http://www.bealto.com/gpu-gemv_v1.html
__kernel void MatVecMultSimple(
  // Y = A * X (matrix-vector mulitply)
  __global const float* A,  // 0  --> Size M (rows) x N (cols) stored column major
  __global const float* X,  // 1  --> Size N
  __global  float* Y,       // 2  --> Size M
  const int M,              // 3
  const int N) {            // 4

  const int i = get_global_id(0);  // row index

  float sum = 0;
  // Perform the linear accumulation
  for (int k = 0; k < N; k++) {
    sum += A[i + M * k] * X[k];
  }

  Y[i] = sum;
}

#define ROW_DIM 0
#define COL_DIM 1

// From here: http://www.bealto.com/gpu-gemv_v2.html
__kernel void MatVecMultThreads(
  // Y = A * X (matrix-vector mulitply)
  __global const float* A,  // 0  --> Size M (rows) x N (cols) stored column major
  __global const float* X,  // 1  --> Size N
  __global  float* Y,       // 2  --> Size M
  __local float* work,      // 3  --> Size M by p
  const int M,              // 4
  const int N) {            // 5

  // Compute partial dot product
  float sum = 0;
  for (int k = get_global_id(COL_DIM); k < N; k += get_global_size(COL_DIM)) {
    sum += A[get_global_id(ROW_DIM) + M * k] * X[k];
  }

  // Each thread stores its partial sum in WORK
  int rows = get_local_size(ROW_DIM); // rows in group
  int cols = get_local_size(COL_DIM); // initial cols in group
  int ii = get_local_id(ROW_DIM); // local row index in group, 0<=ii<rows
  int jj = get_local_id(COL_DIM); // block index in column, 0<=jj<cols
  work[ii+rows*jj] = sum;
  barrier(CLK_LOCAL_MEM_FENCE); // sync group

  // Reduce sums in log2(cols) steps
  while (cols > 1) {
    cols >>= 1;
	if (jj < cols) { 
	  work[ii + rows * jj] += work[ii + rows * (jj + cols)];
	}
	barrier(CLK_LOCAL_MEM_FENCE); // sync group
  }

  // Write final result in Y
  if ( jj == 0 ) {
    Y[ get_global_id(ROW_DIM) ] = work[ii];
  }
}

__kernel void Accum (
  // output = bias
  __global  float* output,          // 0
  const __global float* biases) {   // 1

  const int x_out = get_global_id(0);

  output[x_out] += biases[x_out];
}
