local cutorch = require 'cutorch'
local cunn = require 'cunn'
require 'image'
require 'torch'
require 'optim'
local sys = require 'sys'
require 'strict'
torch.setdefaulttensortype('torch.FloatTensor')

torch.setnumthreads(3)
torch.manualSeed(1)
cutorch.manualSeed(1)
math.randomseed(1)

local jtorch = dofile("../jtorch.lua")
jtorch.init("../")

local num_feats_in = 5
local num_feats_out = 10
local width = 10
local height = 10
local fan_in = 3
local filt_width = 5
local filt_height = 5

local data_in = torch.FloatTensor(num_feats_in, height, width)

for f = 1, num_feats_in do
  local val = f - (width * height) / 16
  for v = 1, height do
    for u = 1, width do
      data_in[{f, v, u}] = val
      val = val + 1 / 8
    end
  end
end

jtorch.saveTensor(data_in, "test_data/data_in.bin")

-- Test tanh
local tanh_model = nn.Tanh()
local tanh_res = tanh_model:forward(data_in)
jtorch.saveTensor(tanh_res, "test_data/tanh_res.bin")

-- Test Threshold
local threshold = 0.5
local val = 0.1
local threshold_model = nn.Threshold(threshold, val)
local threshold_res = threshold_model:forward(data_in)
jtorch.saveTensor(threshold_res, "test_data/threshold_res.bin")

-- Test a Sequential stage with both Tanh and Threshold
local seq_model = nn.Sequential()
seq_model:add(tanh_model)
seq_model:add(threshold_model)
local seq_res = seq_model:forward(data_in)
jtorch.saveTensor(seq_res, "test_data/sequential_res.bin")
jtorch.saveModel(seq_model, "test_data/sequential_model.bin")

-- Test SpatialConvolutionMap
local conn_table = nn.tables.random(num_feats_in, num_feats_out, fan_in)
for f_out = 1, num_feats_out do
  for f_in = 1, fan_in do
    conn_table[{(f_out - 1) * fan_in + f_in, 2}] = f_out
    local cur_f_in = math.fmod((f_out - 1) + (f_in - 1), num_feats_in) + 1
    conn_table[{(f_out - 1) * fan_in + f_in, 1}] = cur_f_in
  end
end
local conv_map = nn.SpatialConvolutionMap(conn_table, filt_width, filt_height)
for f_out = 1, num_feats_out do
  conv_map.bias[{f_out}] = f_out / num_feats_out - 0.5
end
local num_filt = fan_in * num_feats_out
local sigma_x_sq = 1
local sigma_y_sq = 1
for filt = 1, num_filt do
  for v = 1, filt_height do
    for u = 1, filt_width do
      local x = u - 1 - (filt_width - 1) / 2
      local y = v - 1 - (filt_height - 1) / 2
      conv_map.weight[{filt, v, u}] =
          (filt / num_filt) * math.exp(-((x * x) / (2 * sigma_x_sq) +
          (y * y) / (2 * sigma_y_sq)))
    end
  end
end
local conv_map_res = conv_map:forward(data_in)
jtorch.saveTensor(conv_map_res, "test_data/spatial_convolution_map_res.bin")

-- Test SpatialConvolution
local conv = nn.SpatialConvolution(num_feats_in, num_feats_out, filt_width,
                                   filt_height)
local conv_res = conv:forward(data_in)
jtorch.saveTensor(conv_res, "test_data/spatial_convolution_res.bin")
jtorch.saveModel(conv, "test_data/spatial_convolution_model.bin")

-- Test SpatialConvolutionMM with padding
local padw = 6
local padh = 3
local conv_mm = nn.SpatialConvolutionMM(num_feats_in, num_feats_out, filt_width,
                                        filt_height, 1, 1, padw, padh)
local conv_mm_res = conv_mm:forward(data_in)
jtorch.saveTensor(conv_mm_res, "test_data/spatial_convolution_mm_res.bin")
jtorch.saveModel(conv_mm, "test_data/spatial_convolution_mm_model.bin")

-- Test SpatialLPPooling
local pnorm = 2.0
local poolsize_u = 2
local poolsize_v = 2
local pooldw = 1
local pooldh = 1
local poolpadw = 1
local poolpadh = 1
local lp_pool = nn.SpatialLPPooling(num_feats_in, poolsize_u, poolsize_v,
                                    poolsize_u, poolsize_v)
local lp_pool_res = lp_pool:forward(data_in)
jtorch.saveTensor(lp_pool_res, "test_data/spatial_lp_pooling_res.bin")

-- Test SpatialMaxPooling
local max_pool = nn.SpatialMaxPooling(poolsize_u, poolsize_v, poolsize_u,
                                      poolsize_v)
local max_pool_res = max_pool:forward(data_in)
jtorch.saveTensor(max_pool_res, "test_data/spatial_max_pooling_res.bin")

-- Test SpatialMaxPooling with padding and stride
local kw = 4
local kh = 5;
local dw = 1;
local dh = 3;
local padw = 2;
local padh = 0;
local max_pool_stride = nn.SpatialMaxPooling(kw, kh, dw, dh, padw, padh)
local max_pool_stride_res = max_pool_stride:forward(data_in)
jtorch.saveTensor(max_pool_stride_res,
                  "test_data/spatial_max_pooling_stride_res.bin")

-- Test SpatialSubtractiveNormalization 1D kernel
local normkernel = image.gaussian1D(7)
local sub_norm = nn.SpatialSubtractiveNormalization(num_feats_in, normkernel)
local sub_norm_res = sub_norm:forward(data_in)
jtorch.saveTensor(sub_norm_res,
                  "test_data/spatial_subtractive_normalization_1d_res.bin")

-- Test SpatialSubtractiveNormalization 2D kernel
local normkernel = image.gaussian(7)
local sub_norm = nn.SpatialSubtractiveNormalization(num_feats_in, normkernel)
local sub_norm_res = sub_norm:forward(data_in)
jtorch.saveTensor(sub_norm_res,
                  "test_data/spatial_subtractive_normalization_2d_res.bin")

-- Test SpatialDivisiveNormalization 1D kernel
local normkernel = image.gaussian1D(7)
local div_norm = nn.SpatialDivisiveNormalization(num_feats_in, normkernel)
local div_norm_res = div_norm:forward(data_in)
jtorch.saveTensor(div_norm_res,
                  "test_data/spatial_divisive_normalization_1d_res.bin")

-- Test SpatialDivisiveNormalization 2D kernel
local normkernel = image.gaussian(7)
local div_norm = nn.SpatialDivisiveNormalization(num_feats_in, normkernel)
local div_norm_res = div_norm:forward(data_in)
jtorch.saveTensor(div_norm_res,
                  "test_data/spatial_divisive_normalization_2d_res.bin")

-- Test SpatialContrastiveNormalization
local lena = image.scale(image.lena():float(), 32, 32)
jtorch.saveTensor(lena, 'test_data/lena_image.bin')
local normkernel = torch.Tensor(7):fill(1)
local cont_norm = nn.SpatialContrastiveNormalization(3, normkernel)
local cont_norm_res = cont_norm:forward(lena)
jtorch.saveTensor(cont_norm_res,
                  'test_data/spatial_contrastive_normalization_res.bin')

-- Test Linear
local lin_size = num_feats_in * width * height
local lin_size_out = 20
local linear = nn.Sequential()
linear:add(nn.Reshape(lin_size))
linear:add(nn.Linear(lin_size, lin_size_out))
local linear_res = linear:forward(data_in)
jtorch.saveTensor(linear_res, "test_data/linear_res.bin")
jtorch.saveModel(linear, "test_data/linear_model.bin")

-- Test Concat
local concat_dim = 1  -- For now only dim 1 is supported
local concat = nn.Concat(concat_dim)
local num_tensors = 4
for i = 1, num_tensors do
  concat:add(nn.MulConstant(torch.rand(1):squeeze()))
end
local concat_res = concat:forward(data_in)
jtorch.saveTensor(concat_res, "test_data/concat_res.bin")
jtorch.saveModel(concat, "test_data/concat_model.bin")

-- Test Narrow
local narrow_model = nn.Concat(concat_dim)
assert(num_feats_in == 5)  -- otherwise this logic will break.
local narrow_indices = {1, 2, 3, 4, 5, 1, 2, 3, 4}
local narrow_lengths = {5, 4, 3, 2, 1, 4, 3, 2, 1}
local narrow_dim = 1
for i = 1, #narrow_indices do
  narrow_model:add(nn.Narrow(narrow_dim, narrow_indices[i], narrow_lengths[i]))
end
local narrow_res = narrow_model:forward(data_in)
jtorch.saveTensor(narrow_res, "test_data/narrow_res.bin")
jtorch.saveModel(narrow_model, "test_data/narrow_model.bin")

-- Test SpatialBatchNormalization
local nfeats_bn = 32
local eps = 1e-5
local momentum = 0.1
local affine = true
local batchNorm = nn.SpatialBatchNormalization(nfeats_bn, eps, momentum, affine)
batchNorm.train = false
batchNorm.bias:copy(torch.rand(nfeats_bn))
batchNorm.weight:copy(torch.rand(nfeats_bn))
batchNorm.running_mean:copy(torch.rand(nfeats_bn))
batchNorm.running_std:copy(torch.rand(nfeats_bn))
local batchNormIn = torch.rand(1, nfeats_bn, 16, 8)
local batchNormOutAffine = batchNorm:forward(batchNormIn)
jtorch.saveModel(batchNorm, 'test_data/batch_norm_affine_model.bin')
jtorch.saveTensor(batchNormIn[1], 'test_data/batch_norm_in.bin')
jtorch.saveTensor(batchNormOutAffine[1], 'test_data/batch_norm_affine_out.bin')
batchNorm.affine = false
local batchNormOut = batchNorm:forward(batchNormIn)
jtorch.saveModel(batchNorm, 'test_data/batch_norm_model.bin')
jtorch.saveTensor(batchNormOut[1], 'test_data/batch_norm_out.bin')


-- test SpatialUpSamplingNearest
local up = nn.SpatialUpSamplingNearest(4)
local up_res = up:forward(data_in)
jtorch.saveTensor(up_res, "test_data/spatial_up_sampling_nearest_res.bin")

-- Test model
local test_model = nn.Sequential()
test_model:add(nn.SpatialConvolution(num_feats_in, num_feats_out, filt_width,
                                     filt_height))
test_model:add(nn.Tanh())
test_model:add(nn.Threshold())
test_model:add(nn.SpatialMaxPooling(poolsize_u, poolsize_v, poolsize_u,
                                    poolsize_v))
test_model:add(nn.SpatialConvolutionMM(num_feats_out, num_feats_out, filt_width,
                                       filt_height, 1, 1,
                                       math.floor(filt_width / 2),
                                       math.floor(filt_width / 2)))
local width_out = (width - filt_width + 1) / 2
local height_out = (height - filt_height + 1) / 2
local lin_size_in = num_feats_out * height_out * width_out
test_model:add(nn.Reshape(lin_size_in))
test_model:add(nn.Linear(lin_size_in, 6))
local test_model_res = test_model:forward(data_in)
jtorch.saveTensor(test_model_res, "test_data/test_model_res.bin")
jtorch.saveModel(test_model, "test_data/test_model.bin")

-- Profile convolutions
do
  local fin = 128
  local fout = 512
  local kw = 11
  local kh = 11
  local pad = 5
  local imw = 90
  local imh = 60
  local t_test = 5

  local model = nn.SpatialConvolutionMM(fin, fout, kw, kh, 1, 1, pad):cuda()
  local input = torch.ones(1, fin, imh, imw):cuda()

  print('Profiling convolution for ' .. t_test .. ' seconds')

  sys.tic()
  local t_start = sys.toc()
  local t_end = t_start
  local niters = 0
  while (t_end - t_start < t_test) do
    model:forward(input)
    cutorch.synchronize()
    niters = niters + 1
    t_end = sys.toc()
  end
  print('Execution time: ' .. 1 / (niters / (t_end - t_start)) ..
        ' seconds per FPROP')
end

print('All data saved to disk.')
