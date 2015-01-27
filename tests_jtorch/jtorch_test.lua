require 'nn'
require 'image'
require 'torch'
require 'optim'
require 'sys'
torch.setdefaulttensortype('torch.FloatTensor')

dofile("save_array.lua")

num_feats_in = 5
num_feats_out = 10
width = 10
height = 10
math.mod = math.fmod

data_in = torch.FloatTensor(num_feats_in, height, width)

for f=1,num_feats_in do
  val = f - (width * height) / 16
  for v=1,height do
    for u=1,width do
      data_in[{f,v,u}] = val
      val = val + 1/8
    end
  end
end

saveArray(data_in, "test_data/data_in.bin")
print('Data In: saved to test_data/data_in.bin')

model = nn.Sequential()

-- Test tanh
model:add(nn.Tanh())
res = model:forward(data_in)
saveArray(res, "test_data/tanh_result.bin")
print('Tanh result saved to test_data/tanh_result.bin')

-- Test Threshold
threshold = 0.5
val = 0.1
model:add(nn.Threshold(threshold, val))
res = model:forward(data_in)
saveArray(res, "test_data/threshold.bin")
print('Threshold result saved to test_data/threshold.bin')

-- Test SpatialConvolutionMap
n_states_in = num_feats_in
n_states_out = num_feats_out
fan_in = 3
filt_width = 5
filt_height = 5
conn_table = nn.tables.random(n_states_in, n_states_out, fan_in)
for f_out=1,num_feats_out do
  for f_in=1,fan_in do
    conn_table[{(f_out-1) * fan_in + f_in, 2}] = f_out
    cur_f_in = math.mod((f_out-1) + (f_in-1), n_states_in) + 1
    conn_table[{(f_out-1) * fan_in + f_in, 1}] = cur_f_in
  end
end
-- print('Spatial Convolution Map Connection Table')
-- print(conn_table)
spat_conv_map = nn.SpatialConvolutionMap(conn_table, filt_width, filt_height)
-- print('Spatial Convolution Map Connection Table Rev')
-- print(spat_conv_map.connTableRev)
for f_out=1,num_feats_out do
  spat_conv_map.bias[{f_out}] = f_out / num_feats_out - 0.5
end
-- print('Spatial Convolution Map Biases')
-- print(spat_conv_map.bias)
num_filt = fan_in * num_feats_out
sigma_x_sq = 1
sigma_y_sq = 1
for filt=1,num_filt do
  for v=1,filt_height do
    for u=1,filt_width do
      x = u - 1 - (filt_width-1) / 2
      y = v - 1 - (filt_height-1) / 2
      spat_conv_map.weight[{filt, v, u}] = (filt / num_filt) * math.exp(-((x*x) / (2*sigma_x_sq) + 
        (y*y) / (2*sigma_y_sq)))
    end
  end
end
-- print('Spatial Convolution Map Weights')
-- print(spat_conv_map.weight)
model:add(spat_conv_map)
res = model:forward(data_in)
saveArray(res, "test_data/spatial_convolution_map.bin")
print('Spatial Convolution Map result saved to test_data/spatial_convolution_map.bin')

-- Test SpatialConvolution
n_states_in = num_feats_in
n_states_out = num_feats_out
fan_in = n_states_in
filt_width = 5
filt_height = 5
spat_conv = nn.SpatialConvolution(n_states_in, n_states_out, filt_width, filt_height)
for f_out=1,num_feats_out do
  spat_conv.bias[{f_out}] = f_out / num_feats_out - 0.5
end
-- print('Spatial Convolution Biases')
-- print(spat_conv.bias)
num_filt = num_feats_out * num_feats_in
sigma_x_sq = 1
sigma_y_sq = 1
for fout=1,num_feats_out do
  for fin=1,num_feats_in do
    for v=1,filt_height do
      for u=1,filt_width do
        filt = (fout-1) * num_feats_out + (fin - 1) + 1
        x = u - 1 - (filt_width-1) / 2
        y = v - 1 - (filt_height-1) / 2
        spat_conv.weight[{fout, fin, v, u}] = (filt / num_filt) * math.exp(-((x*x) / (2*sigma_x_sq) + 
          (y*y) / (2*sigma_y_sq)))
      end
    end
  end
end
-- print('Spatial Convolution Weights')
-- print(spat_conv.weight)
res = spat_conv:forward(model:get(2).output)
saveArray(res, "test_data/spatial_convolution.bin")
print('Spatial Convolution result saved to test_data/spatial_convolution.bin')

-- Test SpatialLPPooling
pnorm = 2.0
nstates = num_feats_out
poolsize_u = 2
poolsize_v = 2
pool_stage = nn.SpatialLPPooling(nstates, poolsize_u, poolsize_v, poolsize_u, poolsize_v)
model:add(pool_stage)
res = model:forward(data_in)
saveArray(res, "test_data/spatial_lp_pooling.bin")
print('Spatial LP Pooling result saved to test_data/spatial_lp_pooling.bin')

-- Test SpatialMaxPooling
model3 = nn.Sequential()
max_pool_stage = nn.SpatialMaxPooling(poolsize_u, poolsize_v, poolsize_u, poolsize_v)
model3:add(max_pool_stage)
res = model3:forward(data_in)
saveArray(res, "test_data/spatial_max_pooling.bin")
print('Spatial Max Pooling result saved to test_data/spatial_max_pooling.bin')

-- Test SpatialSubtractiveNormalization
model4 = nn.Sequential()
normkernel = image.gaussian1D(7)
print('Normalization Kernel1D')
print(normkernel)
norm = nn.SpatialSubtractiveNormalization(num_feats_in, normkernel)
model4:add(norm)
res = model4:forward(data_in)
saveArray(res, "test_data/spatial_subtractive_normalization.bin")
print('SpatialSubtractiveNormalization result saved to test_data/spatial_subtractive_normalization.bin')

model4 = nn.Sequential()
normkernel = image.gaussian(7)
print('Normalization Kernel2D')
print(normkernel)
norm = nn.SpatialSubtractiveNormalization(num_feats_in, normkernel)
model4:add(norm)
res = model4:forward(data_in)
saveArray(res, "test_data/spatial_subtractive_normalization_2d.bin")
print('SpatialSubtractiveNormalization result saved to test_data/spatial_subtractive_normalization_2d.bin')

-- Test SpatialDivisiveNormalization
model5 = nn.Sequential()
normkernel = image.gaussian1D(7)
print('Normalization Kernel1D')
print(normkernel)
spatial_div_norm = nn.SpatialDivisiveNormalization(num_feats_in, normkernel)
model5:add(spatial_div_norm)
res = model5:forward(data_in)
saveArray(res, "test_data/spatial_divisive_normalization.bin")
print('SpatialDivisiveNormalization result saved to test_data/spatial_divisive_normalization.bin')

model5 = nn.Sequential()
normkernel = image.gaussian(7)
print('Normalization Kernel2D')
print(normkernel)
spatial_div_norm = nn.SpatialDivisiveNormalization(num_feats_in, normkernel)
model5:add(spatial_div_norm)
res = model5:forward(data_in)
saveArray(res, "test_data/spatial_divisive_normalization_2d.bin")
print('SpatialDivisiveNormalization result saved to test_data/spatial_divisive_normalization_2d.bin')

-- return spatial_div_norm.localstds

-- Test SpatialContrastiveNormalization  --> Doesn't work on blackbox (works on my PC)
--[[
model6 = nn.Sequential()
lena = image.rgb2y(image.lena()):float()
file = torch.DiskFile("lena_image.bin", 'w')
file:binary()
for i=1,lena:size()[2] do
  for v=1,lena:size()[3] do
    file:writeFloat(lena[{1, i, v}])
  end
end
file:close()

normkernel = torch.Tensor(7):fill(1)
spatial_contrast_norm = nn.SpatialContrastiveNormalization(1, normkernel)
model6:add(spatial_contrast_norm)
res = model6:forward(lena)
image.display(lena)
image.display(res)
file = torch.DiskFile("lena_image_processed.bin", 'r')
file:binary()
lena_processed = file:readFloat(lena:size()[2] * lena:size()[3])
lena_processed = torch.FloatTensor(lena_processed, 1, torch.LongStorage{1, 
lena:size()[2], lena:size()[3]}):float()
file:close()
image.display(lena_processed)
err = lena_processed - res
err_abs = torch.abs(err)
image.display(err_abs)
print('Maximum error')
print(err_abs:max())
--]]

-- Test the local contrast normalization of the hand image generator
--[[
normkernel = torch.Tensor(11):fill(1)
spatial_contrast_norm = nn.SpatialContrastiveNormalization(1, normkernel)
file = torch.DiskFile("processed_hands_4618452720732.bin", 'r')
file:binary()
depth = file:readFloat(96*96)
depth = torch.FloatTensor(depth, 1, torch.LongStorage{1, 96, 96}):float()
file:close()
file = torch.DiskFile("hpf_processed_hands_4618452720732.bin", 'r')
file:binary()
hpf_depth = file:readFloat(96*96)
hpf_depth = torch.FloatTensor(hpf_depth, 1, torch.LongStorage{1, 96, 96}):float()
file:close()
image.display{image=hpf_depth, zoom=(6.0)}
image.display{image=depth, zoom=(6.0)}
hpf_depth_torch = spatial_contrast_norm:forward(depth)
image.display{image=hpf_depth_torch, zoom=(6.0)}
--]]

-- Test Linear
model2 = nn.Sequential()
lin_size = num_feats_in * width * height
lin_size_out = 20
model2:add(nn.Reshape(lin_size))
lin_stage = nn.Linear(lin_size, lin_size_out)
for n=1,lin_size do
  for m=1,lin_size_out do
    k = (m-1) * lin_size + (n-1) + 1
    lin_stage.weight[{m,n}] = k / (lin_size * lin_size_out)
  end
end
for i=1,lin_size_out do
  lin_stage.bias[{i}] = i / lin_size_out
end
model2:add(lin_stage)
res = model2:forward(data_in)
saveArray(res, "test_data/linear.bin")
print('Linear result saved to test_data/linear.bin')

-- Test model
--[[
test_model = nn.Sequential()
n_states_in = num_feats_in
n_states_out = num_feats_out
fan_in = n_states_in
filt_width = 5
filt_height = 5
test_model:add(nn.SpatialConvolution(n_states_in, n_states_out, filt_width, filt_height))
test_model:add(nn.Tanh())
test_model:add(nn.Threshold())
test_model:add(nn.SpatialMaxPooling(poolsize_u, poolsize_v, poolsize_u, poolsize_v))
width_out = (width - filt_width + 1) / 2
height_out = (height - filt_height + 1) / 2
lin_size_in = n_states_out * height_out * width_out
test_model:add(nn.Reshape(lin_size_in))
test_model:add(nn.Linear(lin_size_in, 6))
--]]
test_model = torch.load("testmodel.torch.bin")
res = test_model:forward(data_in)
saveArray(res, "test_data/test_model_result.bin")
print('Test model result saved to test_data/test_model_result.bin')

-- test SpatialUpSamplingNearest
do
  local model = nn.SpatialUpSamplingNearest(4)
  local input = image.gaussian(7):resize(1,7,7)
  local res = model:forward(input)
  saveArray(res, "test_data/spatial_up_sampling_nearest.bin")
  print('SpatialUpSamplingNearest result saved to test_data/spatial_up_sampling_nearest.bin')
end

-- Save the Test model
--[[
jtorch_root = "../jtorch/"
dofile("../jtorch/jtorch.lua")
saveModel(test_model, "testmodel.bin")
torch.save("testmodel.torch.bin", test_model)
--]]

-- Check the real model
--[[
require 'nn'
require 'image'
require 'torch'
require 'optim'
require 'sys'
torch.setdefaulttensortype('torch.FloatTensor')
require 'cunn'
require 'cutorch'
model = torch.load("../HandNets/handmodel.net")
collectgarbage()

width = 96
height = 96
bank_dim = {}
num_banks = 3
data_file_size = 0
num_features = 8
heat_map_width = 24
heat_map_height = 24
im_data = { data = {}, size = function() return 1 end, 
  heat_maps = torch.FloatTensor(1, num_features * heat_map_width * heat_map_height) }
w = width
h = height
for i=1,num_banks do
  table.insert(bank_dim, {h, w})
  data_file_size = data_file_size + w * h
  w = w / 2
  h = h / 2
  table.insert(im_data.data, torch.FloatTensor(1, 1, bank_dim[i][1], 
    bank_dim[i][2]))
end
w = nil
h = nil
filename = "../data/hand_depth_data_processed_for_CN/hpf_processed_1294371228_hands0_263917398000.bin"
hpf_depth_file = torch.DiskFile(filename,'r')
hpf_depth_file:binary()
hpf_depth_data = hpf_depth_file:readFloat(data_file_size)
hpf_depth_file:close()
ind = 1
for j=1,num_banks do
  im_data.data[j][{1, 1, {}, {}}] = torch.FloatTensor(
    hpf_depth_data, ind, torch.LongStorage{bank_dim[j][1], bank_dim[j][2]}):float()
  ind = ind + (bank_dim[j][1]*bank_dim[j][2]) -- Move pointer forward
end
filename = "../data/heatmaps/heatmap_hpf_processed_1294371228_hands0_263917398000.bin"
heatmap_file = torch.DiskFile(filename, 'r')
heatmap_file:binary()
heatmap_data = heatmap_file:readFloat(num_features * heat_map_width * heat_map_height)
heatmap_file:close()
im_data.heat_maps[{1,{}}]:copy(torch.FloatTensor(heatmap_data, 1,
  torch.LongStorage{num_features * heat_map_width * heat_map_height}):float())
dofile("../HandNets/visualize_data.lua")
-- VisualizeImage(im_data, 1, 1, 0)

for j=1,num_banks do
  im_data.data[j] = im_data.data[j]:cuda()
end

im_data.heat_maps = model:forward(im_data.data)
VisualizeImage(im_data, 1, 1, 0)
print("First 30 numbers of the output: ")
print(im_data.heat_maps[{{},{1,30}}])
print("First bank last module (before reshape)")
print(model:get(1):get(1):get(14))
print("First 6x6 of the 2nd feature of the first bank output: ")
print(model:get(1):get(1):get(14).output[{1, 2, {1,6}, {1,6}}])
print("First 6x6 of the 10th feature of the second bank output: ")
print(model:get(1):get(2):get(14).output[{1, 10, {1,6}, {1,6}}])
print("0:30 of the join table stage: ")
print(model:get(2).output[{1, {1, 30}}])
print("7000:7030 of the 1st linear stage output: ")
print(model:get(2).output[{1, {7001, 7031}}])

-- See how fast / slow the model is:
print("profiling model for 5 seconds...")
time_accum = 0;
num_iterations = 0;
while (time_accum < 5) do
  time0 = sys.clock()
  res = model:forward(im_data.data)
  time1 = sys.clock()
  time_accum = time_accum + (time1 - time0)
  num_iterations = num_iterations + 1
end
print("time per evaluation: " .. (1000 * (time_accum / num_iterations)) .. "ms")

VisualizeImage(im_data, 1, 1, 0)

file = torch.DiskFile("convnet_output.bin", 'r')
file:binary()
cpp_out = file:readFloat(num_features * heat_map_width * heat_map_height)
cpp_out = torch.FloatTensor(cpp_out, 1, 
  torch.LongStorage{num_features, heat_map_width, heat_map_height}):float()
file:close()

zoom_factor = 0.5 * 5 * height / heat_map_height
image.display{image=cpp_out, padding=2, nrow=4, zoom=zoom_factor, scaleeach=false}

-- Print out a few odd numbers
print(model:get(1):get(1):get(4).output[{1, 1, {1,20}, {1,6}}])  -- First threshold out
print(res[{1,{1,10}}])
--]]
