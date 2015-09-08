function jtorch.saveTensor(tensor, filename)
  local ofile = torch.DiskFile(filename, 'w')
  ofile:binary()

  local sz = 1
  for i = 1, tensor:dim() do
    sz = sz * tensor:size(i)
  end

  if (tensor:storage():size() ~= sz) then
    print 'WARNING: underlining storage is not the same size as the tensor'
    print '         saveTensor will clone the tensor and save the clone to disk'
    tensor = tensor:clone()
  end

  ofile:writeInt(tensor:dim())
  for i = 1, tensor:dim() do
    ofile:writeInt(tensor:size(i))
  end

  ofile:writeFloat(tensor:storage())
  ofile:close()
end
