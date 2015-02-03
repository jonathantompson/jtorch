function saveTensor(tensor, filename)
  ofile = torch.DiskFile(filename, 'w')
  ofile:binary()

  local sz = 1
  for i = 1, tensor:dim() do
    sz = sz * tensor:size(i)
  end
  assert(tensor:storage():size() == sz,
    'underlining storage is not the same size as the tensor')

  ofile:writeInt(tensor:dim())
  for i = 1, tensor:dim() do
    ofile:writeInt(tensor:size(i))
  end

  ofile:writeFloat(tensor:storage())
  ofile:close()
end
