function jtorch.loadTensor(filename)
  ifile = torch.DiskFile(filename, 'r')
  ifile:binary()

  local dim = ifile:readInt()
  local size = {}
  local total_size = 1
  for i = 1, dim do
    size[i] = ifile:readInt()
    total_size = total_size * size[i]
  end

  local data = ifile:readFloat(total_size)
  data = torch.FloatTensor(data)
  data = data:view(unpack(size))

  ifile:close()

  return data

end
