function saveArray(tensor, filename)
  ofile = torch.DiskFile(filename, 'w')
  ofile:binary()

  if (tensor:dim() == 1) then
    for x=1,tensor:size()[1] do
      ofile:writeFloat(tensor[{x}])
    end
  elseif (tensor:dim() == 2) then
    for y=1,tensor:size()[1] do
      for x=1,tensor:size()[2] do
        ofile:writeFloat(tensor[{y, x}])
      end
    end
  elseif (tensor:dim() == 3) then
    for z=1,tensor:size()[1] do
      for y=1,tensor:size()[2] do
        for x=1,tensor:size()[3] do
          ofile:writeFloat(tensor[{z, y, x}])
        end
      end
    end
  else
    error("saveArray() - 4D+ tensors not supported!")
  end

  ofile:close()
end
