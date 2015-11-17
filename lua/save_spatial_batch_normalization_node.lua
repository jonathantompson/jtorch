function jtorch._saveSpatialBatchNormalizationNode(node, ofile)

  if node.affine then
    ofile:writeInt(1)
  else
    ofile:writeInt(0)
  end

  assert(node.running_mean:dim() == 1)
  local sz = node.running_mean:size(1)
  assert(node.running_std:dim() == 1 and node.running_std:size(1) == sz)
  ofile:writeInt(sz)

  jtorch._saveFloatTensorSafe(ofile, node.running_mean)
  jtorch._saveFloatTensorSafe(ofile, node.running_std)

  if node.affine then
    assert(node.weight:dim() == 1 and node.weight:size(1) == sz)
    assert(node.bias:dim() == 1 and node.bias:size(1) == sz)
    jtorch._saveFloatTensorSafe(ofile, node.weight)
    jtorch._saveFloatTensorSafe(ofile, node.bias)
  end

end
