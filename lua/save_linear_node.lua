function jtorch._saveLinearNode(node, ofile)
  local noutputs = node.weight:size()[1]
  local ninputs = node.weight:size()[2]
  ofile:writeInt(noutputs)
  ofile:writeInt(ninputs)
 
  assert(node.weight:dim() == 2, 'weight matrix is not 2D!')
  
  jtorch._saveFloatTensorSafe(ofile, node.weight:clone():t():contiguous())
  
  assert(node.bias:dim() == 1, 'bias vector is not 1D!')
  jtorch._saveFloatTensorSafe(ofile, node.bias)
end
