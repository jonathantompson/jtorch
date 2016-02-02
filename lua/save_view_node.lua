function jtorch._saveViewNode(node, ofile)
  assert(node.size:size() > 0)
  local dim = node.size:size()
  ofile:writeInt(dim)
  local numElements = 1
  for i = 1, dim do
    ofile:writeInt(node.size[i])
    numElements = numElements * node.size[i]
  end
  assert(numElements == node.numElements)
end
