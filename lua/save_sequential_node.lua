function saveSequentialNode(node, ofile)
  -- Write the number of nodes
  ofile:writeInt(#node.modules)
  -- Now just save each node recursively
  for i=1,#node.modules do
    saveNNNode(node:get(i), ofile)
  end
end
