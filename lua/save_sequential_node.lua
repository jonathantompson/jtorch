function jtorch._saveSequentialNode(node, ofile)
  -- Write the number of nodes
  ofile:writeInt(#node.modules)
  -- Now just save each node recursively
  for i=1,#node.modules do
    jtorch._saveNNNode(node:get(i), ofile)
  end
end
