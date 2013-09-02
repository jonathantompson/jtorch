function saveJoinTableNode(node, ofile)
  -- Save the size of the last dimension
  ofile:writeInt(node.output:size()[node.output:dim()])
end
