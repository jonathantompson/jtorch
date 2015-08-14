function jtorch._saveJoinTableNode(node, ofile)
  -- Save the dimension to join
  ofile:writeInt(node.dimension)
end
