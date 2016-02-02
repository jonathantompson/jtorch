function jtorch._saveSelectNode(node, ofile)
  ofile:writeInt(node.dimension)
  ofile:writeInt(node.index)
end
