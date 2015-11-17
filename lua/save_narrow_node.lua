function jtorch._saveNarrowNode(node, ofile)
  ofile:writeInt(node.dimension)
  ofile:writeInt(node.index)
  ofile:writeInt(node.length)
end
