function jtorch._saveThresholdNode(node, ofile)
  ofile:writeFloat(node.threshold)
  ofile:writeFloat(node.val)
end
