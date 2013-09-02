function saveSpatialMaxPoolingNode(node, ofile)

  -- The layout is as follows:
  -- 1. filter width (int)
  -- 2. filter height (int)

  ofile:writeInt(node.kW)
  ofile:writeInt(node.kH)

end
