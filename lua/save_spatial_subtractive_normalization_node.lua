function saveSpatialSubtractiveNormalizationNode(node, ofile)

  if (node.kernel:dim() > 2) then
    error("saveSpatialSubtractiveNormalizationNode() - ERROR: Only 1D and 2D kernels are supported for now!")
  end

  -- The layout is as follows:
  -- 1. filter kernel size 1 (int)
  -- 2. filter kernel size 2 (int)
  -- 3. filter kernel (float array)

  if (node.kernel:dim() == 1) then
    ofile:writeInt(node.kernel:size())
    ofile:writeInt(1)  -- i.e. export as 2D even if it's 1D
  else
    ofile:writeInt(node.kernel:size(1))
    ofile:writeInt(node.kernel:size(2))
  end
  saveFloatTensorSafe(ofile, node.kernel)
end
