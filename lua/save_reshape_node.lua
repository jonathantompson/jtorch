function jtorch._saveReshapeNode(node, ofile)
  local sz = node.size:totable()
  if #sz <= 0 then
    error('Bad module')
  end
  
  ofile:writeInt(#sz)
  for i = 1, #sz do
    ofile:writeInt(sz[i])
  end
end
