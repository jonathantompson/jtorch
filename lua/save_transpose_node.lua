function saveTransposeNode(node, ofile)
  -- The CPP framework wont use transposes, but we'll save the
  -- permutations anyway

  ofile:writeInt(#node.permutations)

  for i=1,#node.permutations do
    local cur_perm = node.permutations[i]
    ofile:writeInt(cur_perm[1])
    ofile:writeInt(cur_perm[2])
  end
end
