-- Jonathan Tompson
-- NYU, MRL
-- This script turns the serialized neural network file into a file that is
-- readable by my c++ code.

dofile(jtorch_root.."/lua/save_nn_node.lua")

function saveModel(model, model_filename)
  model = model:float()

  -- Open an output file
  ofile = torch.DiskFile(model_filename, 'w')
  ofile:binary()

  -- Now recursively save the network
  saveNNNode(model, ofile)

  ofile:close()

  print("All done saving convnet")

end
