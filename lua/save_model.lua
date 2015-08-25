-- Jonathan Tompson
-- NYU, MRL
-- This script turns the serialized neural network file into a file that is
-- readable by my c++ code.

dofile(jtorch.jtorchRoot .. "/lua/save_nn_node.lua")

function jtorch.saveModel(model, model_filename)
  -- Open an output file
  local ofile = torch.DiskFile(model_filename, 'w')
  ofile:binary()

  -- Now recursively save the network
  jtorch._saveNNNode(model, ofile)

  ofile:close()

end
