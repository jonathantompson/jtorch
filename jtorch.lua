local jtorch = {}

function jtorch.init(jtorchRoot)
  jtorch.jtorchRoot = jtorchRoot
  dofile(jtorchRoot .. "/lua/save_tensor.lua")
  dofile(jtorchRoot .. "/lua/load_tensor.lua")
  dofile(jtorchRoot .. "/lua/save_model.lua")
end

-- Add the package to the global namespace.
rawset(_G, 'jtorch', jtorch)

return jtorch
