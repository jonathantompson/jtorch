**jtorch - Jonathan's Torch7 Utility Library (C++, OpenCL)**
---------
---------

**Overview**
--------

Torch7 (<http://www.torch.ch/>) is an AMAZING machine learning library written and maintained by some very smart people.  For me it's only downside is that interfacing with it from C++ on Windows 7 is difficult (if not impossible).  This library is a C++ framework for doing the forward propagation of various torch modules.  It uses OpenCL to perform the forward prop on the GPU.  I have even found that some of the OpenCL modules here are actually faster than the Torch7 CUDA modules on Linux (i.e., the linear network here is very fast when doing forward prop on non-batch inputs).

Please note that this is not supposed to be a replacement for Torch7.  There is no back propagation (so no learning), and only a very limited subset of the modules are implemented.

The library consists of a simple lua codebase for recursively saving a torch model to a compact binary format (all in the ./lua folder):

```lua
-- TORCH USAGE:
dofile("./lua/save_nn_node.lua")
model = nn.Sequential()
model.add(x)  -- Add all the model components here
jtorch_root = "path_to/jtorch/"
dofile("../jtorch/jtorch.lua")
saveModel(model, "my_model.bin")
```

The library also contains a CPP framework for loading it and doing the forward prop.  See jtorch_test for more details of usage.  It uses OpenCL for all GPU computing.  The following stages have full  implementations:

- SpatialConvolution
- SpatialConvolutionCUDA  --> Only 3D tensor is supported (no batch processing)
- SpatialConvolutionMap   --> Full implementation but it is slow (all on CPU)
- Sequential
- Parallel  --> Uses the c++ jtorch::Table as a container for multiple jtorch::Tensor<float> instances
- Tanh
- Threshold
- Linear
- SpatialLPPooling  --> Full implementation but it is slow (all on CPU)
- SpatialMaxPooling
- SpatialMaxPoolingCUDA  --> Only 3D tensor is supported (no batch processing)
- SpatialSubtractiveNormalization
- SpatialDivisiveNormalization
- SpatialContrastiveNormalization

The following stages have partial implementations:
- JoinTable --> Nothing fancy.  Just concatenates along the 0th dimension. The output is always 1D.
- Reshape --> Only reshapes from a flattened N-D to 1-D vector (ie, for use before a linear stage after a convolution stage).  Even then, it wont do the copy, it just points to the previous stage's output.
- Transpose --> Just a pass through stage.  Again, it just points to the previous stage's output.

**Compilation**
---------------

Building jtorch uses Visual Studio 2012 on Windows, and cmake + gcc 4.7 (or greater) on Mac OS X.  The only real dependancy is the jtil + jcl libraries.  See <http://github.com/jonathantompson/jtil> and <http://github.com/jonathantompson/jcl> for more details.

VS2012 and cmake expect a specific directory structure:

- \\include\\WIN\\
- \\include\\MAC\_OS\_X\\
- \\lib\\WIN\\
- \\lib\\MAC\_OS\_X\\
- \\jtil\\
- \\jcl\\
- \\jtorch\\

So the dependancy headers and static libraries (.lib on Windows and .a on Mac OS X) are separated by OS and exist in directories at the same level as jtorch, jtil and jcl.  I have pre-compiled the dependencies and put them in dropbox, let me know if you need the link.

**Style**
---------

This project follows the Google C++ style conventions: 

<http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml>
