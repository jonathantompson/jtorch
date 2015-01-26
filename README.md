**jtorch - Jonathan's Torch7 Utility Library (C++, OpenCL)**
---------
---------

**Overview**
--------

Torch7 (<http://www.torch.ch/>) is an AMAZING machine learning library written and maintained by some very smart people :-)  For me it's only downside is that interfacing with it from C++ on Windows 7 (and other operating systems other than Mac OS X and Linux) is difficult (if not sometimes impossible).  This library is a C++ framework for doing the forward propagation of various torch modules.  It uses OpenCL to perform the forward prop on the GPU.  I have even found that some of the OpenCL modules here are actually faster than the Torch7 CUDA modules on Linux.  With this said, you should profile torch vs jtorch and make sure there are no super slow modules in this library (since I haven't spent all that much time optimizing GPU code).

Please note that this is not supposed to be a replacement for Torch7.  There is no back propagation (so no learning), and only a very limited subset of the modules are implemented.  Additionally, none of the modules support batches (so single sample FPROP only).  The use-case for this library is for people who do model development on Linux, but want to run real-time FPROP of their models on other operating systems.

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

Building jtorch uses Visual Studio 2012 on Windows, and cmake + gcc 4.7 (or greater) on Mac OS X.  The only dependancy is the jcl library.  See <http://github.com/jonathantompson/jcl> for more details.

VS2012 and cmake expect a specific directory structure:

- \\jcl\\
- \\jtorch\\

So jtorch and jcl must exist at the same directory level.

### Windows:
- **Follow all the compilation steps in <https://github.com/jonathantompson/jcl/blob/master/README.md>**
- **Compiling the library:**
    - open jtorch.sln
    - right click "jtorch_test" in the Solution Explorer -> "Set as StartUp Project"
    - From the file menu: DEBUG -> Start Debugging... (or press F5)
    - The tests should run and at the end of each test there should be a "PASSED" printout (otherwise it will print FAILED).

### MAC OS X:
- **Follow all the compilation steps in <https://github.com/jonathantompson/jcl/blob/master/README.md>**
- Just run cmake and then build (all frameworks should be included).  
- cl.hpp doesn't exist by default but there is a copy of it in the local directory opencl_cpp_header.

**Style**
---------

This project follows the Google C++ style conventions: 

<http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml>

**License**
-----------
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
