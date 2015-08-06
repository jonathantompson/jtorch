**jtorch - Jonathan's Torch7 Utility Library (C++, OpenCL)**
---------
---------

**Overview**
--------

Torch7 (<http://www.torch.ch/>) is an AMAZING machine learning library written and maintained by some very smart people :-) For me it's only downside is that interfacing with it from C++ on Windows 7 (and other operating systems other than Mac OS X and Linux) is difficult (if not sometimes impossible). This library is a C++ framework for doing the forward propagation of various torch modules. It uses OpenCL to perform the forward prop on the GPU. I have even found that some of the OpenCL modules here are actually faster than the Torch7 CUDA modules on Linux. With this said, you should profile torch vs jtorch and make sure there are no super slow modules in this library (since I haven't spent all that much time optimizing GPU code).

Please note that this is not supposed to be a replacement for Torch7. There is *no back propagation* (so no learning), and only a very limited subset of the modules are implemented. Additionally, none of the modules support batches (so single sample FPROP only). The use-case for this library is for people who do model development on Linux, but want to run real-time FPROP of their models on other operating systems.

The library consists of a simple lua codebase for recursively saving a torch model to a compact binary format (all in the ./lua folder):

```lua
-- TORCH USAGE:
model = nn.Sequential()
model.add(x)  -- Add all the model components here
jtorch_root = "path_to/jtorch/"
dofile(jtorch_root .. "/jtorch.lua")
saveModel(model, "my_model.bin")
```

The library also contains a CPP framework for loading it and doing the forward prop. See jtorch_test for more details of usage. It uses OpenCL for all GPU computing. The following stages have full implementations (without batch support):

- CAddTable
- Identity
- Linear
- Parallel
- Reshape
- SelectTable
- Sequential
- SpatialContrastiveNormalization
- SpatialConvolution
- SpatialConvolutionMM --> (using clBLAS)
- SpatialConvolutionMap   --> (only on CPU)
- SpatialDivisiveNormalization
- SpatialLPPooling  --> (only on CPU)
- SpatialMaxPooling
- SpatialSubtractiveNormalization
- SpatialUpSamplingNearest
- Tanh
- Threshold

The following stages have partial implementations:
- JoinTable --> Only joins along the top most dimension are allowed for now
- Transpose --> Just a pass through stage. Again, it just points to the previous stage's output.

**Compilation Overview**
------------------------

Building jtorch uses Visual Studio 2012 on Windows, and cmake + gcc 4.7 (or greater) on Mac OS X (note: the CMakeLists need updating, but shouldn't be too hard). I have also built on clang and gcc on Ubuntu 14.04, but I don't supply the build files. The only dependencies are the OpenCL drivers / SDKs for your desired OpenCL compute plateform and clBLAS. See <https://github.com/clMathLibraries/clBLAS> for more details (also for clBLAS I have a "Compiling_clBLAS.txt" helper if you get stuck).

VS2012 and cmake expect a specific directory structure:

- \\jtorch\\
- \\clBLAS\\

So jtorch and clBLAS must exist at the same directory level.

** Build and run jcl (an internal project) **
---------------

### Windows:
- **Intel SDK (for CPU support)**
    - Download and install the CPU Only runtime from: http://registrationcenter.intel.com/irc_nas/3782/intel_sdk_for_ocl_applications_2013_r3_runtime_x64_setup.msi
- **For ATI CARDS:**
    - Download AMD APP SDK and install using express settings: (I used version 2.8.1016.5) http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/#one
    - Create windows environment variables (right click My Computer -> Properties -> Advanced System Settings -> Environment Variables -> System Variables -> New)
        - Name = `OPENCL_INC_DIR`, Value = C:\Program Files (x86)\AMD APP\include
        - Name = `OPENCL_LIB_DIR`, Value = C:\Program Files (x86)\AMD APP\lib\x86_64
- **FOR NVIDIA CARDS:**
    - Download the CUDA Toolkit 7 - https://developer.nvidia.com/cuda-downloads and install it. Note, this has also been tested with V5.5 and I'm sure there won't be too many issues (if any) with versions newer than 7 since I don't use any late version OpenCL features.
    - Create windows environment variables (right click My Computer -> Properties -> Advanced System Settings -> Environment Variables -> System Variables -> New): 
        - Name = `OPENCL_INC_DIR`, Value = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include
        - Name = `OPENCL_LIB_DIR`, Value = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\lib\x64
    - Now download the hpp header from http://www.khronos.org/registry/cl/api/1.2/cl.hpp and put it in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include\CL
	- You may need to follow the instructions here: https://github.com/clMathLibraries/clBLAS/issues/117 (you need all the headers)
- **Compiling the library:**
    - open jtorch.sln
    - right click "test_jcl" in the Solution Explorer -> "Set as StartUp Project"
    - From the file menu: DEBUG -> Start Debugging... (or press F5)
    - TWO tests should run and at the end of each test there should be a "PASSED" printout.

### MAC OS X:
 - Just run cmake and then build (all frameworks should be included).  
 - cl.hpp doesn't exist by default but there is a copy of it in the local directory opencl_cpp_header.

** Build and run jtorch **
---------------

### Windows:
- **Follow all the compilation steps above ('Build and run jcl')**
- **Follow all the compilation steps in Compiling_clBLAS.txt**
- **Compiling the library:**
    - open jtorch.sln
    - right click "jtorch_test" in the Solution Explorer -> "Set as StartUp Project"
    - From the file menu: DEBUG -> Start Debugging... (or press F5)
    - The tests should run and at the end of each test there should be a "PASSED" printout (otherwise it will print FAILED).

### MAC OS X:
- Before integration of clBLAS everything compiled and ran correctly. However, post clBLAS I have not tested it on Mac OS X or Linux.

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
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
