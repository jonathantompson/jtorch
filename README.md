**jtorch - An OpenCL Torch Utility Library**
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
local jtorch = dofile("path/to/jtorch/jtorch.lua")
jtorch.init("path/to/jtorch/")
jtorch.saveModel(model, "my_model.bin")
```

The library also contains a CPP framework for loading it and doing the forward prop. See jtorch_test for more details of usage. It uses OpenCL for all GPU computing. The following stages have full implementations (without batch support):

- CAddTable
- ConcatTable
- Identity
- Linear
- MulConstant
- Parallel
- ParallelTable
- Reshape
- SelectTable
- Sequential
- SpatialBatchNormalization
- SpatialContrastiveNormalization
- SpatialConvolution
- SpatialConvolutionMM --> (using clBLAS)
- SpatialConvolutionMap   --> (only on CPU)
- SpatialDivisiveNormalization
- SpatialDropout
- SpatialLPPooling  --> (only on CPU)
- SpatialMaxPooling
- SpatialSubtractiveNormalization
- SpatialUpSamplingNearest
- Tanh
- Threshold
- View

The following stages have partial implementations:
- Concat --> We can only concatenate along the outer dimension for now.
- JoinTable --> Only joins along the top most dimension are allowed for now
- Narrow --> We can only narrow along the outer dimension for now.
- Select --> We can only select along the outer dimension for now.
- Transpose --> Just a pass through stage. Again, it just points to the previous stage's output.

**Compilation Overview**
------------------------

Building jtorch uses cmake + Visual Studio on Windows, and cmake + gcc 4.7 (or greater) on Mac OS X (not tested recently) and Linux (tested recently). The only dependencies are the OpenCL drivers / SDKs for your desired OpenCL compute plateform and clBLAS. See <https://github.com/clMathLibraries/clBLAS> for more details (also for clBLAS I have a "Compiling_clBLAS.txt" helper if you get stuck).

On windows, the cmake scripts expect a specific directory structure:

- \\jtorch\\
- \\clBLAS\\

So jtorch and clBLAS must exist at the same directory level.

** Build and run jcl and the tests **
---------------

### Windows:
- **Intel SDK (for CPU support)**
    - Download and install the CPU Only runtime from: http://registrationcenter.intel.com/irc_nas/3782/intel_sdk_for_ocl_applications_2013_r3_runtime_x64_setup.msi
- **For ATI CARDS:**
    - Download AMD APP SDK and install using express settings: (I used version 2.8.1016.5) http://developer.amd.com/tools-and-sdks/heterogeneous-computing/amd-accelerated-parallel-processing-app-sdk/downloads/#one
    - Create windows environment variables (right click My Computer -> Properties -> Advanced System Settings -> Environment Variables -> System Variables -> New)
        - Name = `OPENCL_INC_DIR`, Value = C:\Program Files (x86)\AMD APP\include
        - Name = `OPENCL_LIB_DIR`, Value = C:\Program Files (x86)\AMD APP\lib\x86_64
	- Also define the environment variable AMDAPPSDKROOT.
- **FOR NVIDIA CARDS:**
    - Download the CUDA Toolkit 7 - https://developer.nvidia.com/cuda-downloads and install it. Note, this has also been tested with V5.5 and V7.5.
    - Create windows environment variables (right click My Computer -> Properties -> Advanced System Settings -> Environment Variables -> System Variables -> New): 
        - Name = `OPENCL_INC_DIR`, Value = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include
        - Name = `OPENCL_LIB_DIR`, Value = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\lib\x64
	- Also define the environment variables 'CUDA_INC_PATH' and 'CUDA_LIB_PATH' using the same values above.
    - ~~Now download the hpp header from http://www.khronos.org/registry/cl/api/1.2/cl.hpp and put it in C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.0\include\CL~~ Update, it looks like Toolkit 7.5 includes it.
	- You may need to follow the instructions here: https://github.com/clMathLibraries/clBLAS/issues/117 (you need all the headers)
- **Compiling the library:**
    - Open the cmake gui.
	- In "Where is the source code:" put "C:/path/to/jtorch"
	- In "Where to build the binaries:" put "C:/path/to/jtorch/build"
	- Click "Configure twice", then "Generate"
    - open build/jtorch.sln
    - right click "test_jcl" in the Solution Explorer -> "Set as StartUp Project"
    - From the file menu: DEBUG -> Start Debugging... (or press F5)
    - The tests should run and at the end of each test there should be a "PASSED" printout.

### MAC OS X / LINUX:
 - Just run cmake and then build (all frameworks should be included):
```
git clone git@github.com:jonathantompson/jtorch.git
cd jtorch; mkdir build; cd build
cmake ..
make -j 8
```
 - cl.hpp doesn't exist by default but there is a copy of it in the local directory opencl_cpp_header.

** Build and run jtorch **
---------------

### Windows:
- **Follow all the compilation steps above ('Build and run jcl')**
- **Follow all the compilation steps in Compiling_clBLAS.txt**
- **Compiling the library:**
    - The make target should be defined in the cmake above.
	- Note: the tests_jtorch target requres the path/to/test_data as the first argement to the executable.

### MAC OS X / LINUX:
- **Follow all the compilation steps in Compiling_clBLAS.txt**
- Again, just run cmake from this directory.
- Then run the test using:
>> cd build/tests_jtorch
>> ./tests_jtorch ../../tests_jtorch/test_data/

**Style**
---------

This project follows the Google C++ style conventions: 

<http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml>

**License**
-----------
Copyright (c) 2015, Jonathan Tompson, NYU, Google Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
