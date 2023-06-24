https://stackoverflow.com/questions/27590166/how-to-compile-multiple-files-in-cuda
DynamicParallelism.png

nvcc a.cu ut.cu -rdc=true -I ../../Common
./a.out


# --relocatable-device-code {true|false}          (-rdc)                          
#         Enable (disable) the generation of relocatable device code.  If disabled,
#         executable device code is generated.  Relocatable device code must be linked
#         before it can be executed.
#         Default value:  false.

