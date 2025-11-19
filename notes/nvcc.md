compilar: `nvcc main.cu`
ejecutar: `./a.out`

compilar con optimizaciones: `nvcc -O2 main.cu`
linkar cublas: `nvcc -lcublas main.cu`
ptx: `nvcc -ptx main.cu`
binary: `nvcc -cubin main.cu` -> `cuobjdump --dump-sass main.cubin`