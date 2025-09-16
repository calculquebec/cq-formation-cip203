import torch
import time
import sys

if len(sys.argv) > 1:
    matrix_size = int(sys.argv[1])
    print(f"Performing matrix multiplication of two {matrix_size}x{matrix_size} matrices...")
else:
    print("No arguments provided. Quiting ...")
    sys.exit()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Define the dimensions for large matrices
matrix_size_cpu = 5000 # Adjust this value to increase/decrease GPU intensity

A1 = torch.randn(matrix_size, matrix_size, device=device)
B1 = torch.randn(matrix_size, matrix_size, device=device)
A1_cpu = torch.randn(matrix_size_cpu, matrix_size_cpu, device=torch.device("cpu"))
B1_cpu = torch.randn(matrix_size_cpu, matrix_size_cpu, device=torch.device("cpu"))


print(f"Allocated GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB")

# Perform matrix multiplication
start_time = time.perf_counter()
# Matrix multiplication on GPU
C1 = torch.matmul(A1, B1)

# Matrix multiplication on CPU
C1_cpu = torch.matmul(A1_cpu, B1_cpu)

torch.cuda.synchronize()
end_time = time.perf_counter()
elapsed_time = end_time - start_time
#print("Matrix multiplication complete.")
print(f"Execution time: {elapsed_time:.6f} seconds")
