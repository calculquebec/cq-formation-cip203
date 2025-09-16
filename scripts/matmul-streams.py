import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Using CPU")

matrix_size = 10000  # Adjust this value to increase/decrease GPU intensity

# Initialize CUDA streams

# Allocate matrices in the GPU memory
A1 = torch.randn(matrix_size, matrix_size, device=device)
B1 = torch.randn(matrix_size, matrix_size, device=device)
A2 = torch.randn(matrix_size, matrix_size, device=device)
B2 = torch.randn(matrix_size, matrix_size, device=device)


# Perform matrix multiplication
print(f"Performing matrix multiplication of two {matrix_size}x{matrix_size} matrices...")
C1 = torch.matmul(A1, B1)
C2 = torch.matmul(A2, B2)

torch.cuda.synchronize()
