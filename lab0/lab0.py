import random
import time

# 生成随机矩阵
def generate_random_matrix(rows, cols):
    matrix = []
    for i in range(rows):
        row = [random.random() for _ in range(cols)]
        matrix.append(row)
    return matrix

# 矩阵乘法
def matrix_multiplication(A, B):
    M = len(A)
    N = len(A[0])
    K = len(B[0])
    C = [[0.0 for _ in range(K)] for _ in range(M)]
    
    for i in range(M):
        for j in range(K):
            for k in range(N):
                C[i][j] += A[i][k] * B[k][j]
    
    return C

def main():
    M, N, K = map(int, input("Enter values for M, N, and K (512-2048): ").split())

    if M < 512 or M > 2048 or N < 512 or N > 2048 or K < 512 or K > 2048:
        print("Invalid input. All values must be between 512 and 2048.")
        return 1

    # 生成随机矩阵
    random.seed(time.time())
    A = generate_random_matrix(M, N)
    B = generate_random_matrix(N, K)

    # 计算矩阵乘法的时间
    start_time = time.time()
    C = matrix_multiplication(A, B)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # print("Matrix A:")
    # for row in A:
    #     print(row)

    # print("Matrix B:")
    # for row in B:
    #     print(row)

    # print("Matrix C:")
    # for row in C:
    #     print(row)

    print(f"Matrix multiplication took {elapsed_time:.4f} seconds.")

if __name__ == "__main__":
    main()
