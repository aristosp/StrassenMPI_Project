#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
// Use of MPI here, because MPI_Wtime is more accurate than time functions from
// <time.h>
// N indicates the matrices dimensions, e.g. N=120 means matrices of 120x120
#define N 120
// Dynamically allocate memory using the following function
float **mem_alloc(int size){
	int i;
	float **array = (float **) malloc (size * sizeof (float *));
    for (i=0; i< size; i++) array[i]=malloc(size*sizeof(float));
	return array;
}
// Matrix Initialization Function
float **create_matrix(int size){
	// Due to the use of 2d arrays, int pointer is needed. Therefore,
	// using malloc, memory size is being allocated, and then the matrix is
	// randomly initiated
	float **array = mem_alloc(size);
	int i,j;
	for (i = 0; i < size; i++){
		for (j =0; j<size; j++) array[i][j] = rand() % 10; // normalize values
	}
	return array;
}
float **Strassen(float **matrixA,float **matrixB,int n){
		// This function implements the Strassen Multiplication Algorithm.
		// The input arrays are being split into 4 submatrices each and then the
		// Strassen coefficients (Pi,i=1..7) are being computed recursively using
		// this function, until the result is the result of a simple Multiplication
		// between two numbers. So in every recursion the arrays are being reduced
		// until they become 2x2 matrices.Finally, the result submatrices (Ci,j) are
		// concatenated floato one.
		float **P1,**P2,**P3,**P4,**P5,**P6,**P7;
		float **C11,**C12,**C21,**C22;
		float **S1,**S2,**S3,**S4,**S5,**S6,**S7,**S8,**S9,**S10;
		float **A11,**A12,**A21,**A22,**B11,**B12,**B21,**B22;
    int i,j;
    float ** res = mem_alloc(n);
    int new_n = n/2;
    if(n>1) {
				A11 = mem_alloc(new_n);
				A12 = mem_alloc(new_n);
				A21 = mem_alloc(new_n);
				A22 = mem_alloc(new_n);
				B11 = mem_alloc(new_n);
				B12 = mem_alloc(new_n);
				B21 = mem_alloc(new_n);
		    B22 = mem_alloc(new_n);
		    for (i = 0; i < new_n; i++) {
		    	for(j = 0; j<new_n; j++){
						A11[i][j] = matrixA[i][j];
		        A12[i][j] = matrixA[i][j + new_n];
		        A21[i][j] = matrixA[i + new_n][j];
		        A22[i][j] = matrixA[i + new_n][j + new_n];
		        B11[i][j] = matrixB[i][j];
		        B12[i][j] = matrixB[i][j + new_n];
		        B21[i][j] = matrixB[i + new_n][j];
		        B22[i][j] = matrixB[i + new_n][j + new_n];
					}
		    }
				S1 = mem_alloc(new_n);
				S2 = mem_alloc(new_n);
				S3 = mem_alloc(new_n);
				S4 = mem_alloc(new_n);
				S5 = mem_alloc(new_n);
				S6 = mem_alloc(new_n);
				S7 = mem_alloc(new_n);
				S8 = mem_alloc(new_n);
				S9 = mem_alloc(new_n);
				S10 = mem_alloc(new_n);
				for (i=0;i<new_n;i++){
					for(j=0;j<new_n;j++){
						S1[i][j] = B12[i][j] - B22[i][j];
						S2[i][j] = A11[i][j] + A12[i][j];
						S3[i][j] = A21[i][j] + A22[i][j];
						S4[i][j] = B21[i][j] - B11[i][j];
						S5[i][j] = A11[i][j] + A22[i][j];
						S6[i][j] = B11[i][j] + B22[i][j];
						S7[i][j] = A12[i][j] - A22[i][j];
						S8[i][j] = B21[i][j] + B22[i][j];
						S9[i][j] = A11[i][j] - A21[i][j];
						S10[i][j] = B11[i][j] + B12[i][j];
					}
				}
				P1 = Strassen(A11,S1,new_n);
				P2 = Strassen(S2,B22,new_n);
				P3 = Strassen(S3,B11,new_n);
				P4 = Strassen(A22,S4,new_n);
				P5 = Strassen(S5,S6,new_n);
				P6 = Strassen(S7,S8,new_n);
				P7 = Strassen(S9,S10,new_n);
				//free unneeded memory space
				free(A11);free(A12);free(A21);free(A22);free(B11);free(B12);free(B21);free(B22);
				free(S1);free(S2);free(S3);free(S4);free(S5);free(S6);free(S7);free(S8);free(S9);free(S10);
				C11 = mem_alloc(new_n);
				C12 = mem_alloc(new_n);
				C21 = mem_alloc(new_n);
				C22 = mem_alloc(new_n);
				for(i=0;i<new_n;i++){
					for(j=0;j<new_n;j++){
							C11[i][j] = P5[i][j] + P4[i][j] - P2[i][j] + P6[i][j];
							C12[i][j] = P1[i][j] + P2[i][j];
							C21[i][j] = P3[i][j] + P4[i][j];
							C22[i][j] = P5[i][j] + P1[i][j] - P3[i][j] - P7[i][j];
					}
				}
				// Free unneeded space
				free(P1);free(P2);free(P3);free(P4);free(P5);free(P6);free(P7);
		    for (i=0;i<new_n;i++){
		      for(j=0;j<new_n;j++){
		        res[i][j] = C11[i][j];
		        res[i][j+new_n] = C12[i][j];
		        res[new_n+i][j] = C21[i][j];
		        res[new_n+i][new_n+j] = C22[i][j];}
		      }
		    }
    else {res[0][0] = matrixA[0][0]*matrixB[0][0];}
		return res;
}
int i,j,k;
int main(int argc, char *argv[]) {
			float **C = mem_alloc(N);
			int new_n = N/2;
			MPI_Init(&argc,&argv);
		  srand(0);
		  //Initiate matrices
		  printf("Matrices Initiation \n");
		  float **A = create_matrix(N);
			float **B = create_matrix(N);
		  printf("Normal Computation\n");
			// Timer for normal computation
			double start_time_normal = MPI_Wtime();
		  for(i=0;i<N;i++){
		    for(j=0;j<N;j++){
		      for(k=0;k<N;k++) C[i][j] += A[i][k]*B[k][j];
		    }
		  }
			double end_time_normal = MPI_Wtime(); // Timer for normal computation
			printf("Time for Normal Computation is %f \n",end_time_normal - start_time_normal);
			printf("Strassen Computation\n");
			free(C);
			double one_core_start = MPI_Wtime(); // Timer for serial strassen computation
  		float **C_strassen = Strassen(A,B,N);
			double one_core_end = MPI_Wtime(); // Timer for serial strassen computation
			printf("Time for Serial Strassen Computation is %f\n", one_core_end - one_core_start);
			// Free memory space
			free(A);
			free(B);
			free(C_strassen);
			MPI_Finalize();
}
