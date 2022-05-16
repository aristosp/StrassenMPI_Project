#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <mpi.h>
#define N 256
// Memory allocation function
float **mem_alloc(int size){
	int i;
	float **array = (float **) malloc (size * sizeof (float *));
    for (i=0; i< size; i++) array[i]=malloc(size*sizeof(float));
	return array;
}
// Matrix Initialization Function
float ** create_matrix(int size){
	// Due to the use of 2d arrays, float pointer is needed. Therefore,
	// using malloc, memory size is being allocated, and then the matrix is
	// randomly initiated
	float ** array = mem_alloc(size);
	int i,j;
	for (i = 0; i < size; i++){
		for (j =0; j<size; j++) array[i][j] = rand() % 10; // normalize values
	}
	return array;
}
// Split Matrix to submatrices
float **splitMatrix(float ** init_matrix,int n,int row,int column){
	// This functions splits a matrix into submatrices, allocating memory as before
	int starting_c = column;
	int i,j;
	float ** arr = mem_alloc(n);
		for(i = 0; i < n; i++){
			column = starting_c;
			for(j = 0; j < n; j++) {
					arr[i][j] = init_matrix[row][column];
					column++;
			}
			row++;
		}
		return arr;
}
// Addition array function
float **addMatrix(float **matrixA,float **matrixB,int n){
	// A simple addition function between two matrices
	float ** addition = mem_alloc(n);
	int i,j;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++) 	addition[i][j] = matrixA[i][j] + matrixB[i][j];
	}
	return addition;
}
// Subtraction array function
float **subMatrix(float **matrixA,float **matrixB,int n){
	// A simple subtraction function between two matrices
	float ** subtraction = mem_alloc(n);
	int i,j;
	for(i = 0; i < n; i++){
		for(j = 0; j < n; j++) 	subtraction[i][j] = matrixA[i][j] - matrixB[i][j];
	}
	return subtraction;
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
				}// Free unneeded space
				free(P1);free(P2);free(P3);free(P4);free(P5);free(P6);free(P7);
				for (i=0;i<new_n;i++){
					for(j=0;j<new_n;j++){
					res[i][j] = C11[i][j];
					res[i][j+new_n] = C12[i][j];
					res[new_n+i][j] = C21[i][j];
					res[new_n+i][new_n+j] = C22[i][j];
					}
				}
		    }
		else {res[0][0] = matrixA[0][0]*matrixB[0][0];}
		return res;
}
int main(int argc, char *argv[]) {
			int new_n = N/2,rank,comm_size;
			int i,j;
			srand(0);
			//Matrices Initiation
			float **A = create_matrix(N);
			float **B = create_matrix(N);
			//Memory Allocation for matrices
			float **P1 = mem_alloc(new_n);
			float **P2 = mem_alloc(new_n);
			float **P3 = mem_alloc(new_n);
			float **P4 = mem_alloc(new_n);
			float **P5 = mem_alloc(new_n);
			float **P6 = mem_alloc(new_n);
			float **P7 = mem_alloc(new_n);
			float **C11 = mem_alloc(new_n);
			float **C12 = mem_alloc(new_n);
			float **C21 = mem_alloc(new_n);
			float **C22 = mem_alloc(new_n);
			float **C_parallel = mem_alloc(N);
			float parallel_start;
			//MPI area
			MPI_Init(&argc,&argv);
			MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
			MPI_Comm_rank(MPI_COMM_WORLD,&rank);
		  //Initiate matrices
		if (rank == 0){
			parallel_start = MPI_Wtime();
			for (i=1;i<comm_size;i++){
				// Send matrices A,B to everyother node
				MPI_Send(&(A[0][0]),(N*N),MPI_FLOAT,i,1,MPI_COMM_WORLD);
				MPI_Send(&(B[0][0]),(N*N),MPI_FLOAT,i,2,MPI_COMM_WORLD);
			}
			//Create submatrices for root
			float **A11 = mem_alloc(new_n);
			float **A12 = mem_alloc(new_n);
			float **A21 = mem_alloc(new_n);
			float **A22 = mem_alloc(new_n);
			float **B11 = mem_alloc(new_n);
			float **B12 = mem_alloc(new_n);
			float **B21 = mem_alloc(new_n);
			float **B22 = mem_alloc(new_n);
			for (i = 0; i < new_n; i++) {
				for(j = 0; j < new_n; j++){
					A11[i][j] = A[i][j];
					A12[i][j] = A[i][j + new_n];
					A21[i][j] = A[i + new_n][j];
					A22[i][j] = A[i + new_n][j + new_n];
					B11[i][j] = B[i][j];
					B12[i][j] = B[i][j + new_n];
					B21[i][j] = B[i + new_n][j];
					B22[i][j] = B[i + new_n][j + new_n];
				}
			}
			free(A);free(B);
			// Depending on the number of roots, calculate accordingly the
			// Si, Pi
			if (comm_size == 2){
				float **S1 = mem_alloc(new_n);
				float **S2 = mem_alloc(new_n);
				float **S3 = mem_alloc(new_n);
				float **S4 = mem_alloc(new_n);
				for (i=0;i<new_n;i++){
					for(j=0;j<new_n;j++){
						S1[i][j] = B12[i][j] - B22[i][j];
						S2[i][j] = A11[i][j] + A12[i][j];
						S3[i][j] = A21[i][j] + A22[i][j];
						S4[i][j] = B21[i][j] - B11[i][j];
					}
				}
				P1 = Strassen(A11,S1,new_n);
				P2 = Strassen(S2,B22,new_n);
				P3 = Strassen(S3,B11,new_n);
				P4 = Strassen(A22,S4,new_n);
				free(A11);free(A12);free(A21);free(A22);
				free(B11);free(B12);free(B21);free(B22);
				free(S1);free(S2);free(S3);free(S4);
			}
			else if (comm_size == 4){
				float **S1 = mem_alloc(new_n);
				float **S2 = mem_alloc(new_n);
				for (i=0;i<new_n;i++){
					for(j=0;j<new_n;j++){
						S1[i][j] = B12[i][j] - B22[i][j];
						S2[i][j] = A11[i][j] + A12[i][j];
					}
				}
			P1 = Strassen(A11,S1,new_n);
			P2 = Strassen(S2,B22,new_n);
			free(A11);free(A12);free(A21);free(A22);
			free(B11);free(B12);free(B21);free(B22);
			free(S1);free(S2);
			}
			else if (comm_size > 4){
				float **S1 = mem_alloc(new_n);
				free(A12);free(A21);free(A22);
				free(B11);free(B21);
				for (i=0;i<new_n;i++){
					for(j=0;j<new_n;j++){
						S1[i][j] = B12[i][j] - B22[i][j];
					}
				}
				P1 = Strassen(A11,S1,new_n);
				free(A11);free(S1);
				free(B12);free(B22);
			}
		}
		else{	// Allocate memory for A,B for every other node
				float **local_A = mem_alloc(N);
				float **local_B = mem_alloc(N);
				MPI_Recv(&(local_A[0][0]),(N*N),MPI_FLOAT,0,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(local_B[0][0]),(N*N),MPI_FLOAT,0,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				// Create local copies of A,B submatrices
				float **local_A11 = mem_alloc(new_n);
				float **local_A12 = mem_alloc(new_n);
				float **local_A21 = mem_alloc(new_n);
				float **local_A22 = mem_alloc(new_n);
				float **local_B11 = mem_alloc(new_n);
				float **local_B12 = mem_alloc(new_n);
				float **local_B21 = mem_alloc(new_n);
				float **local_B22 = mem_alloc(new_n);
				for (i = 0; i < new_n; i++) {
					for(j = 0; j<new_n; j++){
						local_A11[i][j] = local_A[i][j];
						local_A12[i][j] = local_A[i][j + new_n];
						local_A21[i][j] = local_A[i + new_n][j];
						local_A22[i][j] = local_A[i + new_n][j + new_n];
						local_B11[i][j] = local_B[i][j];
						local_B12[i][j] = local_B[i][j + new_n];
						local_B21[i][j] = local_B[i + new_n][j];
						local_B22[i][j] = local_B[i + new_n][j + new_n];
					}
				}
				free(local_A);free(local_B);
				// Again depending on the number of nodes, calculate accordingly
				if (comm_size == 2){
						float **local_S5 = addMatrix(local_A11,local_A22,new_n);
						float **local_S6 = addMatrix(local_B11,local_B22,new_n);
						float **local_S7 = subMatrix(local_B21,local_B22,new_n);
						float **local_S8 = addMatrix(local_B21,local_B22,new_n);
						float **local_S9 = subMatrix(local_A11,local_A21,new_n);
						float **local_S10 = addMatrix(local_B11,local_B12,new_n);
						free(local_A11);free(local_A12);free(local_A21);free(local_A22);
						free(local_B11);free(local_B12);free(local_B21);free(local_B22);
						float **local_P5 = Strassen(local_S5,local_S6,new_n);
						float **local_P6 = Strassen(local_S7,local_S8,new_n);
						float **local_P7 = Strassen(local_S9,local_S10,new_n);
						// Send to root the calculated Pi coefficients
						MPI_Send(&(local_P5[0][0]),(new_n*new_n),MPI_FLOAT,0,5,MPI_COMM_WORLD);
						MPI_Send(&(local_P6[0][0]),(new_n*new_n),MPI_FLOAT,0,6,MPI_COMM_WORLD);
						MPI_Send(&(local_P7[0][0]),(new_n*new_n),MPI_FLOAT,0,7,MPI_COMM_WORLD);
						free(local_S5);free(local_S6);free(local_S7);free(local_S8);
						free(local_P5);free(local_P6);free(local_P7);
				}
				else if (comm_size == 4){
					if (rank == 1){
							float **local_S3 = addMatrix(local_A21,local_A22,new_n);
							float **local_S4 = subMatrix(local_B21,local_B11,new_n);
							float **local_P3 = Strassen(local_S3,local_B11,new_n);
							float **local_P4 = Strassen(local_A22,local_S4,new_n);
							free(local_A11);free(local_A12);free(local_A21);free(local_A22);
							free(local_B11);free(local_B12);free(local_B21);free(local_B22);
							free(local_S3);free(local_S4);
							// Send to root the calculated Pi coefficients
							MPI_Send(&(local_P3[0][0]),(new_n*new_n),MPI_FLOAT,0,3,MPI_COMM_WORLD);
							MPI_Send(&(local_P4[0][0]),(new_n*new_n),MPI_FLOAT,0,4,MPI_COMM_WORLD);
							free(local_P3);free(local_P4);
					}
					else if(rank == 2){
							float **local_S5 = addMatrix(local_A11,local_A22,new_n);
							float **local_S6 = addMatrix(local_B11,local_B22,new_n);
							float **local_S7 = subMatrix(local_A12,local_A22,new_n);
							float **local_S8 = addMatrix(local_B21,local_B22,new_n);
							free(local_A11);free(local_A12);free(local_A21);free(local_A22);
							free(local_B11);free(local_B12);free(local_B21);free(local_B22);
							float **local_P5 = Strassen(local_S5,local_S6,new_n);
							float **local_P6 = Strassen(local_S7,local_S8,new_n);
							free(local_S5);free(local_S6);free(local_S7);free(local_S8);
							// Send to root the calculated Pi coefficients
							MPI_Send(&(local_P5[0][0]),(new_n*new_n),MPI_FLOAT,0,5,MPI_COMM_WORLD);
							MPI_Send(&(local_P6[0][0]),(new_n*new_n),MPI_FLOAT,0,6,MPI_COMM_WORLD);
							free(local_P5);free(local_P6);
					}
					else if(rank == 3){
							float **local_S9 = subMatrix(local_A11,local_A21,new_n);
							float **local_S10 = addMatrix(local_B11,local_B12,new_n);
							float **local_P7 = Strassen(local_S9,local_S10,new_n);
							// Send to root the calculated Pi coefficients
							MPI_Send(&(local_P7[0][0]),(new_n*new_n),MPI_FLOAT,0,7,MPI_COMM_WORLD);
							free(local_S9);free(local_S10);free(local_P7);
					}
				}
				else if (comm_size > 4){ // each node calculates one Pi, and then sends it to root
					if (rank == 1){
							float **local_S2 = addMatrix(local_A11,local_A12,new_n);
							float **local_P2 = Strassen(local_S2,local_B22,new_n);
							MPI_Send(&(local_P2[0][0]),(new_n*new_n),MPI_FLOAT,0,2,MPI_COMM_WORLD);
					}
					else if (rank == 2){
							float **local_S3 = addMatrix(local_A21,local_A22,new_n);
							float **local_P3 = Strassen(local_S3,local_B11,new_n);
							MPI_Send(&(local_P3[0][0]),(new_n*new_n),MPI_FLOAT,0,3,MPI_COMM_WORLD);
					}
					else if (rank == 3){
							float **local_S4 = subMatrix(local_B21,local_B11,new_n);
							float **local_P4 = Strassen(local_A22,local_S4,new_n);
							MPI_Send(&(local_P4[0][0]),(new_n*new_n),MPI_FLOAT,0,4,MPI_COMM_WORLD);
					}
					else if(rank == 4){
							float **local_S5 = addMatrix(local_A11,local_A22,new_n);
							float **local_S6 = addMatrix(local_B11,local_B22,new_n);
							float **local_P5 = Strassen(local_S5,local_S6,new_n);
							MPI_Send(&(local_P5[0][0]),(new_n*new_n),MPI_FLOAT,0,5,MPI_COMM_WORLD);
					}
					else if(rank == 5){
							float **local_S7 = subMatrix(local_A12,local_A22,new_n);
							float **local_S8 = addMatrix(local_B21,local_B22,new_n);
							float **local_P6 = Strassen(local_S7,local_S8,new_n);
							MPI_Send(&(local_P6[0][0]),(new_n*new_n),MPI_FLOAT,0,6,MPI_COMM_WORLD);
					}
					else if(rank == 6){
							float **local_S9 = subMatrix(local_A11,local_A21,new_n);
							float **local_S10 = addMatrix(local_B11,local_B12,new_n);
							float **local_P7 = Strassen(local_S9,local_S10,new_n);
							MPI_Send(&(local_P7[0][0]),(new_n*new_n),MPI_FLOAT,0,7,MPI_COMM_WORLD);
					}
				}
			}
		if (rank == 0){
			// Depending on the number of nodes, receive accordingly Pi coeffiecients
			if (comm_size == 2){
				MPI_Recv(&(P5[0][0]),(new_n*new_n),MPI_FLOAT,1,5,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P6[0][0]),(new_n*new_n),MPI_FLOAT,1,6,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P7[0][0]),(new_n*new_n),MPI_FLOAT,1,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			}
			else if (comm_size == 4){
				MPI_Recv(&(P3[0][0]),(new_n*new_n),MPI_FLOAT,1,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P4[0][0]),(new_n*new_n),MPI_FLOAT,1,4,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P5[0][0]),(new_n*new_n),MPI_FLOAT,2,5,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P6[0][0]),(new_n*new_n),MPI_FLOAT,2,6,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P7[0][0]),(new_n*new_n),MPI_FLOAT,3,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			}
			else if (comm_size > 4){
				MPI_Recv(&(P2[0][0]),(new_n*new_n),MPI_FLOAT,1,2,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P3[0][0]),(new_n*new_n),MPI_FLOAT,2,3,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P4[0][0]),(new_n*new_n),MPI_FLOAT,3,4,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P5[0][0]),(new_n*new_n),MPI_FLOAT,4,5,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P6[0][0]),(new_n*new_n),MPI_FLOAT,5,6,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
				MPI_Recv(&(P7[0][0]),(new_n*new_n),MPI_FLOAT,6,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			}
			// Calculate Ci,j submatrices
			for (i = 0; i<new_n; i++){
				for (j = 0;j<new_n; j++){
				C11[i][j] = P5[i][j] + P4[i][j] - P2[i][j] + P6[i][j];
				C12[i][j] = P1[i][j] + P2[i][j];
				C21[i][j] = P3[i][j] + P4[i][j];
				C22[i][j] = P5[i][j] + P1[i][j] - P3[i][j] - P7[i][j];
				}
			}
			// Create one matrix C, consisting of Ci,j
			for (i=0;i<new_n;i++){
				for(j=0;j<new_n;j++){
					C_parallel[i][j] = C11[i][j];
					C_parallel[i][j+new_n] = C12[i][j];
					C_parallel[new_n+i][j] = C21[i][j];
					C_parallel[new_n+i][new_n+j] = C22[i][j];
				}
			}
			float parallel_end = MPI_Wtime();
			printf("With %d core(s), parallel calcuation time is %f \n",comm_size,parallel_end - parallel_start);
		}
		MPI_Finalize();
}
