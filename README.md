The code was tested on the cluster. To run the code, simply copy the folder to the cluster and execute the file hello.pbs from within the copied folder. 
This will recompile and run the matrix_transposition.c code for each pair of matrix size and number of threads. Running all combinations takes about 2 minutes. To execute the code with defined 
size and n_threads, first load the gcc91 and mpich--3.2.1--gcc91 module, then compile the code with gcc matrix_transposition.c -o mt -fopenmp -O2, and finally run it with ./mt (size), e.g., ./mt 1024.
