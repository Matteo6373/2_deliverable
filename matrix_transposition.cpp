#include <iostream>
#include <cstdlib>
#include <time.h>
#include <mpi.h>
#include <vector>
#define TRY 5
using namespace std;
//check symmetric matrix
int checkSym(int n, const vector<double>& M){
    int i,j;
    for (i = 0; i < n; i++) {
        for(j = i+1; j < n; j++) {
            if(M[i*n+j]!=M[j*n+i])
                return 0; // matrix is not simmetric
        }
    }
    return 1;//the matrix is simmetric
}

void matTranspose(int n, const vector<double>& source,vector<double>& dest){
    int i,j;
    for (i = 0; i < n; i++) {
        for(j = 0; j < n; j++) {
                dest[i*n+j] = source[j*n+i];
        }
    }
}
void matTransposeMPIrow(int n, const std::vector<double>& source, std::vector<double>& dest, int myrank, int size){
    //matrice divisa per righe
    int rows_per_process = n / size;  // Ogni processo gestisce un sottoinsieme delle righe
    int size_matrix = (myrank<n%size)?(rows_per_process+1)*n:rows_per_process*n;  
    int sendcounts[size]; 
    int displs[size];
    displs[0] = 0;
    for(int i=0;i<size;i++){
      sendcounts[i] = (i<n%size)?(rows_per_process+1)*n:rows_per_process*n;
      if(i>0){
        displs[i] = displs[i-1]+sendcounts[i-1];
      }
    }
    vector<double> sub_matrix(size_matrix); 
    MPI_Scatterv(&source[0], sendcounts, displs, MPI_DOUBLE, &sub_matrix[0], sendcounts[myrank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    // Creiamo una matrice trasposta locale
    vector<double> sub_transposed_matrix(size_matrix);
    for (int i = 0; i < size_matrix/n; i++) {  
        for (int j = 0; j < n; j++) {
            // Trasposizione locale 
            sub_transposed_matrix[size_matrix/n*j + i] = sub_matrix[i * n + j];                          
        }
    }
    for(int i=0;i<size;i++){
      sendcounts[i] /=n;
      displs[i]/= n;
    }   
    for(int i=0;i<n;i++){
      MPI_Gatherv(&sub_transposed_matrix[size_matrix/n*i], sendcounts[myrank], MPI_DOUBLE,&dest[i*n], sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
}
void matTransposeMPIcolumn(int n, const std::vector<double>& source, std::vector<double>& dest, int myrank, int size) {
    //matrice divisa per colonne
    MPI_Datatype sendcol,sendcoltype;
    int columns_per_process = n/size;
    int size_matrix = (myrank<n%size)?(columns_per_process+1)*n:columns_per_process*n;  
    int sendcounts[size]; 
    int displs[size];
    displs[0] = 0;
    for(int i=0;i<size;i++){
      sendcounts[i] = (i<n%size)?(columns_per_process+1):columns_per_process;
      if(i>0){
        displs[i] = displs[i-1]+sendcounts[i-1];
      }
    }
    vector<double> sub_matrix(size_matrix); 
    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &sendcol);
    MPI_Type_commit(&sendcol);
    MPI_Type_create_resized(sendcol, 0, 1*sizeof(double), &sendcoltype);
    MPI_Type_commit(&sendcoltype);
    MPI_Scatterv(&source[0], sendcounts, displs, sendcoltype  , &sub_matrix[0], sendcounts[myrank]*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    for(int i=0;i<size;i++){
      sendcounts[i] *=n;
      displs[i]*= n;
    } 
    MPI_Gatherv(&sub_matrix[0], sendcounts[myrank], MPI_DOUBLE,&dest[0], sendcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Type_free(&sendcol);
    MPI_Type_free(&sendcoltype);
    
}
void matTransposeMPIblock(int n, const std::vector<double>& source, std::vector<double>& dest, int myrank, int size,int block_size) {
    int n_block_per_rows = n / block_size; //numeri di blocchi per ogni riga e colonna
    int n_matrix_per_process = n_block_per_rows*n_block_per_rows/size; //numero di matrici per ogni processo
    int rank_to_sent = 0;
    int tag = 0;
    vector<double> sub_matrix(block_size*block_size*n_matrix_per_process); //array di matrice
    //distribuzione della matrice
    if(myrank==0){
      for(int i=0;i<n_block_per_rows;i++){
        for(int j=0;j<n_block_per_rows;j++){
          //se rank diverso da 0 mando il blocco al processo rank_to_sent
          if(rank_to_sent != 0){
            for(int k=0;k<block_size;k++){
              MPI_Send(&source[i*n*block_size+j*block_size+k*n], block_size, MPI_DOUBLE, rank_to_sent, 0, MPI_COMM_WORLD);
            }
          }
          //se rank da mandare uguale a 0 salva il blocco nella metrice del processo
          else{
            for(int k=0;k<block_size;k++){
              for(int x=0;x<block_size;x++){
                sub_matrix[tag*block_size*block_size+k*block_size+x] = source[i*n*block_size+j*block_size+k*n+x];
              }
            }
          }
          //incremento in modo circolare
          rank_to_sent = (rank_to_sent+1)%size;
          if(rank_to_sent == 0){
            tag++;
          }
        }
      }
    //processi diversi da 0 ricevono i blocchi e li salvano in maniera sequenziale nell'array di matrici
    }else{
      for(int i=0;i<n_matrix_per_process;i++){
        for(int j=0;j<block_size;j++){
          MPI_Recv(&sub_matrix[i*block_size*block_size+j*block_size], block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
      }
    }
    //calcola della trasposta
    vector<double> sub_matrix_transposed(block_size*block_size*n_matrix_per_process);
    for(int i=0;i<n_matrix_per_process;i++){
      for(int j=0;j<block_size;j++){
        for(int k=0;k<block_size;k++){
          sub_matrix_transposed[block_size*block_size*i+j*block_size+k] = sub_matrix[block_size*block_size*i+k*block_size+j];
        }
      }
    }
    //recupero dati
    int step = n_block_per_rows / size; //usato per calcolare l'indice dell'array di matrice da mandare
    int rank_to_recieve = 0;
    //rank 0 recupera i blocchi e li salva nella matrice finale
    if(myrank==0){
      for(int i=0;i<n_block_per_rows;i++){
        rank_to_recieve = i%size;
        //se rank da ricevere è diverso da 0 deve riceverlo dal processo rank_to_recieve
        if(rank_to_recieve!=0){
          for(int j=0;j<n_block_per_rows;j++){
            for(int k=0;k<block_size;k++){
              MPI_Recv(&dest[i*block_size*n+j*block_size+k*n], block_size, MPI_DOUBLE, rank_to_recieve, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
          }
        //se rank da ricevere uguale 0 prendo il blocco dall'array di matrice trasposta.
        }else{
          int num_block = 0;
          for(int j=0;j<n_block_per_rows;j++){
            num_block = i/size+j*step;  
            for(int k=0;k<block_size;k++){
              for(int x=0;x<block_size;x++){
                dest[i*block_size*n+j*block_size+k*n+x]= sub_matrix_transposed[num_block*block_size*block_size+k*block_size+x];
              }
            }
          }
        }
      }
    }
    //processi non rank 0 mandano i blocchi non in ordine sequenazile ma in modo che i dati vengono ricevuti gia in ordine trasposto
    else{
      int num_block = 0;
      for(int i=0;i<n_matrix_per_process/n_block_per_rows;i++){
        for(int j=0;j<n_block_per_rows;j++){
          num_block = i+j*step;  
          for(int j=0;j<block_size;j++){
            MPI_Send(&sub_matrix_transposed[num_block*block_size*block_size+j*block_size], block_size, MPI_DOUBLE, 0, 0 ,MPI_COMM_WORLD);
          }
        }
      }
    }
    
}

int check(int n, const vector<double>& m1,vector<double>& m2){
  int i,j;
  for(i=0;i<n;i++){
    for(j=0;j<n;j++){
      if(m1[i*n+j]!=m2[i*n+j]) return 0;
    }
   }
   return 1;
}

int main(int argc, char **argv) {
    srand(time(NULL));
    MPI_Init(&argc, &argv);
    int myrank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int block_size = 32;
    if(argc!=2){if(myrank==0){printf("inserisci dimensione array\n");return 0;}}
    int n = atoi(argv[1]);
    
    vector<double> source(n * n);
    vector<double> dest_seq(n * n);
    vector<double> dest_mpi_col(n * n);
    vector<double> dest_mpi_row(n * n);
    double Time_Sequential[TRY+1]; //array with time for sequential
    double Time_mpi_row[TRY+1];   //array with time for mpi with block
    double Time_mpi_column[TRY+1];   //array with time for mpi
    int sim = 0;
    
    for(int k=0;k<TRY+1;k++){
      if(myrank==0){
        // Initialize matrices source with random values
        for (int i = 0; i < n; i++) {
          for(int j = 0; j < n; j++) {
            source[i*n+j] = (rand()%10000)/100.0;
          }
        }
        if(checkSym(n,source)==1){
          printf("matrice simmetrica\n");
          sim = 1;
        }
      }
      if(sim == 1){
        MPI_Finalize();
        return 0;
      }
      if(myrank==0){
        //---------------------------------------------------------------------------    
        //SEQUENTIAL PART
        //---------------------------------------------------------------------------     
        double wt1 = MPI_Wtime();
        matTranspose(n,source,dest_seq);
        double wt2 = MPI_Wtime();
        Time_Sequential[k] = (wt2-wt1);
      }
      //---------------------------------------------------------------------------    
      //MPI PART column
      //---------------------------------------------------------------------------  
      MPI_Barrier(MPI_COMM_WORLD);  
      double wt1 = MPI_Wtime();
      matTransposeMPIcolumn(n,source,dest_mpi_col,myrank,size);
      double wt2 = MPI_Wtime();
      if(myrank==0){
        //save time in array
        Time_mpi_column[k] = (wt2-wt1);
      }
      //---------------------------------------------------------------------------    
      //MPI PART ROW
      //---------------------------------------------------------------------------  
      MPI_Barrier(MPI_COMM_WORLD);
      wt1=MPI_Wtime();
      if(size<32){ //con tanto processi il tempo di esecuzione aumenta di molto.  
        matTransposeMPIrow(n,source,dest_mpi_row,myrank,size);
      }
      wt2=MPI_Wtime();
      if(myrank==0){
        //save time in array
        Time_mpi_row[k] = (wt2-wt1);
      }
    }
    
    if(myrank==0){
      //calculate of average  
      double avg_S = 0;
      double avg_M_C = 0;
      double avg_M_R = 0;
      //not consider the first value because is often an outlier
      for(int k=1;k<TRY+1;k++){
        avg_S += Time_Sequential[k];
      }
      avg_S /= TRY;
      for(int k=1;k<TRY+1;k++){
        avg_M_C += Time_mpi_column[k];
      }
      avg_M_C /= TRY;
      for(int k=1;k<TRY+1;k++){
        avg_M_R += Time_mpi_row[k];
      }
      avg_M_R /= TRY;
      int data_transfered_seq = n*n*sizeof(double)*2; //ogni cella della matrice viene accessa in lettura e in scrittura
      int data_transfered_col = data_transfered_seq + size*sizeof(int)*8; // come seq piu i vettori per gestire lo scambio delle righe
      int data_transfered_row = data_transfered_seq*2 + size*sizeof(int)*8; // viene aggiunta una matrice intermedia per la trasposizione
      //print of result
      printf("          |  sequential time   |   mpi time(column)|   mpi time(row)   |  speedup(column)  | efficiency(column) |   speedup(row)    |  efficiency(row)\n");
      printf("-----------------------------------------------------------------------------------------------------------------------------------------------------------\n");
      for(int k=1;k<TRY+1;k++){
          printf("%d' attempt|    ",k);
          printf( "%.6f sec    |   ", Time_Sequential[k]);
          printf( "%.6f sec    |   ", Time_mpi_column[k]);
          printf( "%.6f sec    |   ", Time_mpi_row[k]);
          printf( "%.6f        |   ", Time_Sequential[k]/Time_mpi_column[k]);
          printf( "%.6f        |   ", Time_Sequential[k]/Time_mpi_column[k] /size*100);
          printf( "%.6f        |   ", Time_Sequential[k]/Time_mpi_row[k]);
          printf( "%.6f \n", Time_Sequential[k]/Time_mpi_row[k] /size*100);
      }
      printf("------------------------------------------------------------------------------------------------------------------------------------------------------------\n");
      printf("average   |    %.6f sec    |   %.6f sec    |   %.6f sec    |   %.6f        |   %.6f        |   %.6f        |   %.6f \n\n",avg_S,avg_M_C,avg_M_R,avg_S/avg_M_C,avg_S/avg_M_C/size*100,avg_S/avg_M_R,avg_S/avg_M_R/size*100);
      
      printf("          |  bandwidth(seq)    |  bandwidth(column) |  bandwidth(row)\n");
      printf("--------------------------------------------------------------------\n");
      for(int k=1;k<TRY+1;k++){
          printf("%d' attempt|    ",k);
          printf( "%.6f GB/s   |   ", data_transfered_seq/Time_Sequential[k]/1073741824);//1073741824 serve per passare da byte a GB
          printf( "%.6f GB/s   |   ", data_transfered_col/Time_mpi_column[k]/1073741824);
          printf( "%.6f GB/s\n", data_transfered_row/Time_mpi_row[k]/1073741824);
      }
      printf("---------------------------------------------------------------------\n");
      printf("average   |    %.6f GB/s   |   %.6f GB/s   |   %.6f GB/s\n",data_transfered_seq/avg_S/1073741824,data_transfered_col/avg_M_C/1073741824,data_transfered_row/avg_M_R/1073741824);
      /*
      if(check(n,dest_seq,dest_mpi_col)==0){
        printf("sbagliato\n");
      }
      else{
        printf("giusto\n");
      }
      if(check(n,dest_seq,dest_mpi_row)==0){
        printf("sbagliato\n");
      }
      else{
        printf("giusto\n");
      }
      
      if(myrank==0){
       printf("-----------------------------------------------------------------------\n");
      for (int i = 0; i < n; i++) {
          for(int j = 0; j < n; j++) {
            printf("%.2f ",source[i*n+j]);
          }
          printf("\n");
        }
        printf("-----------------------------------------------------------------------\n");
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
              printf("%.2f ",dest_mpi_col[i*n+j]);
            }
            printf("\n");
        }
      }        
      */
    }    
    MPI_Finalize();
    return 0;
}