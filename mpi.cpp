#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <vector>
#include "common.h"
using namespace std;

#define density 0.0005
#define mass    0.01
#define cutoff  0.01
#define min_r   (cutoff/100)
#define dt      0.0005


int procNum(particle_t &p, int nx_proc, double local_size_proc_x, double local_size_proc_y) 
{
    return 1; // FILL THIS IN
}

int binNum(particle_t &p, int bpr, double bprsize, double off_x, double off_y) 
{
    return 1; // FILL THIS IN
}

//
//  benchmarking program
//
int main( int argc, char **argv )
{    
    int navg,nabsavg=0;
    double dmin, absmin = 1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg;
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );

    
    //
    //  set up MPI
    //-
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  Initiallize processor layout .. try to get as square as possible
    //
    int nx_proc, ny_proc;
    ny_proc=1;
    nx_proc=n_proc; 
    //
    // Initialize partitions and sizes
    //
    
    set_size( n );
    double size = sqrt( density*n );
    int tmp_bpr = floor(size/cutoff);
    double tmptmp = tmp_bpr/n_proc;
    int bpr = n_proc * floor ( tmptmp );
    int bprx = bpr / nx_proc;
    int bpry = bpr / ny_proc;
    double binsize = size / bpr;        
    double proc_s_x = binsize * bprx;
    double proc_s_y = binsize * bpry;
    
    //
    // Calculating neccesary ghostzones for each processor
    // 
    int gsl = 1, gsr = 1 ,gsu = 1, gsd = 1;
    if (rank < nx_proc)
        gsd = 0;
    if ((rank % nx_proc) == 0)
        gsl = 0;
    if ((rank % nx_proc) == (nx_proc-1))
        gsr = 0;
    if (rank >= nx_proc*(ny_proc-1))
        gsu = 0;
    
    int numbins = (bprx+gsl+gsr)*(bpry+gsd+gsu);
    double off_x = rank%nx_proc * proc_s_x;
    double off_y = rank/nx_proc * proc_s_y;
        
                
    // Allocating memory for Send data
    int gh_s_l, gh_s_r;
    particle_t *gh_v_l = (particle_t*) malloc( 3*(bpry+gsd+gsu) * sizeof(particle_t) );
    particle_t *gh_v_r = (particle_t*) malloc( 3*(bpry+gsd+gsu) * sizeof(particle_t) );
    MPI_Request req_l;
    MPI_Request req_r;
    
    // Allocating memory for Receive data
    int gh_r_s_l, gh_r_s_r;
    particle_t *gh_r_v_l = (particle_t*) malloc( 3*(bpry+gsd+gsu) * sizeof(particle_t) );
    particle_t *gh_r_v_r = (particle_t*) malloc( 3*(bpry+gsd+gsu) * sizeof(particle_t) );
    MPI_Request r_req_l;
    MPI_Request r_req_r;
    MPI_Status r_st_l;
    MPI_Status r_st_r;
    
    // Some data on the splitting of the domain
    if(rank == 0)
        printf("size= %lf numbins = %d bprx = %d bpry = %d bpr = %d\n",size,numbins,bprx,bpry,bpr);
        
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;

    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    particle_t *proc_split_particles = (particle_t*) malloc ( 2 * n * sizeof(particle_t)); 
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
    
    //
    //  set up the data partitioning across processors
    //
    
    int particle_per_proc = 2 * n / n_proc;
    
    int *partition_offsets = (int*) malloc( n_proc * sizeof(int) );
    int *partition_sizes = (int*) malloc( n_proc * sizeof(int) );
    int *partition_ns = (int*) malloc( n_proc * sizeof(int) );

    for( int i = 0; i < n_proc; i++ )
    {
        partition_offsets[i] = i * particle_per_proc;
        partition_sizes[i] = particle_per_proc;
        partition_ns[i]=0;
    }

    //
    // Initializing particles and doing initial binning acording to processors
    //
    if( rank == 0 )
    {   
        init_particles( n, particles );
        for (int i = 0; i < n; i++)
                {
                int tmp_i=procNum(particles[i],nx_proc,proc_s_x,proc_s_y);
            proc_split_particles[ tmp_i * particle_per_proc + partition_ns[tmp_i]] = particles[i];
            partition_ns[tmp_i]++;
        }
    }
    
    //
    //  allocate storage for local partition
    //
    int nlocal=0;
    MPI_Scatter(partition_ns,1,MPI_INT,&nlocal,1,MPI_INT,0,MPI_COMM_WORLD);
    
    particle_t *local = (particle_t*) malloc( particle_per_proc * sizeof(particle_t) );
    vector<particle_t*> *bins = new vector<particle_t*>[numbins];

    
    //
    //  Distributing particles 
    //
    MPI_Scatterv( proc_split_particles, partition_sizes, partition_offsets, PARTICLE, local, particle_per_proc, PARTICLE, 0, MPI_COMM_WORLD );
    
    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    for( int step = 0; step < NSTEPS; step++ )
    {
      navg = 0;
      dmin = 1.0;
      davg = 0.0;
      // clear bins at each time step
      for (int m = 0; m < numbins; m++)
        bins[m].clear();

      // place particles in bins
        for (int i = 0; i < nlocal; i++) 
            bins[binNum(local[i],bprx+gsl+gsr,binsize,off_x-gsl*binsize,off_y-gsd*binsize)].push_back(local + i);
        
        // Initializing ghost sizes
        gh_s_l = 0;
        gh_s_r = 0;
        
        gh_r_s_l = 0;
        gh_r_s_r = 0;
        
        // Checking which particles are on the edge and packing them
        for (int i=0; i<bpry+gsd+gsu; i++)
        {
            if (gsr)
                for(int k=0; k<bins[bprx-1+gsl+i*(bprx+gsl+gsr)].size();k++)
                {
                    gh_v_r[gh_s_r] = *bins[bprx-1+gsl+i*(bprx+gsl+gsr)][k];
                    gh_s_r++;
                }
            if (gsl)
                for(int k=0; k<bins[gsl+i*(bprx+gsl+gsr)].size();k++)
                {
                    gh_v_l[gh_s_l] = *bins[gsl+i*(bprx+gsl+gsr)][k];
                    gh_s_l++;
                }
        }

        // FILL OUT THESE SENDS/RECVS
        // receive numbers
        // MPI_Irecv gh_r_s_l FILL OUT THE FUNCTION
        // MPI_Irecv gh_r_s_r FILL OUT THE FUNCTION

        //send numbers
        // MPI_Isend gh_s_l FILL OUT THE FUNCTION
        // MPI_Isend gh_s_r FILL OUT THE FUNCTION
        
        // wait for receipt of numbers
        // MPI_Wait FILL OUT THE FUNCTION
        // MPI_Wait FILL OUT THE FUNCTION
        

        
        //
        // Posting receives for ghostzones
        //
        // MPI_Irecv gh_r_v_l FILL OUT THE FUNCTION
        // MPI_Irecv gh_r_v_r FILL OUT THE FUNCTION
        
        //
        // Sending ghostzones to other processors
        //
        // MPI_Isend gh_r_v_l FILL OUT THE FUNCTION
        // MPI_Isend gh_r_v_r FILL OUT THE FUNCTION

        // waiting for receipt of ghostzones
        // MPI_Wait FILL OUT THE FUNCTION
        // MPI_Wait FILL OUT THE FUNCTION
        
        //
        // Add ghostzone particles to bins
        //
        for (int i = 0; i < gh_r_s_l; i++) 
            bins[binNum(gh_r_v_l[i],bprx+gsl+gsr,binsize,off_x-gsl*binsize,off_y-gsd*binsize)].push_back(gh_r_v_l + i);
            
        for (int i = 0; i < gh_r_s_r; i++) 
            bins[binNum(gh_r_v_r[i],bprx+gsl+gsr,binsize,off_x-gsl*binsize,off_y-gsd*binsize)].push_back(gh_r_v_r + i);
            
        //
        //  compute forces
        //
        for( int p = 0; p < nlocal; p++ )
        {
            local[p].ax = local[p].ay = 0;
          
            // find current particle's bin, handle boundaries
            int cbin = binNum( local[p],bprx+gsl+gsr,binsize,off_x-gsl*binsize,off_y-gsd*binsize);
            int lowi = -1, highi = 1 ,lowj = -1, highj = 1;
            if ((cbin < (bprx+gsl+gsr))&&gsd==0)
                lowj = 0;
            if (((cbin % (bprx+gsl+gsr)) == 0)&&gsl==0)
                lowi = 0;
            if (((cbin % (bprx+gsl+gsr)) == (bprx+gsl+gsr-1))&&gsr==0)
                highi = 0;
            if ((cbin >= (bprx+gsl+gsr)*(bpry+gsd-1))&&gsu==0)
                highj = 0;
            
            // apply nearby forces
            for (int j = lowj; j <= highj; j++)
                for (int i = lowi; i <= highi; i++)
                {
                    // APPLY FORCE FROM EVERY PARTICLE TO EVERY PARTICLE IN SURROUNDING BINS
                }
        }
            
        if( find_option( argc, argv, "-no" ) == -1 )
        {

          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);


          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }
    
        
    //
        //  move particles
        //
        for( int p = 0; p < nlocal; p++ ) 
            move( local[p]);
        
        
        // Initialising numbers
        gh_s_l = 0;
        gh_s_r = 0;
        
        gh_r_s_l = 0;
        gh_r_s_r = 0;
        
        int tmp_ch;
        
        // Check particles if still local/move to correct processor
        for (int i=0; i < nlocal; i++ )
        {
            tmp_ch=0;
            if (local[i].x < off_x) 
            {
                gh_v_l[gh_s_l] = local[i];
                gh_s_l++;
                tmp_ch=1;
            }
            
            if (local[i].x >= off_x + proc_s_x)
            {
                gh_v_r[gh_s_r] = local[i];
                gh_s_r++;
                tmp_ch=1;
            }
            
            if(tmp_ch) 
            {
                local[i]=local[nlocal-1];
                nlocal--;
                i--;
            }
        }

        // FILL OUT THESE SENDS/RECVS
        // receive numbers
        // MPI_Irecv gh_r_s_l FILL OUT THE FUNCTION
        // MPI_Irecv gh_r_s_r FILL OUT THE FUNCTION

        //send numbers
        // MPI_Isend gh_s_l FILL OUT THE FUNCTION
        // MPI_Isend gh_s_r FILL OUT THE FUNCTION
        
        // wait for receipt of numbers
        // MPI_Wait FILL OUT THE FUNCTION
        // MPI_Wait FILL OUT THE FUNCTION
        

        
        //
        // Posting receives for ghostzones
        //
        // MPI_Irecv gh_r_v_l FILL OUT THE FUNCTION
        // MPI_Irecv gh_r_v_r FILL OUT THE FUNCTION
        
        //
        // Sending ghostzones to other processors
        //
        // MPI_Isend gh_r_v_l FILL OUT THE FUNCTION
        // MPI_Isend gh_r_v_r FILL OUT THE FUNCTION

        // waiting for receipt of ghostzones
        // MPI_Wait FILL OUT THE FUNCTION
        // MPI_Wait FILL OUT THE FUNCTION
        
        //
        // Add move particles to local
        //
        for (int i = 0; i < gh_r_s_l; i++)
        {
            local[nlocal] = gh_r_v_l[i];
            nlocal++;
        }
            
        for (int i = 0; i < gh_r_s_r; i++)
        {
            local[nlocal] = gh_r_v_r[i];
            nlocal++;
        }
                

        // Leave this part alone
      if( find_option( argc, argv, "-no" ) == -1 )      
      {
        if( (step%SAVEFREQ) == 0 )
        {
            MPI_Gather(&nlocal,1,MPI_INT,partition_ns,1,MPI_INT,0,MPI_COMM_WORLD);
            if (rank == 0)
            {   
                int tmp_nsum=0;
                for(int i=0;i<n_proc;i++)
                {
                    partition_offsets[i]=tmp_nsum;
                    tmp_nsum+=partition_ns[i];
                }
            }
            
            MPI_Gatherv( local, nlocal, PARTICLE, particles, partition_ns, partition_offsets, PARTICLE,0, MPI_COMM_WORLD );
            if (rank == 0)
                save( fsave, n, particles );
        }
       }
       // All the way to here

        MPI_Barrier(MPI_COMM_WORLD);
    }
    simulation_time = read_timer( ) - simulation_time;
    
    
    if (rank == 0) {
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -the minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation were particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");

      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
 
    //
    //  release resources
    //
    free(gh_v_l);
    free(gh_v_r);
    
    free(gh_r_v_l);
    free(gh_r_v_r);
    
    free( partition_ns);
    free( partition_offsets );
    free( partition_sizes );
    free( local );
    free( particles );
    free( proc_split_particles);
    delete [] bins;
    if( fsave )
        fclose( fsave );
    if (fsum )
        fclose (fsum );
    
    MPI_Finalize( );
    
    return 0;
}
