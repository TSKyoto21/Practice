//////////////////////////////////////////
//Coded by T.Taniguchi
//Modified and Parallelized by K.Harada
//Modified by T.Sato(20170601)
//Ver 1.4 2014/09/01
//////////////////////////////////////////
#include "mpi.h"
#include "Pasta.h"

#define MCW MPI_COMM_WORLD
int debug_step;
int main(int argc, char** argv){

  MPI_Init( &argc, &argv );
  try{
 
    IOParameterSet param(argv[1]); //param.info();
    static const int N_particle = param.int_value("N_particle");
    static const int MAXSTEP    = param.value("MAXSTEP");
    static const int Num_of_polymer_data   = param.int_value("Num_of_polymer_data");
    static const int datastep_polymer_data = int( MAXSTEP/Num_of_polymer_data );
    static const int Equilibration_step  = param.int_value("Equilibration_step");
    static const int Data_step           = param.int_value("Data_step");
    static const int IntervalStepOfData  = param.int_value("IntervalStepOfData");
    static const int time_interval       = param.int_value("time_interval");
    static const double T_ref            = param.value("T_ref");
    static const double T                = param.value("T");
    static const double delta_T          = T - T_ref;
    static const double C1               = param.value("C1");
    static const double C2               = param.value("C2");
    static const double aT               = exp(-2.302585093*C1*delta_T/(C2+delta_T));
    static const double bT               = T / T_ref;
    static const double ave_data         = param.value("ave_data");
    static const int Analysis_start_step = (int)( Equilibration_step+ave_data*Data_step+0.1);
    static int counter = 0;
    static bool output_data = false;
    
    int NOW=0, NEW=1;

    double ***D_all;    // D[id:Smooth_Particle][a][b] Dab = dv_a / dx_b
    double **sigma_all; // sigma_ab (Slip-link model), sigma[id:Smooth_Particle][xx or yy or ..]
    double **SS_all;    // S_ab S[id:Smooth_Particle][xx or yy or ..]
    double ***D;   //D in a PE
    double **sigma;//sigma in a PE
    double **SS;   //SS ina PE
    //--------------------------------------------------------------
    //<< Slip-link models >>
    //--------------------------------------------------------------
    int ***N_hook;      // Number of hooking or re-hooking
    //                  // N_hook[id:Smooth_Particle][hook: 0, re-hook: 1][id:PolymerChain]
    int **Ze;           // number of entanglement of each polymer
    //                  // Ze[id:Smooth_Particle][id:PolymerChain]
    int **Zeq;          // number of entanglement of each polymer
    //                  // Zeq[id:Smooth_Particle][id:PolymerChain]
    int ****SL_pair;    // information about a pair of slip-links
    //                  // SL_pair[id:Smooth_Particle][id:PolymerChain][id:slip-link]
    //                  //        [0:id_PolymerChain or 1:id_slip-link]
    //
    double *****SL_pos; // coordinate of slip-link
    //                  // SL_po[id:Smooth_Particle][NOW_NEW][id:PolymerChain]
    //                  //      [id:slip-link][x or y or z]
    //
    double ****r;       // vector between two slip-links
    //                  // SL_rr[id:Smooth_Particle][id:PolymerChain][id:segment][x or y or z]
    //
    double ****s;       // length of the tails
    //                  // s[id:Smooth_Particle][NOW:NEW][id:PolymerChain][0:head or 1:tail]
    //
    double ***L;        // length of each polymer 
    //                  // L[id:Smooth_Particle][NOW:NEW][id:Polymerchain]
    //
    double **w;          // random number 
    //                   // w[id:Smooth_Particle][id:PolymerChain*2]
    int ***Entanglement; //Chack of Entanglement (-1 or 1)[id:Smooth_Particle][id:PolymerChain][id:slip-link]
    //--------------------------------------------------------------
    //for alalyze
    double** sg_sum;
    double** SS_sum;
    double*  Z_sum;
    double*  L_sum;
    double*  d2_Z;
    double*  d2_L;
    double** ave_s; //20180112
    double** dis_s; //20180112
    int*     counter_analyze;
    //-------------------------------------------------------------
    //MPI
    double *D_local_1D, *D_all_1D;
    double *sigma_local_1D, *sigma_all_1D;
    double *SS_local_1D, *SS_all_1D;
    double Ts1, Ts2, Tf;//time_start time_finish
    int size, rank;
    int *N_in_PE;       //num of particle in a Processor Element
    int *N_disp;
    int *N_in_PE_D;     //N_in_PE_D = N_in_PE*SPACE_DIM*SPACE_DIM
    int *N_in_PE_sigma, *N_in_PE_SS;//N_in_PE_sigma = N_in_PE*MM  { MM=6 (3D), 4 (2D) }
    int *N_disp_D, *N_disp_sigma, *N_disp_SS;
    //------------------------------------------------------------
    MPI_Barrier( MCW );
    Ts1 = MPI_Wtime();
    
    MPI_Comm_size( MCW, &size );
    MPI_Comm_rank( MCW, &rank );
    
    if( rank==0 ){
      Allocation_PE1( param, D_all, sigma_all, SS_all, D_all_1D, sigma_all_1D, SS_all_1D );
    }

    Allocation_MPI( param,
		    N_in_PE, N_disp, 
		    D_local_1D, sigma_local_1D, SS_local_1D,
		    N_in_PE_D, N_disp_D,
		    N_in_PE_sigma, N_disp_sigma,
		    N_in_PE_SS, N_disp_SS );
 
    int N = N_in_PE[rank]; //Num of particle in a PE
    int real_ip;

    Allocation( param, N, N_hook, Ze, Zeq,
		D, sigma, SS,
		SL_pair, SL_pos,
		r, s, L, w, Entanglement, counter_analyze,
		sg_sum, SS_sum, Z_sum, L_sum, d2_Z, d2_L, ave_s, dis_s );

    MY_MPI_scatter( param, D_all_1D, D_local_1D, D,
		    N_in_PE, N_disp, N_in_PE_D, N_disp_D );

    Set_InitValue_Pasta( param, N, argv, Ze, Zeq, sigma, SS, 
			 SL_pair, SL_pos, r, s, L, Entanglement );

    for( int ip=0; ip<N; ip++ ){
      real_ip = ip+N_disp[rank];
      Write_Data( param, Ze[ip], SL_pair[ip], SL_pos[ip][NOW], r[ip],
		  s[ip][NOW], L[ip][NOW], 0, real_ip );
    }
    MPI_Barrier( MCW );
    Ts2 = MPI_Wtime();
    for( int step=0; step<MAXSTEP; step++ ){
      debug_step = step;
      if( step >= Equilibration_step && step % IntervalStepOfData == 0 ){
	output_data = true;
	counter++;
      }
      if( rank==0 && step%100==0 ){ printf("step=%d \n",step); }
      //if( step%10==0 ){ printf(" rank=%d  step=%d \n",rank, step); }
      NOW =  step    % 2;
      NEW = (step+1) % 2;
      for( int ip=0; ip<N; ip++ ){
	Generating_Random_Numbers( param, w[ip] );
	Evaluate_Stress( param, step, NOW, N_hook[ip], Ze[ip], Zeq[ip], 
			 SL_pair[ip], sigma[ip], SS[ip], D[ip], SL_pos[ip], 
			 r[ip], s[ip], L[ip], w[ip] );
      }
      MPI_Barrier( MCW );
      for( int ip=0; ip<N; ip++ ){
	real_ip = ip+N_disp[rank];
	if( output_data ){
	  Write_Data( param, Ze[ip], SL_pair[ip], SL_pos[ip][NEW], r[ip],
		      s[ip][NEW], L[ip][NEW], counter, real_ip );
	  if( ip==N-1 ){
	    output_data = false;
	  }
	}
	if( step >= Equilibration_step && step%time_interval == 0 ){	
	  Write_Data_Step( param, N, Ze[ip], sigma[ip], SS[ip], L[ip][NEW], step, ip, real_ip ); 
	}
	if( step >= Analysis_start_step ){	
	  Analysis_Z_and_L( param, N, Ze[ip], sigma[ip], SS[ip], sg_sum[ip], SS_sum[ip],
			    L[ip][NEW], Z_sum[ip], L_sum[ip], d2_Z[ip], d2_L[ip], 
			    D[ip], s[ip][NEW], ave_s[ip], dis_s[ip], step, ip, real_ip, counter_analyze[ip] ); //20180112
	}
      }// ip
      MY_MPI_gather( param, rank, sigma, SS, D, sigma_all, D_all, N_in_PE, N_disp,
		     D_local_1D, D_all_1D, sigma_local_1D, sigma_all_1D, SS_local_1D, SS_all_1D,
		     N_in_PE_D, N_disp_D, N_in_PE_sigma, N_disp_sigma, N_in_PE_SS, N_disp_SS ); 
      if( rank==0 ){
	Substitution( param, step, sigma_all, SS_all,  D_all, 
		      sigma_all_1D, SS_all_1D, D_all_1D );
      }
    }//end of main loop
    
    for( int ip=0; ip<N; ip++ ){
      int real_ip = ip+N_disp[rank];
      Write_Data_end( param, N_hook[ip], Ze[ip], SL_pair[ip], SL_pos[ip][NEW], s[ip][NEW],
		      ave_s[ip], dis_s[ip], ip, real_ip, counter_analyze[ip]);
    }
    if( rank==0 ){
      Write_D_Data( param, D_all ); 
    }
    MPI_Barrier( MCW );
    Tf = MPI_Wtime();
    if( rank==0 ){
      cout<<"Time1 = "<<Tf-Ts1<<"  Time2 = "<<Tf-Ts2<<endl;
      cout<<"a_T = "<<aT<<endl;
      cout<<"b_T = "<<bT<<endl;
      printf("\n finished normally !\n");
    }
  }
  catch (MotherErr& a){ cout << a.Get() << endl ; exit (1); }
  catch (...){  cout << "unknown err!!" << endl ; exit (1); }
  MPI_Finalize();
  return(0);
}
//-----------------------------------------------------------------------
int Analysis_Z_and_L( IOParameterSet &param, 
		      int             N,
		      int*           &Ze, 
		      double*        &sigma,
		      double*        &SS,
		      double*        &sg_sum,
		      double*        &SS_sum,
		      double*        &L, 
		      double         &Z_sum,
		      double         &L_sum,
		      double         &d2_Z,
		      double         &d2_L,
		      double**       &D,
		      double**       &s,
		      double*        &ave_s,
		      double*        &dis_s,
		      const int      &step,
		      int             ip,
		      int             real_ip,
		      int            &counter_analyze )
{
  static const int XX=0, YY=1, ZZ=2, XY=3, YZ=4, ZX=5;
  static const int X=0, Y=1, Z=2;
  static const int N_polymer          = param.int_value("N_polymer");
  static const int MAXSTEP            = param.int_value("MAXSTEP");
  static const int Equilibration_step = param.int_value("Equilibration_step");
  static const double inv_Np = 1.0/N_polymer;
  static const int SPACE_DIM = 3;
  static const int MM = SPACE_DIM*(( SPACE_DIM-1 )/2 + 1 );
  static bool first_flag_analyze = true;

  if( first_flag_analyze == true ){
    for( int mm=0; mm<MM; mm++ ){
      SS_sum[mm] = 0.0;
      sg_sum[mm] = 0.0;//sg=sigma
    }
    Z_sum = 0.0;
    L_sum = 0.0;
    d2_Z  = 0.0;
    d2_L  = 0.0;
    counter_analyze = 0;
    if( ip==N-1 ){
      first_flag_analyze = false;
    }
  }

  counter_analyze++;

  SS_sum[XX] +=   SS[XX];  SS_sum[YY] +=   SS[YY];  SS_sum[ZZ] +=   SS[ZZ];
  SS_sum[XY] +=   SS[XY];  SS_sum[YZ] +=   SS[YZ];  SS_sum[ZX] +=   SS[ZX];

  sg_sum[XX] +=sigma[XX];  sg_sum[YY] +=sigma[YY];  sg_sum[ZZ] +=sigma[ZZ];  
  sg_sum[XY] +=sigma[XY];  sg_sum[YZ] +=sigma[YZ];  sg_sum[ZX] +=sigma[ZX]; 

  //-------------------

  double Z_av=0.0;
  double L_av=0.0;
  double sh_av = 0.0;
  double st_av = 0.0;
  for( int p=0; p<N_polymer; p++ ){ 
    Z_av  += Ze[p]; 
    L_av  +=  L[p];
    sh_av += s[p][0];
    st_av += s[p][1];
  }
  Z_av  *= inv_Np;
  L_av  *= inv_Np;
  sh_av *= inv_Np;
  st_av *= inv_Np;

  Z_sum    += Z_av;
  L_sum    += L_av;
  ave_s[0] += sh_av;
  ave_s[1] += st_av;

  for( int p=0; p<N_polymer; p++ ){ 
    double dZ  =  Ze[p]-Z_av;
    double dL  =  L[p]-L_av;
    double dsh =  s[p][0]-sh_av;
    double dst =  s[p][1]-st_av;
    
    d2_Z += dZ*dZ;
    d2_L += dL*dL;
    dis_s[0] += dsh*dsh;
    dis_s[1] += dst*dst;
  }

  //-------------------

  if( step == MAXSTEP-1 ){
    //int STEP = MAXSTEP-Equilibration_step;
    //double inv_STEP = 1.0/STEP;
    double inv_STEP = 1.0/counter_analyze;
    char filename[32];
    sprintf( filename, "Z_and_L_ip_%.4d.dat", real_ip );
    FILE *fp = fopen(filename,"w");
    fprintf(fp, 
	    "# (01)Z_av,\t(02)L_av,\t(03)variation(sigma^2_z),\t(04)variation(sigma^2_L),\t"
	    "(05)sg_xx,\t(06)sg_yy,\t(07)sg_zz,\t(08)sg_xy,\t(09)sg_yz,\t(10)sg_zx,\t"
	    "(11)SS_xx,\t(12)SS_yy,\t(13)SS_zz,\t(14)SS_xy,\t(15)SS_yz,\t(16)SS_zx,\t"
            "(17)Average_Num_of_Step"
	    "\n");
    fprintf(fp, 
	    "%e\t%e\t%e\t%e\t%e\t"
	    "%e\t%e\t%e\t%e\t%e\t"
	    "%e\t%e\t%e\t%e\t%e\t"
	    "%e\t%d\n",
	    Z_sum*inv_STEP,              //1
	    L_sum*inv_STEP,              //2
	    sqrt(d2_Z*inv_STEP*inv_Np),  //3
	    sqrt(d2_L*inv_STEP*inv_Np),  //4
	    sg_sum[XX]*inv_STEP,         //5
	    sg_sum[YY]*inv_STEP,         //6
	    sg_sum[ZZ]*inv_STEP,         //7
	    sg_sum[XY]*inv_STEP,         //8
	    sg_sum[YZ]*inv_STEP,         //9
	    sg_sum[ZX]*inv_STEP,         //10
	    SS_sum[XX]*inv_STEP,         //11
	    SS_sum[YY]*inv_STEP,         //12
	    SS_sum[ZZ]*inv_STEP,         //13
	    SS_sum[XY]*inv_STEP,         //14
	    SS_sum[YZ]*inv_STEP,         //15
	    SS_sum[ZX]*inv_STEP,         //16
	    counter_analyze              //17
	    );
    fprintf(fp, 
	    "# (01)Dxx,\t(02)Dxy,\t(03)Dxz,\t(04)Dyx,\t"
	    "(05)Dyy,\t(06)Dyz,\t(07)Dzx,\t(08)Dzy,\t(09)Dzz\n");
    fprintf(fp, 
	    "%e\t%e\t%e\t%e\t%e\t"
	    "%e\t%e\t%e\t%e\n",
	    D[X][X],              //1
	    D[X][Y],              //2
	    D[X][Z],              //3
	    D[Y][X],              //4
	    D[Y][Y],              //5
	    D[Y][Z],              //6
            D[Z][X],              //7
	    D[Z][Y],              //8	    
	    D[Z][Z]               //9
            );	  	
    fflush(fp);
    fclose(fp);
  }
  return(0);
}
//-----------------------------------------------------------------------
int Allocation_PE1( IOParameterSet &param,
		    double***      &D_all,
		    double**       &sigma_all,
		    double**       &SS_all,
		    double*        &D_all_1D,
		    double*        &sigma_all_1D,
		    double*        &SS_all_1D )
{
  static const int N_particle = param.int_value("N_particle");
  static const int N_polymer  = param.int_value("N_polymer");
  const int SPACE_DIM         = 3;
  int MM                      = SPACE_DIM*( (SPACE_DIM - 1)/2 + 1 );

  ALLOC_3D( D_all,       N_particle,  SPACE_DIM, SPACE_DIM );
  ALLOC_2D( sigma_all,   N_particle,  MM );
  ALLOC_2D( SS_all,      N_particle,  MM );
  ALLOC_1D( D_all_1D,    N_particle*SPACE_DIM*SPACE_DIM );
  ALLOC_1D( sigma_all_1D,  N_particle*MM );
  ALLOC_1D( SS_all_1D,     N_particle*MM );

  for( int ip=0; ip<N_particle; ip++ ){
    for( int dim1=0; dim1<SPACE_DIM; dim1++ ){
      for( int dim2=0; dim2<SPACE_DIM; dim2++ ){
	D_all[ip][dim1][dim2] = param.value("D",dim1*SPACE_DIM+dim2);
	D_all_1D[ip*SPACE_DIM*SPACE_DIM+dim1*SPACE_DIM+dim2] 
	  = D_all[ip][dim1][dim2];
      }
    }
    for( int mm=0; mm<MM; mm++ ){
      sigma_all[ip][mm] = 0.0;
      SS_all   [ip][mm] = 0.0;
    }
  }
  return(0);
}
//----------------------------------------------------------------------
int Allocation_MPI( IOParameterSet &param, 
		    int*           &N_in_PE,
		    int*           &N_disp,
		    double*        &D_local_1D,
		    double*        &sigma_local_1D,
		    double*        &SS_local_1D,
		    int*           &N_in_PE_D,
		    int*           &N_disp_D,
		    int*           &N_in_PE_sigma,
		    int*           &N_disp_sigma,
		    int*           &N_in_PE_SS,
		    int*           &N_disp_SS )
{
  static const int N_particle = param.int_value("N_particle");
  static const int N_polymer  = param.int_value("N_polymer");
  const int SPACE_DIM         = 3;
  int MM                      = SPACE_DIM*( (SPACE_DIM - 1)/2 + 1 );
  int size, rank;

  
  MPI_Comm_rank( MCW, &rank );
  MPI_Comm_size( MCW, &size );
  int ndiv = N_particle/size;
  int nmod = N_particle%size;
  ALLOC_1D_int( N_in_PE, size );
  ALLOC_1D_int( N_disp, size );
  ALLOC_1D_int( N_in_PE_D, size );
  ALLOC_1D_int( N_disp_D, size );
  ALLOC_1D_int( N_in_PE_sigma, size );
  ALLOC_1D_int( N_disp_sigma, size );
  ALLOC_1D_int( N_in_PE_SS, size );
  ALLOC_1D_int( N_disp_SS, size );

  for( int i=0; i<size; i++ ){
    if( i+1>nmod ){
      N_in_PE[i] = ndiv;
    }else if(i+1<=nmod){
      N_in_PE[i] = ndiv+1;
    }
    N_in_PE_D[i]     = N_in_PE[i]*SPACE_DIM*SPACE_DIM;
    N_in_PE_sigma[i] = N_in_PE[i]*MM;
    N_in_PE_SS[i]    = N_in_PE[i]*MM;
    if( rank==0 ){
      cout<<"rank= "<<i<<" N_in_PE= "<<N_in_PE[i]<<endl;
    }
  }
  int i1=0;
  for( int i=0; i<size; i++ ){
    N_disp[i]       = i1;
    N_disp_D[i]     = i1*SPACE_DIM*SPACE_DIM;
    N_disp_sigma[i] = i1*MM;
    N_disp_SS[i]    = i1*MM;
    if( rank==0 ){
      cout<<"rank= "<<i<<" N_disp= "<<N_disp[i]<<endl;
    }
    i1 += N_in_PE[i];
  }
  int N = N_in_PE[rank];

  ALLOC_1D( D_local_1D,     N*SPACE_DIM*SPACE_DIM );
  ALLOC_1D( sigma_local_1D, N*MM );
  ALLOC_1D( SS_local_1D,    N*MM );

  return(0);
}
//----------------------------------------------------------------------
int Allocation( IOParameterSet &param,
		int             N,
		int***         &N_hook,
		int**          &Ze,
		int**          &Zeq,
		double***      &D,
		double**       &sigma,
		double**       &SS,
		int****        &SL_pair,
		double*****    &SL_pos,
		double****     &r,
		double****     &s,
		double***      &L,
		double**       &w,
		int***         &Entanglement,
		int*           &counter_analyze,
		double**       &sg_sum,
		double**       &SS_sum,
		double*        &Z_sum,
		double*        &L_sum,
		double*        &d2_Z,
		double*        &d2_L,
		double**       &ave_s,
		double**       &dis_s )
{
  static const int N_polymer      = param.int_value("N_polymer");
  //static const int init_L_factor  = param.int_value("init_L_factor"); // Comment out by TS (20180409TS)
  static const int Zeq_data       = param.int_value("Zeq");
  static const int Zeq_short      = param.int_value("Zs");
  static const int Zeq_long       = param.int_value("Zl");
  static const int Zeq_arm_data   = param.int_value("Zeq_arm");
  static const int Melt_Flag      = param.int_value("Melt_Flag"); // 0: Linear, 1: Linear and Star
  static const double rate_of_long_polymer = param.value("rate_of_long_polymer");
  static const double rate_of_star_polymer = param.value("rate_of_star_polymer");
  static const int SPACE_DIM = 3;
  static const int MM_pasta  = SPACE_DIM*( (SPACE_DIM - 1)/2 + 1 );
  static const int N_long_polymer = int(N_polymer*rate_of_long_polymer);
  static const int N_star_polymer = int(N_polymer*rate_of_star_polymer);
  int L_factor;

  ALLOC_3D_int( N_hook, N, 2, N_polymer );
  ALLOC_2D_int( Ze,     N,    N_polymer );
  ALLOC_2D_int( Zeq,    N,    N_polymer );
  ALLOC_3D( D,          N,    SPACE_DIM, SPACE_DIM );
  ALLOC_2D( sigma,      N,    MM_pasta );
  ALLOC_2D( SS,         N,    MM_pasta );
  ALLOC_4D( s,          N,    2,  N_polymer,  2         );
  ALLOC_3D( L,          N,    2,  N_polymer             );
  ALLOC_2D( w,          N,    2*N_polymer               );
  ALLOC_2D( sg_sum,     N,    MM_pasta );
  ALLOC_2D( SS_sum,     N,    MM_pasta );
  ALLOC_1D( Z_sum,      N );
  ALLOC_1D( L_sum,      N );
  ALLOC_1D( d2_Z,       N );  
  ALLOC_1D( d2_L,       N );
  ALLOC_2D( ave_s,      N, 2 );
  ALLOC_2D( dis_s,      N, 2 );    
  ALLOC_1D_int( counter_analyze, N );

  for( int ip=0; ip<N; ip++ ){
    ave_s[ip][0]  = 0.0;
    ave_s[ip][1]  = 0.0;
    dis_s[ip][0]  = 0.0;
    dis_s[ip][1]  = 0.0;
    for( int p=0; p<N_polymer; p++ ){
      N_hook[ip][0][p] = 0;
      N_hook[ip][1][p] = 0;
    }
  }


  SL_pair = new int*** [N];
  SL_pos  = new double**** [N];
  r       = new double*** [N];
  Entanglement = new int** [N];
  
  for( int ip=0; ip<N; ip++ ){
    SL_pair[ip] = new int** [N_polymer];
    Entanglement[ip] = new int* [N_polymer];
    if( Melt_Flag == 0){
      for( int p=0; p<N_polymer; p++ ){
	// added by harada (20140425)
	// modified by TS (20180409TS)
	if( p< N_long_polymer ){
	  //L_factor = init_L_factor;
	  Zeq[ip][p] = Zeq_long;
	}else{
	  //L_factor = 1;
	  Zeq[ip][p] = Zeq_short;
	}
	//Zeq[ip][p] = Zeq_data*L_factor;      
	SL_pair[ip][p] = new int* [Zeq[ip][p]];
	Entanglement[ip][p] = new int [Zeq[ip][p]];
	for( int m=0; m<Zeq[ip][p]; m++ ){
	  SL_pair[ip][p][m] = new int [2];
	}
      }
    }else if( Melt_Flag == 1 ){ // For Linear and Star Melt(20160822)
      for( int p=0; p<N_polymer; p++ ){
	if( p< N_star_polymer ){	 
          Zeq[ip][p] = Zeq_arm_data;      	  
	}else{
	  Zeq[ip][p] = Zeq_data;
	}
	SL_pair[ip][p] = new int* [Zeq[ip][p]];
	Entanglement[ip][p] = new int [Zeq[ip][p]];
	for( int m=0; m<Zeq[ip][p]; m++ ){
	  SL_pair[ip][p][m] = new int [2];
	}
      }
    }
    SL_pos[ip] = new double*** [2];
    SL_pos[ip][0] = new double** [N_polymer];
    SL_pos[ip][1] = new double** [N_polymer];
    for( int t=0; t<2; t++ ){
      for( int p=0; p<N_polymer; p++ ){
	SL_pos[ip][t][p] = new double* [Zeq[ip][p]];
	for( int m=0; m<Zeq[ip][p]; m++ ){
	  SL_pos[ip][t][p][m] = new double [3];
	}
      }
    }

    r[ip] = new double** [N_polymer];
    for( int p=0; p<N_polymer; p++ ){
      r[ip][p] = new double* [Zeq[ip][p]];
      for( int m=0; m<Zeq[ip][p]-1; m++ ){
	r[ip][p][m] = new double [3];
      }
    }
  }// For loop of ip
  
  for( int ip=0; ip<N; ip++ ){
    for( int p=0; p<N_polymer; p++ ){
      for( int k=0; k<Zeq[ip][p]-1; k++ ){
	for( int m=0; m<2; m++ ){	
	  SL_pair[ip][p][k][m] = 0;
	}
      }
    }
  }
  return(0);
}


//-----------------------------------------------------------------------
int nran( const int& N, double* &w )
{
  static double Two_PI = 2.0*M_PI;
  for( int i=0; i<N; i+=2 ){
    double r0 = drand48();
    double r1 = drand48();
    double sqrt_2log_r0 =  sqrt( fabs( 2.0 * log(r0) ) );
    double Two_PI_r1    =  Two_PI*r1;
    w[i  ] = sqrt_2log_r0 * cos( Two_PI_r1 );
    w[i+1] = sqrt_2log_r0 * sin( Two_PI_r1 );
  }
  return(0);
}
//------------------------------------------------------------------------
int Generating_Random_Numbers( IOParameterSet &param,
			       double*        &w    )
{
  static int N_polymer = param.int_value("N_polymer");
  static double Two_PI = 2.0*M_PI;
  static int N_total   = 2*N_polymer;
  
  for( int p=0; p<N_total; p+=2 ){

    double w1 = genrand_real3();
    double w2 = genrand_real3();
    double sqrt_2log_w1 =  sqrt( fabs( 2.0 * log(w1) ) );
    double Two_PI_w2  =  Two_PI*w2;
    w[p  ] = sqrt_2log_w1 * cos( Two_PI_w2 );
    w[p+1] = sqrt_2log_w1 * sin( Two_PI_w2 );
  }

  double sum = 0.0;
  double sum2= 0.0;

  for( int p=0; p<N_total; p++ ){ sum += w[p]; }
  double ave = sum / N_total;

  for( int p=0; p<N_total; p++ ){ sum2 += (w[p]-ave)*(w[p]-ave); }
  double factor = sqrt( N_total / sum2 );

  for( int p=0; p<N_total; p++ ){ w[p] = (w[p]-ave)*factor; }

  return(0);
}
//------------------------------------------------------------------------
int New_entanglement( IOParameterSet &param,
		      const int      &NEW,
		      const int      &p,
		      const int      &end,
		      int*           &Ze,
		      int*           &Zeq,
		      int***         &SL_pair,
		      double****     &SL_pos,
		      double***      &r,
		      double***      &s,
		      double**       &L )
{
  static int N_polymer  = param.int_value("N_polymer");
  static const double Lmax_factor   = param.value("Lmax_factor");   
  double inv_Zeq = 1.0/Zeq[p];
  static const int X=0, Y=1, Z=2;
  static const int head=0, tail=1;
  static const int PC=0, SL=1;
  static double* nr = new double[4]; // Gauss normal random number
  static double MAGIC_FACTOR = 1.4;

  int NOW = (NEW+1)%2;
  int q=-1; // = SL_pair[p][end_SL][PC];
  int qk;   // = SL_pair[p][end_SL][SL];
  int end_SL = end*( Ze[p] );     
  int pk     = end_SL;

  //   Polymer chain ( Z : number of slip-link )
  //
  //             SL(0)    SL(1)       SL(2)    SL(Z-2)   SL(Z-1)
  //       x------o---------o----------o-------||-o---------o-------x
  //        <---->|<------->|<-------->|          |<------->|<------>
  //           s0 |   r(0)  |    r(1)  |          |  r(Z-2) |   s1   
  //              |	    |	       |	  |	    |	     
  //         head |  tube_0 |  tube_1  |	  | tube_z-2|	tail     
  //
  //      # of slip-link = Z
  //      # of tubes     = Z-1  tube_id=[0:Z-2]
  //
  // entangled polymer

  //------------------------------------------------------------
  //[ Select the opponent polymer (q) of new entanglement of p ]
  //------------------------------------------------------------
  // (a) selection probablity is proportial to its length //
  // Comment by TS (20180411)
  // double L_total = 0.0;
  // for( int pp=0; pp<N_polymer; pp++ ){
  //   L_total += L[NEW][pp];
  // }
  // L_total -= L[NEW][p];

  // double ra = drand48();
  // double  x = 0.0;
  // for( int pp=0; pp<N_polymer; pp++ ){
  //   if( pp != p ){
  //     x += L[NEW][pp];
  //     if( ra < x/L_total ){ q = pp; break; }
  //   }
  // }

  // modified by TS because of the detailed balance consition (20180418TS)
  while( 1 ){
    int rand_num = genrand_int31(); 
    q = rand_num % N_polymer;
    if( q != p ){ break; }
  }
  
  if( q == -1 ){ 
    printf("Selction does not work well, then exit(1)\n"); 
    exit(1);
  }
  //---------------------------------------------------------
  // (b) selection probablity is uniform
  // an opponent polymer chain is selsected RANDOMLY.
  // random number should be [0,1), i.e. it should not include "1."
  // There is possibility that Ze[q] = 0 or Ze[q] = 1
  // while(1){
  //   if( q != p && Ze[q] >=2 ){ break; }
  //   q = (int)( N_polymer*drand48() ); // [0,1)
  //   if( q != p ){ break; }
  // }
  //---------------------------------------------------------


  //==================================================================
  // << Opponent >>
  //debug
  if(        Ze[q] == 0 ){

    qk = 0; // Because there is no entanglement point now.
    L[NEW][q] = 0.0; //added by harada (20140901)
    // a new entanglement point
    //printf("New Entangle 1 Ze[q]=0\n");

    double** pTMP = new double* [Ze[q]+1];
    pTMP[Ze[q]] = new double [3];
    SL_pos[NEW][q] = pTMP;
    
    pTMP = new double* [Ze[q]+1];
    pTMP[Ze[q]] = new double [3];
    SL_pos[NOW][q] = pTMP;

    int** pITMP = new int* [Ze[q]+1]; 
    pITMP[Ze[q]] = new int [2];
    SL_pair[q] = pITMP;

    // << Renewal of Position >>
    SL_pos[NEW][q][Ze[q]][X] = 0;
    SL_pos[NEW][q][Ze[q]][Y] = 0; 
    SL_pos[NEW][q][Ze[q]][Z] = 0;

  }else{  // Ze[q] >= 1

    // a new entanglement point
    //printf("New Entangle 1\n");

    double** pTMP = new double* [Ze[q]+1];
    for( int m=0; m<Ze[q]; m++ ){ pTMP[m] = SL_pos[NEW][q][m]; }
    pTMP[Ze[q]] = new double [3];
    delete [] SL_pos[NEW][q];
    SL_pos[NEW][q] = pTMP;

    pTMP = new double* [Ze[q]+1];
    for( int m=0; m<Ze[q]; m++ ){ pTMP[m] = SL_pos[NOW][q][m]; }
    pTMP[Ze[q]] = new double [3];
    delete [] SL_pos[NOW][q];
    SL_pos[NOW][q] = pTMP;

    pTMP = new double* [Ze[q]];
    for( int m=0; m<Ze[q]-1; m++ ){ pTMP[m] = r[q][m]; }
    pTMP[Ze[q]-1] = new double [3];
    if( Ze[q] != 1 ){ delete [] r[q]; }
    r[q] = pTMP;

    int** pITMP = new int* [Ze[q]+1]; 
    for( int m=0; m<Ze[q]; m++ ){ pITMP[m] = SL_pair[q][m]; }
    pITMP[Ze[q]] = new int [2];
    delete [] SL_pair[q];
    SL_pair[q] = pITMP;

    if(  Ze[q] == 1 ){

      qk = 1; // without loss of generality we can select qk=1

      // << Renewal of Position >>
      double rx, ry, rz, rho;
      while( 1 ){
	rx = 2*( drand48()-0.5 );
	ry = 2*( drand48()-0.5 );
	rz = 2*( drand48()-0.5 );
	rho= sqrt( rx*rx + ry*ry + rz*rz );
	if( rho < 1.0 ){ break; }
      }
      double dx = rx/rho;
      double dy = ry/rho;
      double dz = rz/rho;

      SL_pos[NEW][q][1][X] = SL_pos[NEW][q][0][X] + dx;
      SL_pos[NEW][q][1][Y] = SL_pos[NEW][q][0][Y] + dy; 
      SL_pos[NEW][q][1][Z] = SL_pos[NEW][q][0][Z] + dz;
      L[NEW][q]            = 1.0;

      //Added by harada
      r[q][0][X]=dx;
      r[q][0][Y]=dy;
      r[q][0][Z]=dz;
      // printf("dx=%e dy=%e dz=%e length=%e\n",
      // dx,dy,dz,sqrt( dx*dx + dy*dy + dz*dz ));
    
    }else{  // Ze[q] >= 2 //
      
      //-----------------------------------------------------------
      //-----------------------------------------------------------
      //[ Select the opponent strand (qk) of new entanglement of p ]//
      //-----------------------------------------------------------
      // (a) selection probablity is proportial to its length //

      double length_t = 0.0;
      double* len = new double [Ze[q]-1];
      for( int k=0; k <= Ze[q]-2; k++ ){
      	double dx = SL_pos[NEW][q][k+1][X] - SL_pos[NEW][q][k][X];
      	double dy = SL_pos[NEW][q][k+1][Y] - SL_pos[NEW][q][k][Y];
      	double dz = SL_pos[NEW][q][k+1][Z] - SL_pos[NEW][q][k][Z];
      	len[k]    = sqrt( dx*dx + dy*dy + dz*dz );
      	length_t += len[k];
      }
      double ra = drand48();
      double  x = 0.0;
      for( int k=0; k<=Ze[q]-2; k++ ){
      	x += len[k];
      	if( ra < x/length_t ){ qk = k + 1; break; }
      }
      delete[] len;
      
      //-----------------------------------------------------------
      // (b) the selection probability is uniform.
      // qk = (int)( drand48()*(Ze[q]-1) )+1; // qk in [1,Ze[q]-1]
      //-----------------------------------------------------------
      //-----------------------------------------------------------

      for( int k=Ze[q]; k >= qk+1; k-- ){

	// position shift
	SL_pos[NEW][q][k][X] = SL_pos[NEW][q][k-1][X];
	SL_pos[NEW][q][k][Y] = SL_pos[NEW][q][k-1][Y];
	SL_pos[NEW][q][k][Z] = SL_pos[NEW][q][k-1][Z];

	// chain id PC=0, slip-link id SL = 1
	SL_pair[q][k][PC] = SL_pair[q][k-1][PC];
	SL_pair[q][k][SL] = SL_pair[q][k-1][SL];

	// opponent chain id to shifted slip-link = SL_pair[q][k][PC]    
	// opponent slip-link id to ...           = SL_pair[q][k][SL];

	int id_q = SL_pair[q][k][PC];
	int  k_q = SL_pair[q][k][SL];
	SL_pair[ id_q ][ k_q ][ SL ] ++;
      }

      // << Insertion : Position >>
      double rx, ry, rz, rho;
      //modified by harada(20140710)
      double L_max      = Zeq[q]*Lmax_factor;
      double L_min      = 0.1; // From original code (20180412TS)
      double L_max_090  = L_max*0.90;
      double L_max_099  = L_max*0.99;
      double factor     = 1.0;
      double L_new = L[NEW][q];
      
      if( L_new>L_max_099 ){
	double r1, r2;
	r1 = drand48();
	r2 = 1.0 - r1;

	double xm =  SL_pos[NEW][q][qk-1][X]*r1+SL_pos[NEW][q][qk+1][X]*r2;
	double ym =  SL_pos[NEW][q][qk-1][Y]*r1+SL_pos[NEW][q][qk+1][Y]*r2;
	double zm =  SL_pos[NEW][q][qk-1][Z]*r1+SL_pos[NEW][q][qk+1][Z]*r2;
    
	SL_pos[NEW][q][qk][X] = xm;
	SL_pos[NEW][q][qk][Y] = ym;
	SL_pos[NEW][q][qk][Z] = zm;
	
      }else{

	// Old (comment out by TS)
	// while( 1 ){
	//   rx = 2*( drand48()-0.5 );
	//   ry = 2*( drand48()-0.5 );
	//   rz = 2*( drand48()-0.5 );
	//   rho= sqrt( rx*rx + ry*ry + rz*rz );
	//   if( rho < 1.0 ){ break; }
	// }
	// if( L_new>L_max_090 ){
	//   factor = (1.0-L[NEW][q]/L_max);
	// }
	
	// double dx = rx/rho*factor;
	// double dy = ry/rho*factor;
	// double dz = rz/rho*factor;

	double dx = 0.0;
	double dy = 0.0;
	double dz = 0.0;
	double xm = ( SL_pos[NEW][q][qk-1][X]+SL_pos[NEW][q][qk+1][X] )*0.5;
	double ym = ( SL_pos[NEW][q][qk-1][Y]+SL_pos[NEW][q][qk+1][Y] )*0.5;
	double zm = ( SL_pos[NEW][q][qk-1][Z]+SL_pos[NEW][q][qk+1][Z] )*0.5;

        // 20170531(TS) -> Consider Gauss chain statistics between i- & (i+1)-th chain!
	//=================================================================
	// This if section is needed. If L is negative (or positive but too small), dr becomes too large. 
	if( L[NEW][q] > L_min ){
	  
	  rx = ( SL_pos[NEW][q][qk-1][X]-SL_pos[NEW][q][qk+1][X] );
	  ry = ( SL_pos[NEW][q][qk-1][Y]-SL_pos[NEW][q][qk+1][Y] );
	  rz = ( SL_pos[NEW][q][qk-1][Z]-SL_pos[NEW][q][qk+1][Z] );
	  double len= sqrt( rx*rx + ry*ry + rz*rz );
	  double R0 = MAGIC_FACTOR * sqrt( len * Zeq[q] / ( 12.0*L[NEW][q] ) );
	  nran( 4, nr );
	  dx = nr[X]*R0;
	  dy = nr[Y]*R0;
	  dz = nr[Z]*R0;
	  
	}else{
	  
	  while( 1 ){
	    rx = 2*( drand48()-0.5 );
	    ry = 2*( drand48()-0.5 );
	    rz = 2*( drand48()-0.5 );
	    rho= sqrt( rx*rx + ry*ry + rz*rz );
	    if( rho < 1.0 ){ break; }
	  }
	  if( L_new>L_max_090 ){
	    factor = (1.0-L[NEW][q]/L_max);
	  }
	  dx = rx/rho*factor;
	  dy = ry/rho*factor;
	  dz = rz/rho*factor;
	  
	}
	//=================================================================
	
	SL_pos[NEW][q][qk][X] = xm + dx;
	SL_pos[NEW][q][qk][Y] = ym + dy;
	SL_pos[NEW][q][qk][Z] = zm + dz;
	
      }// end of if loop
      
      for( int k=0; k<=Ze[q]-1; k++ ){

	r[q][k][X]=SL_pos[NEW][q][k+1][X]-SL_pos[NEW][q][k][X];
	r[q][k][Y]=SL_pos[NEW][q][k+1][Y]-SL_pos[NEW][q][k][Y];
	r[q][k][Z]=SL_pos[NEW][q][k+1][Z]-SL_pos[NEW][q][k][Z];
      
      }

      double length = 0.0;
      for( int k=0; k<=Ze[q]-1; k++ ){
	double dx = r[q][k][X];
	double dy = r[q][k][Y];
	double dz = r[q][k][Z];
	length += sqrt( dx*dx + dy*dy + dz*dz );	
      }
      L[NEW][q] = length;
      
    }
  }
  //------------------------------------------------------------------
  //------------------------------------------------------------------
  // 
  //------------------------------------------------------------------
  //------------------------------------------------------------------

  if(        Ze[p] == 0 ){

    pk = 0; // Because there is no entanglement point now.

    // a new entanglement point
    // printf("New Entangle 2 Ze[p]=0\n");

    double** pTMP = new double* [Ze[p]+1];
    pTMP[Ze[p]] = new double [3];
    SL_pos[NEW][p] = pTMP;

    pTMP = new double* [Ze[p]+1];
    pTMP[Ze[p]] = new double [3];
    SL_pos[NOW][p] = pTMP;

    int** pITMP = new int* [Ze[p]+1]; 
    pITMP[Ze[p]] = new int [2];
    SL_pair[p] = pITMP;

    // << Renewal of Position >>
    SL_pos[NEW][p][0][X] = 0;
    SL_pos[NEW][p][0][Y] = 0; 
    SL_pos[NEW][p][0][Z] = 0;
    L[NEW][p]            = 0;

  }else if( Ze[p] >= 1 ){ 
    
    // a new entanglement point

    // printf("New Entangle 2\n");
 
    double** pTMP = new double* [Ze[p]+1];
    for( int m=0; m<Ze[p]; m++ ){ pTMP[m] = SL_pos[NEW][p][m]; }
    pTMP[Ze[p]] = new double [3];
    delete [] SL_pos[NEW][p];
    SL_pos[NEW][p] = pTMP;

    pTMP = new double* [Ze[p]+1];
    for( int m=0; m<Ze[p]; m++ ){ pTMP[m] = SL_pos[NOW][p][m]; }
    pTMP[Ze[p]] = new double [3];
    delete [] SL_pos[NOW][p];
    SL_pos[NOW][p] = pTMP;

    pTMP = new double* [Ze[p]];
    for( int m=0; m<Ze[p]-1; m++ ){ pTMP[m] = r[p][m]; }
    pTMP[Ze[p]-1] = new double [3];
    if( Ze[p] != 1 ){ delete [] r[p]; }
    r[p] = pTMP;

    int** pITMP = new int* [Ze[p]+1]; 
    for( int m=0; m<Ze[p]; m++ ){ pITMP[m] = SL_pair[p][m]; }
    pITMP[Ze[p]] = new int [2];
    delete [] SL_pair[p];
    SL_pair[p] = pITMP;

    // << Renewal of Position >>    
    // When Ze[p] =1
    // end=0 (pk=0) -> pk1=1
    // end=1 (pk=1) -> pk1=0

    // pk = end_SL; // end_SL = end*( Ze[p] );     

    for( int k=Ze[p]; k >= pk+1; k-- ){

      // position shift
      SL_pos[NEW][p][k][X] = SL_pos[NEW][p][k-1][X];
      SL_pos[NEW][p][k][Y] = SL_pos[NEW][p][k-1][Y];
      SL_pos[NEW][p][k][Z] = SL_pos[NEW][p][k-1][Z];

      // chain id PC=0, slip-link id SL = 1
      SL_pair[p][k][PC] = SL_pair[p][k-1][PC];
      SL_pair[p][k][SL] = SL_pair[p][k-1][SL];

      int id_p = SL_pair[p][k][PC];    
      int  k_p = SL_pair[p][k][SL];
      //printf("pk=%d k=%d id_p=%d k_p=%d SL=%d\n",pk, k, id_p, k_p, SL );
      SL_pair[ id_p ][ k_p ][ SL ] ++;

    }

    // << Insertion : Position >>
    double rx, ry, rz, rho;
    while( 1 ){
      rx = 2*( drand48()-0.5 );
      ry = 2*( drand48()-0.5 );
      rz = 2*( drand48()-0.5 );
      rho= sqrt( rx*rx + ry*ry + rz*rz );
      if( rho < 1.0 ){ break; }
    }
    double dx = rx/rho;
    double dy = ry/rho;
    double dz = rz/rho;
    
    int pk1 = (Ze[p]-2)*end+1;
    SL_pos[NEW][p][pk][X] = SL_pos[NEW][p][pk1][X]+dx;
    SL_pos[NEW][p][pk][Y] = SL_pos[NEW][p][pk1][Y]+dy;
    SL_pos[NEW][p][pk][Z] = SL_pos[NEW][p][pk1][Z]+dz;

    //-----------------------

    for( int k=0; k<=Ze[p]-1; k++ ){
      r[p][k][X]=SL_pos[NEW][p][k+1][X]-SL_pos[NEW][p][k][X];
      r[p][k][Y]=SL_pos[NEW][p][k+1][Y]-SL_pos[NEW][p][k][Y];
      r[p][k][Z]=SL_pos[NEW][p][k+1][Z]-SL_pos[NEW][p][k][Z];
    }

    double length = 0.0;
    for( int k=0; k<=Ze[p]-1; k++ ){
      double dx = r[p][k][X];
      double dy = r[p][k][Y];
      double dz = r[p][k][Z];
      length += sqrt( dx*dx + dy*dy + dz*dz );
    }
    L[NEW][p] = length;
    s[NEW][p][end] -= 1;
  }else{
    printf("Ze[%d] = %d: Impossible !",p, Ze[p] );
    exit(1);
  }
  
  // << Pairing Info >>
  SL_pair    [q][qk    ][PC] = p;
  SL_pair    [q][qk    ][SL] = end_SL;
  SL_pair    [p][end_SL][PC] = q;
  SL_pair    [p][end_SL][SL] = qk;
  L[NEW][q] += ( s[NEW][q][head] + s[NEW][q][tail] );
  L[NEW][p] += ( s[NEW][p][head] + s[NEW][p][tail] );
  Ze[p]++;
  Ze[q]++;
  
  return(0);
}
//------------------------------------------------------------------------
int Release_of_entangled_polymer( IOParameterSet &param,
				  const int      &NEW,
		      		  const int      &p,
		      		  const int      &end,
		      		  int*           &Ze,
		      		  int***         &SL_pair,
		      		  double****     &SL_pos,
		      		  double***      &r,
		      		  double***      &s,
				  double**       &L )
{
  static int N_polymer = param.int_value("N_polymer");
  static const int X=0, Y=1, Z=2; 
  static const int head=0, tail=1;
  static const int PC=0, SL=1;
  double coeff;
  double theta, phi;
  double TWO_PI = 2.0 * M_PI;
  int NOW = (NEW+1)%2;

  //   Polymer chain ( Z : number of slip-link )
  //
  //          SL(0)   SL(1)    SL(2)          SL(Z-1)
  //    head-----o-------o--------o------||-------o------tail
  //        <---->------->-------->          -----><---->
  //          s0    r(0)    r(1)            r(Z-2)   s1
  //
  //
  // entangled polymer
  //            
  //    Pair    (PC_id,SL_id)=(p, end)  <---(entangled)---> (q,qk)
  //
  //             end = 0, or 1  =>  end_SL = 0 or Z[p] - 1 
  //
  //             end_SL = end * (Ze[p]-1)
  //
  // modified by TT  

  //-------------------------------------------------------------------------
  // Dealing with the opponent entanglement point
  //-------------------------------------------------------------------------
  // added by TT
  // q  -> polymer-chain-id of the opponent entangled polymer chain
  // qk -> slip-link     id of the opponent entangled polymer chain
  //  
  //   (p,pk) <-----> (q,qk)
  //   
  //if( Ze[p] < 1 ){ printf("Ze[p]=%d\n",Ze[p]); }

  int pk = end*( Ze[p] - 1 );     
  int q  = SL_pair[p][pk][PC];
  int qk = SL_pair[p][pk][SL];

  //printf("p=%d pk=%d q=%d qk=%d\n", p, pk, q, qk );

  int end_qSL___id = Ze[q]-1;
  int end_qTube_id = Ze[q]-2;

  if(       Ze[q] == 0 ){
    printf("Something strange !");
    exit(1);

  }else if( Ze[q] == 1 ){

    // There is nothing to do
    delete [] SL_pos[NEW][q][Ze[q]-1];
    delete [] SL_pos[NOW][q][Ze[q]-1];
    delete [] SL_pair    [q][Ze[q]-1];

  }else{  //Ze[q] >= 2 

    if(       qk == end_qSL___id ){ // When the opponent is the tail

      // s_tail
      double dx = r[q][ end_qTube_id ][X];
      double dy = r[q][ end_qTube_id ][Y];
      double dz = r[q][ end_qTube_id ][Z];
      s[NEW][q][tail] += sqrt( dx*dx + dy*dy + dz*dz );

    }else{

      if( qk == 0 ){              // When the opponet is the head
	// s_head 
	int tube_id = 0;
	double dx = r[q][tube_id][X];
	double dy = r[q][tube_id][Y];
	double dz = r[q][tube_id][Z];
	s[NEW][q][head] += sqrt( dx*dx + dy*dy + dz*dz );
      } 

      // When 1=< qk <= Ze-2      
      for( int k=qk; k < end_qSL___id; k++ ){

	SL_pos[NEW][q][k][X]   = SL_pos[NEW][q][k+1][X];
	SL_pos[NEW][q][k][Y]   = SL_pos[NEW][q][k+1][Y];
	SL_pos[NEW][q][k][Z]   = SL_pos[NEW][q][k+1][Z];
	SL_pair[q][k][PC]      = SL_pair[q][k+1][PC];
	SL_pair[q][k][SL]      = SL_pair[q][k+1][SL];

	// renewal of opponent side information
	int q_pc_id = SL_pair[q][k][PC];
	int q_sl_id = SL_pair[q][k][SL];
	//printf("p=%d q_pc_id=%d q_sl_id=%d \n",p,q_pc_id,q_sl_id);
	SL_pair[ q_pc_id ][ q_sl_id ][SL]--;

      }
      
      for( int k=0; k<= end_qTube_id-1; k++ ){ 
	r[q][k][X]=SL_pos[NEW][q][k+1][X]-SL_pos[NEW][q][k][X];
	r[q][k][Y]=SL_pos[NEW][q][k+1][Y]-SL_pos[NEW][q][k][Y];
	r[q][k][Z]=SL_pos[NEW][q][k+1][Z]-SL_pos[NEW][q][k][Z];
      }
    }

    //printf("Release 1\n");
    delete [] SL_pos[NEW][q][Ze[q]-1];
    delete [] SL_pos[NOW][q][Ze[q]-1];
    delete [] r          [q][Ze[q]-2];
    delete [] SL_pair    [q][Ze[q]-1];

    double** pTMP = new double* [Ze[q]-1];
    for( int m=0; m<=Ze[q]-2; m++ ){ pTMP[m] = SL_pos[NEW][q][m]; }
    delete [] SL_pos[NEW][q];
    SL_pos[NEW][q] = pTMP;

    pTMP = new double* [Ze[q]-1];
    for( int m=0; m<=Ze[q]-2; m++ ){ pTMP[m] = SL_pos[NOW][q][m]; }
    delete [] SL_pos[NOW][q];
    SL_pos[NOW][q] = pTMP;

    if( Ze[q] != 2 ){ pTMP = new double* [Ze[q]-2]; }
    for( int m=0; m<=Ze[q]-3; m++ ){ pTMP[m] = r[q][m]; }
    delete [] r[q];
    if( Ze[q] != 2 ){ r[q] = pTMP; }
  
    int** pITMP = new int* [Ze[q]-1]; 
    for( int m=0; m<=Ze[q]-2; m++ ){ pITMP[m] = SL_pair[q][m]; }
    delete [] SL_pair[q];
    SL_pair[q] = pITMP;
  }

  Ze[q]--;

  //--------------------------------------------------------------
  // Re-evaluate the total length of the opponent polymer chain
  double length=0.0;
  end_qTube_id = Ze[q]-2;     
  for( int tube_id=0; tube_id<=end_qTube_id; tube_id++ ){
    double dx = r[q][tube_id][X];
    double dy = r[q][tube_id][Y];
    double dz = r[q][tube_id][Z];
    length += sqrt( dx*dx + dy*dy + dz*dz );
  }
  L[NEW][q] = length + ( s[NEW][q][head] + s[NEW][q][tail] );
  //--------------------------------------------------------------
  //--------------------------------------------------------------
  //--------------------------------------------------------------
  //--------------------------------------------------------------
  //--------------------------------------------------------------
  // Re-newal of the entangled polymer

  int end_pSL___id = Ze[p]-1;   
  int end_pTube_id = Ze[p]-2;

  if(       Ze[p] == 0 ){
    printf("Something strange !");
    exit(1);
  }else if( Ze[p] == 1 ){

    // should be checked because the following three lines are different 
    // from the case in Ze[q] == 1
    double hL = 0.5*L[NEW][p]; 
    s[NEW][p][head] = hL;
    s[NEW][p][tail] = hL;
    //------------------------------------------------------------------

    delete [] SL_pos[NEW][p][Ze[p]-1];
    delete [] SL_pos[NOW][p][Ze[p]-1];
    delete [] SL_pair    [p][Ze[p]-1];

  }else{  //Ze[p] >= 2 

    if(   pk == end_pSL___id ){ // When the opponent is the tail

      // s_tail
      double dx = r[p][ end_pTube_id ][X];
      double dy = r[p][ end_pTube_id ][Y];
      double dz = r[p][ end_pTube_id ][Z];
      s[NEW][p][tail] += sqrt( dx*dx + dy*dy + dz*dz );

    }else{ // pk == 0           // When the opponet is the head

      // s_head 
      int tube_id = 0;
      double dx = r[p][tube_id][X];
      double dy = r[p][tube_id][Y];
      double dz = r[p][tube_id][Z];
      s[NEW][p][head] += sqrt( dx*dx + dy*dy + dz*dz );

      for( int k=0; k < end_pSL___id; k++ ){

	SL_pos[NEW][p][k][X]   = SL_pos[NEW][p][k+1][X];
	SL_pos[NEW][p][k][Y]   = SL_pos[NEW][p][k+1][Y];
	SL_pos[NEW][p][k][Z]   = SL_pos[NEW][p][k+1][Z];
	SL_pair    [p][k][PC]  = SL_pair    [p][k+1][PC];
	SL_pair    [p][k][SL]  = SL_pair    [p][k+1][SL];

	// renewal of opponent side information
	int p_pc_id = SL_pair[p][k][PC];
	int p_sl_id = SL_pair[p][k][SL];

	//printf("p_pc_id=%d p_sl_id=%d \n",p_pc_id,p_sl_id);
	SL_pair[ p_pc_id ][ p_sl_id ][SL]--;

      }
      
      for( int k=0; k<= end_pTube_id-1; k++ ){ 
	r[p][k][X] = SL_pos[NEW][p][k+1][X] - SL_pos[NEW][p][k][X];
	r[p][k][Y] = SL_pos[NEW][p][k+1][Y] - SL_pos[NEW][p][k][Y];
	r[p][k][Z] = SL_pos[NEW][p][k+1][Z] - SL_pos[NEW][p][k][Z];
      }
    }

    //printf("Release 2\n");
    delete [] SL_pos[NEW][p][Ze[p]-1];
    delete [] SL_pos[NOW][p][Ze[p]-1];
    delete [] r          [p][Ze[p]-2];
    delete [] SL_pair    [p][Ze[p]-1];

    double** pTMP = new double* [Ze[p]-1];
    for( int m=0; m<=Ze[p]-2; m++ ){ pTMP[m] = SL_pos[NEW][p][m]; }
    delete [] SL_pos[NEW][p];
    SL_pos[NEW][p] = pTMP;

    pTMP = new double* [Ze[p]-1];
    for( int m=0; m<=Ze[p]-2; m++ ){ pTMP[m] = SL_pos[NOW][p][m]; }
    delete [] SL_pos[NOW][p];
    SL_pos[NOW][p] = pTMP;

    if( Ze[p] != 2 ){ pTMP = new double* [Ze[p]-2]; }
    for( int m=0; m<=Ze[p]-3; m++ ){ pTMP[m] = r[p][m]; }
    delete [] r[p];
    if( Ze[p] != 2 ){ r[p] = pTMP; }
  
    int** pITMP = new int* [Ze[p]-1]; 
    for( int m=0; m<=Ze[p]-2; m++ ){ pITMP[m] = SL_pair[p][m]; }
    delete [] SL_pair[p];
    SL_pair[p] = pITMP;

  }

  Ze[p]--;

  //--------------------------------------------------------------
  // Re-evaluate the total length of the opponent polymer chain
  length=0.0;
  end_pTube_id = Ze[p]-2;     
  for( int tube_id=0; tube_id<=end_pTube_id; tube_id++ ){
    double dx = r[p][tube_id][X];
    double dy = r[p][tube_id][Y];
    double dz = r[p][tube_id][Z];
    length += sqrt( dx*dx + dy*dy + dz*dz );
  }
  L[NEW][p] = length + ( s[NEW][p][head] + s[NEW][p][tail] );
  //--------------------------------------------------------------
  return(0);
}
//-------------------------------------------------------------------------
int Evaluate_Stress( IOParameterSet &param, 
		     const int      &step,
		     const int      &NOW,
		     int**          &N_hook,
		     int*           &Ze,
		     int*           &Zeq,
		     int***         &SL_pair,
		     double*        &sigma, 
		     double*        &SS,
		     double**       &D,
		     double****     &SL_pos,
		     double***      &r,
		     double***      &s,
		     double**       &L,
		     double*        &w )
{
  static const int SPACE_DIM        = 3;
  static const int Melt_Flag        = param.int_value("Melt_Flag"); // 0: Linear, 1: Linear and Star
  static const int Micro_or_Macro   = param.int_value("Micro_or_Macro"); // 0: tau_e, 1: tau_macro
  static const double dt            = param.value("dt");
  static const double Lmax_factor   = param.value("Lmax_factor");   
  static const int    N_polymer     = param.int_value("N_polymer");
  static const int    Zeq_data      = param.int_value("Zeq");
  static const double De            = param.value("De");
  static const double T_ref         = param.value("T_ref");
  static const double T             = param.value("T");
  static const double delta_T       = T - T_ref; //Caution! unit should be K!
  static const double C1            = param.value("C1");
  static const double C2            = param.value("C2");
  static const double aT            = exp(-2.302585093*C1*delta_T/(C2+delta_T));
  static const double inv_aT        = 1.0 / aT;
  static const int Equilibration_step  = param.int_value("Equilibration_step");
  static const int relaxation_modulus  = param.int_value("relaxation_modulus");
  static const double rate_of_star_polymer = param.value("rate_of_star_polymer");
  static const int N_star_polymer          = int(N_polymer*rate_of_star_polymer);
  static double two_third_dt       = (2.0 / 3.0) * dt;
  static double two_dt_over_3_PI2  = 2.0*dt/( 3*M_PI*M_PI );
  static double inv_De             = 1.0 / De;
  static const int X=0, Y=1, Z=2;
  static const int XX=0, YY=1, ZZ=2, XY=3, YZ=4, ZX=5;
  static const int head=0, tail=1;
  static int counter = 0;
  double L_affine, dL, dL_affine;
  int    PC=0, SL=1;
  static bool first_flag = true;
  static int state = 0;
  static double* wd = (w+N_polymer);
  static double** debug_L;

  // 20180112 (TS)
  static const double ave_data         = param.value("ave_data");
  static const int Data_step           = param.int_value("Data_step");
  static const int Analysis_start_step = (int)( Equilibration_step+ave_data*Data_step+0.1);
  
  if( first_flag==true ){
    ALLOC_2D( debug_L, N_polymer, 2 );
    first_flag=false;
  }
	
  int NEW = (NOW+1)%2;

  // Restoring 
  for( int p=0; p<N_polymer; p++ ){
    for( int k=0; k<=Ze[p]-1; k++ ){
      SL_pos[NEW][p][k][X]=SL_pos[NOW][p][k][X];
      SL_pos[NEW][p][k][Y]=SL_pos[NOW][p][k][Y];
      SL_pos[NEW][p][k][Z]=SL_pos[NOW][p][k][Z];
    }
    L[NEW][p] = L[NOW][p];
  }

  for( int p=0; p<N_polymer; p++ ){

    double L_max;
    double Leq;
    double lambda_m;
    double inv_lambda_m2;
    double L_max_099;
    double L_max_090;
    double L_min;
    double tauR;
    double inv_Zeq;
    double inv_tauR;
    double inv_Zeq2;
    double ex;
    double sqrt_two_dt_over_3_PI2_Zeq;
    double sqrt_two_Zeq_dt_over_3_PI2_De;    
    double sqrt_Zeq_over_3_1_minus_ex2; 
    
    if( Melt_Flag == 0 ){
      L_max         = Zeq[p]*Lmax_factor;
      Leq           = Zeq[p]; // Modified by T.Sato(20170421)
      //Leq           = (Zeq[p]+2.0);
      lambda_m      = L_max/Leq;
      inv_lambda_m2 = 1.0/( lambda_m*lambda_m );
      L_max_099     = 0.99*L_max;
      L_max_090     = 0.90*L_max;
      L_min         = 0.1;
      tauR          = aT*Zeq[p]*Zeq[p]; // (Changed by T. Sato 20161215)
      inv_Zeq       = 1.0/Zeq[p];
      inv_tauR      = 1.0/tauR;
      inv_Zeq2      = inv_Zeq*inv_Zeq;
      
      if( Micro_or_Macro == 0 ){
	ex            = exp( - inv_tauR * dt );
      }else if( Micro_or_Macro == 1 ){
	ex            = exp( - inv_De * dt );
      }
      
      sqrt_two_dt_over_3_PI2_Zeq 
	= sqrt( two_dt_over_3_PI2 * inv_aT * inv_Zeq); // tau_e (Changed by T. Sato 20161215), rewrite (20180508)
      sqrt_two_Zeq_dt_over_3_PI2_De 
	= sqrt( two_dt_over_3_PI2 * inv_De * Zeq[p]); // tau_macro
      sqrt_Zeq_over_3_1_minus_ex2 
	= sqrt( Zeq[p]/3.0*( 1 - ex*ex ) );
     
    }else if( Melt_Flag == 1 ){
      L_max         = Zeq[p]*Lmax_factor;
      // ==========Changed by T.Sato(20160822)==========
      if( p <N_star_polymer ){
	Leq           = Zeq[p];
	tauR          = 4.0*aT*Zeq[p]*Zeq[p];
      }else{
	Leq           = Zeq[p];
	tauR          = aT*Zeq[p]*Zeq[p];
      }
      // ================================================
      lambda_m      = L_max/Leq;
      inv_lambda_m2 = 1.0/( lambda_m*lambda_m );
      L_max_099     = 0.99*L_max;
      L_max_090     = 0.90*L_max;
      L_min         = 0.1;      
      inv_Zeq       = 1.0/Zeq[p];
      inv_tauR      = 1.0/tauR;
      inv_Zeq2      = inv_Zeq*inv_Zeq;
      ex            = exp( - inv_tauR*dt );
      
      sqrt_two_dt_over_3_PI2_Zeq 
	= sqrt( two_dt_over_3_PI2 * inv_Zeq);
      sqrt_Zeq_over_3_1_minus_ex2 
	= sqrt( Zeq[p]/3.0*( 1 - ex*ex ) );
      
    } // Melt_flag
    
    // Time evolution //
    // 1. Affine deformation

    double L_now = 0.0;
    for( int k=0; k<=Ze[p]-2; k++ ){
      double dx = SL_pos[NOW][p][k+1][X]-SL_pos[NOW][p][k][X];
      double dy = SL_pos[NOW][p][k+1][Y]-SL_pos[NOW][p][k][Y];
      double dz = SL_pos[NOW][p][k+1][Z]-SL_pos[NOW][p][k][Z];
      L_now += sqrt( dx*dx + dy*dy + dz*dz ); 
    }

    for( int k=0; k<=Ze[p]-1; k++ ){

      double px =SL_pos[NOW][p][k][X];
      double py =SL_pos[NOW][p][k][Y];
      double pz =SL_pos[NOW][p][k][Z];
      //modified by harada(20140830)
      double factor_D = 1.0;
      if( L[NOW][p] < L_max_099 ){
	if(relaxation_modulus == 0){
	  if( step >= Equilibration_step ){
	
	    SL_pos[NEW][p][k][X] +=( D[X][X]*px + D[X][Y]*py + D[X][Z]*pz )*dt*factor_D;
	    SL_pos[NEW][p][k][Y] +=( D[Y][X]*px + D[Y][Y]*py + D[Y][Z]*pz )*dt*factor_D;
	    SL_pos[NEW][p][k][Z] +=( D[Z][X]*px + D[Z][Y]*py + D[Z][Z]*pz )*dt*factor_D;

	  }
	}
	if(relaxation_modulus == 1){
	  if( step == Equilibration_step ){
	
	    SL_pos[NEW][p][k][X] +=( D[X][X]*px + D[X][Y]*py + D[X][Z]*pz )*factor_D;
	    SL_pos[NEW][p][k][Y] +=( D[Y][X]*px + D[Y][Y]*py + D[Y][Z]*pz )*factor_D;
	    SL_pos[NEW][p][k][Z] +=( D[Z][X]*px + D[Z][Y]*py + D[Z][Z]*pz )*factor_D;

	  }
	}
      }else{
	SL_pos[NEW][p][k][X] =px;
	SL_pos[NEW][p][k][Y] =py;
	SL_pos[NEW][p][k][Z] =pz;
      }
    }

    L_affine = 0.0;
    for( int k=0; k<=Ze[p]-2; k++ ){

      r[p][k][X]=SL_pos[NEW][p][k+1][X]-SL_pos[NEW][p][k][X];
      r[p][k][Y]=SL_pos[NEW][p][k+1][Y]-SL_pos[NEW][p][k][Y];
      r[p][k][Z]=SL_pos[NEW][p][k+1][Z]-SL_pos[NEW][p][k][Z];

      L_affine += sqrt( + r[p][k][X] * r[p][k][X]
			+ r[p][k][Y] * r[p][k][Y]
			+ r[p][k][Z] * r[p][k][Z] );

    }
    
    debug_L[p][0] = L_affine+s[NEW][p][head]+s[NEW][p][tail];
  	
    // 2. Contour length fluctuation

    if( Ze[p] == 0.0 ){
      double hL = 0.5*L[NOW][p];
      s[NEW][p][head] = hL;
      s[NEW][p][tail] = hL;
    }else{

      dL_affine = L_affine - L_now;
      double lambda = L[NOW][p]/Leq;
      double lambda2= lambda*lambda;
      double factor = ( 1 - inv_lambda_m2 )/( 1 - lambda2*inv_lambda_m2 );
      
      //factor = abs(factor);
      if( Micro_or_Macro == 0 ){
	
	dL =
	  - (   L[NOW][p]*factor - Leq - dL_affine*tauR/dt )*( 1 - ex )
	  + sqrt_Zeq_over_3_1_minus_ex2 *w[p];
      
      }else if( Micro_or_Macro == 1 ){

	dL =
	  - (   L[NOW][p]*factor - Leq - dL_affine*De/dt )*( 1 - ex )
	  + sqrt_Zeq_over_3_1_minus_ex2 *w[p];
      
      }
      
	
      L[NEW][p] += dL;
      double hdL1 = 0.5*(L[NEW][p] - L[NOW][p] - dL_affine);
      
      debug_L[p][1] = L[NEW][p];
      if( L[NEW][p] < 0 ){
	L[NEW][p] = L_min;
      }
      // deleted by harada (20140901)
      // re-added by TS (20180420)
      else if( L[NEW][p] > L_max_099 ){
	L[NEW][p] = L_max_099;
      }

      double hdL  = 0.5*(dL-dL_affine);
      //printf("hdL_we = %e hdL_taki = %e diff=%e\n", hdL, hdL1, hdL-hdL1);
      if( Melt_Flag == 0 ){
	s[NEW][p][head] = s[NOW][p][head] + hdL1;
	s[NEW][p][tail] = s[NOW][p][tail] + hdL1;
      }else if(Melt_Flag ==1 ){
	if( p < N_star_polymer ){
	  s[NEW][p][head] = L[NEW][p] - L_affine;
	  s[NEW][p][tail] = 0.0;
	}else{
	  s[NEW][p][head] = s[NOW][p][head] + hdL1;
	  s[NEW][p][tail] = s[NOW][p][tail] + hdL1;
	}
      }
      //s[NEW][p][head] = s[NOW][p][head] + hdL;
      //s[NEW][p][tail] = s[NOW][p][tail] + hdL;

    }
      
    // 3. Reptation
    if( Melt_Flag == 0 ){    
      if( Ze[p] >= 1 ){
	
	// int pm = 2*(int)( 2*drand48() )-1;
	// double dx_rep = pm * sqrt_two_dt_over_3_PI2_inv_Zeq;
	// dx_rep = pm * sqrt( dt_over_3_PI2 / Ze[p] );
	
	double dx_rep;
	if( Micro_or_Macro == 0 ){
	  dx_rep = sqrt_two_dt_over_3_PI2_Zeq * wd[p];
	}else if( Micro_or_Macro == 1 ){
	  dx_rep = sqrt_two_Zeq_dt_over_3_PI2_De * wd[p];
	}
	s[NEW][p][head] += dx_rep;
	s[NEW][p][tail] -= dx_rep;

      }
    }else if( Melt_Flag == 1 ){
      if( p >= N_star_polymer ){
	if( Ze[p] >= 1 ){
	
	  // int pm = 2*(int)( 2*drand48() )-1;
	  // double dx_rep = pm * sqrt_two_dt_over_3_PI2_inv_Zeq;
	  // dx_rep = pm * sqrt( dt_over_3_PI2 / Ze[p] );
	  
	  double dx_rep = sqrt_two_dt_over_3_PI2_Zeq * wd[p];
	  s[NEW][p][head] += dx_rep;
	  s[NEW][p][tail] -= dx_rep;

	}
      }
    }
  }// For loop of p

  // 4. Constraint renewal
  for( int p=0; p<N_polymer; p++ ){
    //printf("p=%d\n",p);
    //================================================================
    if( s[NEW][p][head] >= 1.0 ){
      New_entanglement( param,NEW,p,head,Ze,Zeq,SL_pair,SL_pos,r,s,L );
      if( step >= Analysis_start_step ){
	N_hook[0][p]++;
      }
    }
    //================================================================
    if( s[NEW][p][tail] >= 1.0 ){
      New_entanglement( param,NEW,p,tail,Ze,Zeq,SL_pair,SL_pos,r,s,L );
      if( step >= Analysis_start_step ){
	N_hook[0][p]++;
      }
    }
    //================================================================
    if( s[NEW][p][head] < 0.0 && Ze[p] >= 1 ){
      Release_of_entangled_polymer(param,NEW,p,head,Ze,SL_pair,SL_pos,r,s,L);
      if( step >= Analysis_start_step ){
	N_hook[1][p]++;
      }
    }
    //================================================================
    if( s[NEW][p][tail] < 0.0 && Ze[p] >= 1 ){
      Release_of_entangled_polymer(param,NEW,p,tail,Ze,SL_pair,SL_pos,r,s,L);
      if( step >= Analysis_start_step ){
	N_hook[1][p]++;
      }
    }
    //================================================================
  }//  

  for( int p=0; p<N_polymer; p++ ){
   
    double px=0.0, py=0.0, pz=0.0;
    for( int k=0; k<=Ze[p]-1; k++ ){
      px += SL_pos[NEW][p][k][X];
      py += SL_pos[NEW][p][k][Y];
      pz += SL_pos[NEW][p][k][Z];
    }
    if( Ze[p] >= 1 ){ 
      px /= Ze[p], py /= Ze[p], pz /= Ze[p]; 
    }

    for( int k=0; k<=Ze[p]-1; k++ ){
      SL_pos[NEW][p][k][X] -= px;
      SL_pos[NEW][p][k][Y] -= py;
      SL_pos[NEW][p][k][Z] -= pz;
    }
    if( L[NEW][p]>Zeq[p]*Lmax_factor ){ 
      // cout<<" p= "<<p<<endl;
      // cout<<"L= "<<L[NEW][p]<<" polymer is too long!"<<endl;
      // cout<<"L_after_affine = "<<debug_L[p][0]<<" L_after_relax = "<<debug_L[p][1]<<endl;
      // cout<<"step = "<<step<<endl;
      // exit(1);
    }
  }

  Calc_Stress( param, NOW, Ze, Zeq, r, sigma, SS, L, s );

  return(0);
}
//-----------------------------------------------------------------------
int Calc_Stress( IOParameterSet &param,
		 const int      &NOW, 
		 int*           &Ze,
		 int*           &Zeq,
		 double***      &r,
		 double*        &sigma,
		 double*        &SS,
		 double**       &L,
		 double***      &s )
{ 
  static const int    N_polymer= param.int_value("N_polymer");
  static const double Lmax_factor   = param.value("Lmax_factor");
  static const double T_ref         = param.value("T_ref");
  static const double T             = param.value("T");
  static const double bT            = T / T_ref;
  static const double inv_Np   = 1.0/(N_polymer);
  static const int X=0, Y=1, Z=2;
  static const int XX=0, YY=1, ZZ=2, XY=3, YZ=4, ZX=5;
  static const int head=0, tail=1;
  static double Rx=0.0, Ry=0.0, Rz=0.0;

  const int NEW=(NOW+1)%2;  

  sigma[XX] = 0.0;
  sigma[YY] = 0.0;
  sigma[ZZ] = 0.0;
  sigma[XY] = 0.0;
  sigma[YZ] = 0.0;
  sigma[ZX] = 0.0;
  SS[XX] = 0.0;
  SS[YY] = 0.0;
  SS[ZZ] = 0.0;
  SS[XY] = 0.0;
  SS[YZ] = 0.0;
  SS[ZX] = 0.0;

  for( int p=0; p<N_polymer; p++ ){

    double L_max    = Zeq[p]*Lmax_factor;
    //double Leq      = (Zeq[p]+2.0); //20170502(TS)
    double Leq      = Zeq[p];
    double inv_Leq  = 1.0/Leq;
    double inv_Zeq  = 1.0/Zeq[p];
    double inv_Zeq2 = inv_Zeq*inv_Zeq;
    double lambda_m = L_max*inv_Leq;
    double inv_lambda_m2 = 1.0/( lambda_m*lambda_m );
    

    double sum_xx = 0.0;
    double sum_yy = 0.0;
    double sum_zz = 0.0;
    double sum_xy = 0.0;
    double sum_yz = 0.0;
    double sum_zx = 0.0;
    
    Rx=0.0;
    Ry=0.0;
    Rz=0.0;

    double lambda  = L[NEW][p]*inv_Leq;
    double lambda2 = lambda*lambda;

    double factor
      = ( 1-inv_lambda_m2 )/( 1-lambda2*inv_lambda_m2 )*L[NEW][p]*inv_Zeq;

    for( int k=0; k<Ze[p]-1; k++ ){

      double* &dr = r[p][k];
      double xx = dr[X]*dr[X];
      double yy = dr[Y]*dr[Y];
      double zz = dr[Z]*dr[Z];

      double xy = dr[X]*dr[Y];
      double yz = dr[Y]*dr[Z];
      double zx = dr[Z]*dr[X];
      
      Rx += dr[X];
      Ry += dr[Y];
      Rz += dr[Z];

      double length2  = xx + yy + zz;
      double inv_len2 = 1.0/length2;
      double inv_len  = sqrt( inv_len2 );

      sum_xx += xx *inv_len;
      sum_yy += yy *inv_len;
      sum_zz += zz *inv_len;
      sum_xy += xy *inv_len;
      sum_yz += yz *inv_len;
      sum_zx += zx *inv_len;
      
    }

    sum_xx *= factor;
    sum_yy *= factor;
    sum_zz *= factor;
    sum_xy *= factor;
    sum_yz *= factor;
    sum_zx *= factor;

    double R2 = ( Rx*Rx + Ry*Ry + Rz*Rz );
    double inv_R = 0.0;
    if( R2 != 0.0 ){ inv_R = sqrt( 1.0/R2 ); }

    sigma[XX] += sum_xx*inv_Zeq;
    sigma[YY] += sum_yy*inv_Zeq;
    sigma[ZZ] += sum_zz*inv_Zeq;
    sigma[XY] += sum_xy*inv_Zeq;
    sigma[YZ] += sum_yz*inv_Zeq;
    sigma[ZX] += sum_zx*inv_Zeq;

    Rx *= inv_R;
    Ry *= inv_R;
    Rz *= inv_R;
    SS[XX] += Rx*Rx;
    SS[YY] += Ry*Ry;
    SS[ZZ] += Rz*Rz;
    SS[XY] += Rx*Ry;
    SS[YZ] += Ry*Rz;
    SS[ZX] += Rz*Rx;

  }// for loop p

  sigma[XX] *= inv_Np;
  sigma[YY] *= inv_Np;
  sigma[ZZ] *= inv_Np;
  sigma[XY] *= inv_Np;
  sigma[YZ] *= inv_Np;
  sigma[ZX] *= inv_Np;

  // T.Sato (20170125), rewrite (20180508)
  sigma[XX] *= bT;
  sigma[YY] *= bT;
  sigma[ZZ] *= bT;
  sigma[XY] *= bT;
  sigma[YZ] *= bT;
  sigma[ZX] *= bT;
  
  SS[XX] *= inv_Np;
  SS[YY] *= inv_Np;
  SS[ZZ] *= inv_Np;
  SS[XY] *= inv_Np;
  SS[YZ] *= inv_Np;
  SS[ZX] *= inv_Np;

  return(0);
}
//-----------------------------------------------------------------------
int Search_of_another_slip_link( IOParameterSet &param2,
				 const int &p,
				 const int &k,
				 int*      &Ze,
				 int*      &Zeq,
				 int**     &Entanglement,
				 int***    &SL_pair   )
{
  static int N_polymer = param2.int_value("N_polymer");
  static const int PC=0, SL=1;
  int APC, APC_2, ASL, ASL_2;

  //   Polymer chain ( Z : number of slip-link )
  //
  //          SL(0)   SL(1)    SL(2)          SL(Z-1)
  //    head----o-------o--------o------||-------o------tail
  //        <--->------->-------->          -----><---->
  //         s0    r(0)    r(1)            r(Z-2)   s1


  APC = genrand_int31() % N_polymer;
  ASL = genrand_int31() % Zeq[APC];

  //APC = (int)( drand48()*N_polymer );
  //ASL = (int)( drand48()*Zeq[APC]       );

  if(    Entanglement[APC][ASL] < 0
	 && APC != p
	 && Ze[APC] <= p+4 
	 ){   // << = Should be checked

    SL_pair[p][k][PC]     = APC;
    SL_pair[p][k][SL]     = ASL;
    SL_pair[APC][ASL][PC] = p;
    SL_pair[APC][ASL][SL] = k;

    Entanglement[p  ][k  ] = 1;
    Entanglement[APC][ASL] = 1;

    Ze[p  ]++;
    Ze[APC]++;
    return(0);

  }else{

    for( int kk=0; kk<Zeq[APC]; kk++ ){

      if( (ASL+1) > Zeq[APC]-1 ){
	ASL = 0;
      }else{
        ASL++;
      }

      if(    Entanglement[APC][ASL] < 0
	     && APC != p
	     && Ze[APC] <= p+4 ){  // << = Should be checked

        SL_pair[p][k][PC]     = APC;
        SL_pair[p][k][SL]     = ASL;
        SL_pair[APC][ASL][PC] = p;
        SL_pair[APC][ASL][SL] = k;

        Entanglement[p  ][k  ] = 1;
        Entanglement[APC][ASL] = 1;

        Ze[p  ]++;
        Ze[APC]++;
        return(0);
      }
    }

    for( int pp=0; pp<N_polymer; pp++ ){

      if( (APC+1) > N_polymer-1 ){
        APC = 0;
      }else{
        APC++;
      }
      //added by harada (20140508)
      ASL = genrand_int31()%Zeq[APC];
      
      if(    Entanglement[APC][ASL] < 0
	     && APC != p
	     && Ze[APC] <= p+4 ){  // << = Should be checked

        SL_pair[p][k][PC]     = APC;
        SL_pair[p][k][SL]     = ASL;
        SL_pair[APC][ASL][PC] = p;
        SL_pair[APC][ASL][SL] = k;

        Entanglement[p  ][k  ] = 1;
        Entanglement[APC][ASL] = 1;

        Ze[p  ]++;
        Ze[APC]++;
        return(0);

      }else{

        for( int kk=0; kk<Zeq[APC]; kk++ ){

          if( (ASL+1) > Zeq[APC]-1 ){
            ASL = 0;
          }else{
            ASL++;
          }

          if(    Entanglement[APC][ASL] < 0
		 && APC != p
		 && Ze[APC] <= p+4 ){  // << = Should be checked

            SL_pair[p][k][PC]     = APC;
            SL_pair[p][k][SL]     = ASL;
            SL_pair[APC][ASL][PC] = p;
            SL_pair[APC][ASL][SL] = k;

            Entanglement[p  ][k  ] = 1;
            Entanglement[APC][ASL] = 1;

            Ze[p  ]++;
            Ze[APC]++;
            return(0);
          }
        }
      }
    }
  }
  
  printf( "No proper pair!\tp = %d\tk = %d\n", p, k );
  //modified by harada (20140425)
  while(1){
    APC_2 =  genrand_int31() % N_polymer;
    ASL_2 =  genrand_int31() % Zeq[APC_2];
    if( APC_2!=p && Entanglement[APC_2][ASL_2]>0 && SL_pair[APC_2][ASL_2][PC] !=p ){
      break; 
    } 
  }
  if( Entanglement[APC_2][ASL_2]>0 ){
    if( SL_pair[APC_2][ASL_2][PC] != p ){
      int PC2, SL2;
      PC2 = SL_pair[APC_2][ASL_2][PC];
      SL2 = SL_pair[APC_2][ASL_2][SL];

     
      Entanglement[PC2][SL2] = -1;
      Ze[PC2] -= 1;
      
      SL_pair[p][k][PC] = APC_2;
      SL_pair[p][k][SL] = ASL_2;
      SL_pair[APC_2][ASL_2][PC] = p;
      SL_pair[APC_2][ASL_2][SL] = k;
      
      Entanglement[p][k] = 1;
      Entanglement[APC_2][ASL_2] = 1;

      Ze[p]++;

      return(0);
    }
  }
  printf( "Error!!  Search_of_another_slip_link does not work!");
  exit(1);

  return(0);
}
//-----------------------------------------------------------------------
int Set_InitValue_Pasta( IOParameterSet &param,
			 int        N,
			 char**     &argv,
			 int**       &Ze,
			 int**       &Zeq,
			 double**    &sigma,
			 double**    &SS,
			 int****     &SL_pair,		   
			 double***** &SL_pos,
			 double****  &r,
			 double****  &s,
			 double***   &L,
			 int***      &Entanglement )
{
  static int    N_polymer = param.int_value("N_polymer");
  static int    seed      = param.int_value("seed");
  static const int Melt_Flag               = param.int_value("Melt_Flag"); // 0: Linear, 1: Linear and Star
  static const double rate_of_star_polymer = param.value("rate_of_star_polymer");
  static const int N_star_polymer          = int(N_polymer*rate_of_star_polymer);  
  static const int NOW=0, NEW=1;
  static int restart_flag = param.int_value("restart_flag");
  static int restart_entangle = param.int_value("restart_entangle");
  static const int X=0, Y=1, Z=2;
  static const int XX=0, YY=1, ZZ=2, XY=3, YZ=4, ZX=5;
  static const int head=0, tail=1;
  static const int PC=0, SL=1;
  static const double TWO_PI    = 2*M_PI;

  srand48(seed);

  if( restart_flag == 1 ){
    /*
    //-------------------------------
    // Z and s data
    static FILE *fp = fopen( argv[4],"r"); // ex.) Ze.data 
    int dummy;

    for( int p=0; p<N_polymer; p++ ){

    fscanf( fp, "%d\t%d\t%lf\t%lf\n",
    &dummy, 
    &Ze[p], 
    &s[NEW][p][head], 
    &s[NEW][p][tail]  );

    s[NOW][p][head] = s[NEW][p][head];
    s[NOW][p][tail] = s[NEW][p][tail];
    }
    fclose(fp);
    //-------------------------------

    // Slip-link data
    fp = fopen( argv[5],"r"); // SL.data

    for( int p=0; p<N_polymer; p++ ){

    for( int k=0; k<Ze[p]; k++ ){

    fscanf( fp, "%d\t%d\t%d\t%d\t%lf\t%lf\t%lf\n",
    &dummy, 
    &dummy, 
    &SL_pair[p][k][PC],
    &SL_pair[p][k][SL],
    &SL_pos[NEW][p][k][X],
    &SL_pos[NEW][p][k][Y],
    &SL_pos[NEW][p][k][Z]  );

    SL_pos[NOW][p][k][X] = SL_pos[NEW][p][k][X];
    SL_pos[NOW][p][k][Y] = SL_pos[NEW][p][k][Y];
    SL_pos[NOW][p][k][Z] = SL_pos[NEW][p][k][Z];
    }
    }
    fclose(fp);
    //-------------------------------

    for( int p=0; p<N_polymer; p++ ){

    for( int k=0; k<Ze[p]-1; k++ ){

    r[p][k][X] = SL_pos[NEW][p][k+1][X] - SL_pos[NEW][p][k][X];
    r[p][k][Y] = SL_pos[NEW][p][k+1][Y] - SL_pos[NEW][p][k][Y];
    r[p][k][Z] = SL_pos[NEW][p][k+1][Z] - SL_pos[NEW][p][k][Z];
    }
    }
    */
  }else{

    //   Polymer chain ( Z : number of slip-link )
    //
    //          SL(0)   SL(1)    SL(2)          SL(Z-1)
    //    head----o-------o--------o------||-------o------tail
    //        <--->------->-------->          -----><---->
    //         s0    r(0)    r(1)            r(Z-2)   s1

    for( int ip=0; ip<N; ip++ ){
      for( int p=0; p<N_polymer; p++ ){
	Ze[ip][p] = 0;
	for( int k=0; k<Zeq[ip][p]; k++ ){
	  Entanglement[ip][p][k] = -1;
	}
      }

      for( int p=0; p<N_polymer; p++ ){

	int k=0;
	SL_pos[ip][NEW][p][k][X] = 0;
	SL_pos[ip][NEW][p][k][Y] = 0;
	SL_pos[ip][NEW][p][k][Z] = 0;;
	SL_pos[ip][NOW][p][k][X] = SL_pos[ip][NEW][p][k][X];
	SL_pos[ip][NOW][p][k][Y] = SL_pos[ip][NEW][p][k][Y];
	SL_pos[ip][NOW][p][k][Z] = SL_pos[ip][NEW][p][k][Z];

	for( int k=1; k<Zeq[ip][p]; k++ ){

	  double rx, ry, rz, rho;
	  while( 1 ){
	    rx = 2*( drand48()-0.5 );
	    ry = 2*( drand48()-0.5 );
	    rz = 2*( drand48()-0.5 );
	    rho= sqrt( rx*rx + ry*ry + rz*rz );
	    if( rho < 1.0 ){ break; }
	  }
	  double dx = rx/rho;
	  double dy = ry/rho;
	  double dz = rz/rho;
	  r[ip][p][k-1][X] = dx;
	  r[ip][p][k-1][Y] = dy;
	  r[ip][p][k-1][Z] = dz;

	  SL_pos[ip][NEW][p][k][X] = SL_pos[ip][NEW][p][k-1][X] + r[ip][p][k-1][X];
	  SL_pos[ip][NEW][p][k][Y] = SL_pos[ip][NEW][p][k-1][Y] + r[ip][p][k-1][Y];
	  SL_pos[ip][NEW][p][k][Z] = SL_pos[ip][NEW][p][k-1][Z] + r[ip][p][k-1][Z];
	  SL_pos[ip][NOW][p][k][X] = SL_pos[ip][NEW][p][k  ][X];
	  SL_pos[ip][NOW][p][k][Y] = SL_pos[ip][NEW][p][k  ][Y];
	  SL_pos[ip][NOW][p][k][Z] = SL_pos[ip][NEW][p][k  ][Z];
	}

	if( Melt_Flag == 0 ){
	  // modified by TT (20130907)
	  s[ip][NOW][p][head] = drand48();                 //drand48() -> generate random number (0,1).
	  s[ip][NOW][p][tail] = 1.0 - s[ip][NOW][p][head]; //changed by TS (20170421) 
	  s[ip][NEW][p][head] = s[ip][NOW][p][head];
	  s[ip][NEW][p][tail] = s[ip][NOW][p][tail];
	}else if( Melt_Flag ==1 ){
	  if(p<N_star_polymer){
	    s[ip][NOW][p][head] = drand48();
	    s[ip][NOW][p][tail] = 0.0;
	    s[ip][NEW][p][head] = s[ip][NOW][p][head];
	    s[ip][NEW][p][tail] = s[ip][NOW][p][tail];
	  }else{
	    s[ip][NOW][p][head] = drand48();
	    s[ip][NOW][p][tail] = 1.0 - s[ip][NOW][p][head]; //changed by TS (20170421)
	    s[ip][NEW][p][head] = s[ip][NOW][p][head];
	    s[ip][NEW][p][tail] = s[ip][NOW][p][tail];
	  }	  
	}
	
	//added by harada (20140627)
	double length = 0.0;
	for( int k=0; k<=Zeq[ip][p]-2; k++ ){
	  double dx = r[ip][p][k][X];
	  double dy = r[ip][p][k][Y];
	  double dz = r[ip][p][k][Z];	
	  length += sqrt(dx*dx+dy*dy+dz*dz);
	}
	L[ip][NOW][p] = length + ( s[ip][NOW][p][head] + s[ip][NOW][p][tail] );
	L[ip][NEW][p] = L[ip][NOW][p];
      }
      if( restart_entangle == 0 ){
	//modified by harada (20140425)
	for( int p=0; p<N_polymer; p++ ){
	  for( int k=0; k<Zeq[ip][p]; k++ ){
	    if( Entanglement[ip][p][k] < 0 ){
	      Search_of_another_slip_link( param, p, k, Ze[ip], Zeq[ip], Entanglement[ip], SL_pair[ip] );
	    }
	  }
	}
	
	//int k=0;
	//while(1){
	//for( int p=0; p<N_polymer; p++ ){
	//  if( Entanglement[p][k] < 0 ){
	//    Search_of_another_slip_link( param, p, k, Ze[ip],
	//				 Entanglement, SL_pair[ip] );
	//  }
	//}
	//k++;
	//if( k == Zeq_const ){ break; }
	//}
	
	//debug
	//for(int p=0; p<N_polymer; p++ ){
	//for(int k=0; k<Zeq[ip][p]; k++ ){
	//  fout_debug<<ip<<"\t"<<p<<"\t"<<k<<"\t"<<Entanglement[ip][p][k]<<endl;
	//}
	//}
	//fout_debug<<endl;
	
	//for(int p=0; p<N_polymer; p++ ){
	//for(int k=0; k<Zeq[ip][p]; k++ ){
	//fout_debug<<ip<<"\t"<<p<<"\t"<<k<<"\t"<<SL_pair[ip][p][k][0]<<"\t"<<SL_pair[ip][p][k][1]<<endl;
	//}
	//}

	//fout_debug << "###########################################"<<endl;

	for( int p=0; p<N_polymer; p++ ){
	  for( int k=0; k<Ze[ip][p]; k++ ){
	    if( Entanglement[ip][p][k] <0 ){
	      cout<<"Error! Search_of_another_slip_link does not work!"<<endl;
	      cout<<"ip= "<<ip<<" p= "<<p<<" k="<<k<<endl;
	      exit(1);
	    }
	    //debug
	    
	    int pair_PC = SL_pair[ip][p][k][0];
	    int pair_SL = SL_pair[ip][p][k][1];
	    if( p != SL_pair[ip][pair_PC][pair_SL][0] || k != SL_pair[ip][pair_PC][pair_SL][1] ){
	      cout<<"faital_error!!!"<<endl;
	      exit(1);
	    } 
	    
	  }
	}
      }
    }//ip
    ALLOC_3D_int_FREE( Entanglement, N, N_polymer );
  }
  
  // Calculation of contour length and Stress on particle
  for( int ip=0; ip<N; ip++ ){
    Calc_Stress( param, NOW, Ze[ip], Zeq[ip], r[ip], sigma[ip], SS[ip], L[ip], s[ip] );
  }
  return(0);
}

//------------------------------------------------------------------------
int Write_Data_Pair( IOParameterSet &param,
 		     int*           &Ze,
 		     int**          &Entanglement,
		     double*        &sigma )
{
  static int N_polymer = param.int_value("N_polymer");
  static int Zeq_data       = param.int_value("Zeq");
  static int counter = 0;
  static const int NEW=1;
  static const int XX=0, YY=1, ZZ=2;
  char filename[32];

  sprintf(filename, "pair_check%.4d.dat", counter++);
  FILE *fp = fopen(filename,"w");

  for( int p=0; p<N_polymer; p++ ){
    for( int k=0; k<Ze[p]; k++ ){
      fprintf( fp, "p=%d\tk=%d\t%d\n", p, k, Entanglement[p][k] );
    }
  }
  fclose(fp);
  return(0);
}
//-----------------------------------------------------------------------
int Write_Data_Step( IOParameterSet &param,
		     int             N,
 		     int*           &Ze,
 		     double*        &sigma,
 		     double*        &SS,
 		     double*        &L,
 		     const int      &step,
		     int             ip,
		     int             real_ip )
{
  static char filename[32];
  static FILE  *fp;
  static const int    N_polymer = param.int_value("N_polymer");
  static const double dt        = param.value("dt");
  static const double T_ref         = param.value("T_ref");
  static const double T             = param.value("T");
  static const double delta_T       = T - T_ref;
  static const double C1            = param.value("C1");
  static const double C2            = param.value("C2");
  static const double aT            = exp(-2.302585093*C1*delta_T/(C2+delta_T));
  static const int Equilibration_step      = param.int_value("Equilibration_step");
  static const double rate_of_long_polymer = param.value("rate_of_long_polymer");
  static const int N_long_polymer          = int(N_polymer*rate_of_long_polymer);
  static const double inv_Np               = 1.0/double(N_polymer);
  static const double inv_Nlp              = 1.0/double(N_long_polymer);
  static const double inv_Nsp              = 1.0/double(N_polymer - N_long_polymer);
  static const int XX=0, YY=1, ZZ=2, XY=3, YZ=4, ZX=5;
  static bool flag=true;

  int rank;
  MPI_Comm_rank( MCW, &rank );
  
  double t = dt*(step-Equilibration_step);
  
  sprintf( filename, "sigma_L_ip_%.4d.dat", real_ip );
  if( flag == true ){
    if(rank == 0){
      sprintf( filename, "Name_of_Data.dat" ); //(20180405TS)
      fp = fopen( filename, "w" );
      fprintf( fp, 
	       "#  (1) t, (2) Z_ave, (3) Zl_ave, (4) Zs_ave, (5) L_ave,"
	       " (6) sigmaxx, (7) sigmayy, (8) sigmazz,"
	       " (9) sigmaxy, (10) sigmayz, (11) sigmazx,"
	       " (12) Z_min, (13) Z_max, (14) L_min, (15) L_max, (16) Num_of_Z_zero, "
	       " (17) Sxx, (18) Syy, (19) Szz, (20) Sxy, (21) Syz, (22) Szx"
	       "\n" );
      fclose(fp);
    }
    flag = false;
  }

  int Z_max=-1, Z_min=1000000;
  int count_Z_zero=0;
  double L_ave =0.0,  Z_ave=0.0;
  double Zs_ave=0.0, Zl_ave=0.0;
  double L_min=1.0e+10, L_max=-1.0e+10;
  
  for( int p=0; p<N_polymer; p++ ){ 
    L_ave += L[p]; Z_ave += Ze[p]; 
    if( Z_max < Ze[p] ){ Z_max=Ze[p]; }else if( Z_min > Ze[p] ){ Z_min=Ze[p]; }
    if( L_max <  L[p] ){ L_max= L[p]; }else if( L_min >  L[p] ){ L_min= L[p]; }
    if( p< N_long_polymer ){ Zl_ave += Ze[p]; }else{ Zs_ave += Ze[p]; }
    if( Ze[p] == 0 ){ count_Z_zero++; }
  }
  
  L_ave *= inv_Np;
  Z_ave *= inv_Np;
  if( N_long_polymer != 0 ){
    Zl_ave *= inv_Nlp;
  }
  Zs_ave *= inv_Nsp;
  
  sprintf( filename, "sigma_L_ip_%.4d.dat", real_ip );
  fp = fopen( filename, "a" );
  fprintf( fp, 
  	   "%e\t%e\t%e\t%e\t%e\t%e\t"
  	   "%e\t%e\t%e\t%e\t%e\t%d\t"
  	   "%d\t%e\t%e\t%d\t%e\t%e\t"
  	   "%e\t%e\t%e\t%e\n", 
  	   t,            // 1
  	   Z_ave,        // 2
	   Zl_ave,       // 3
	   Zs_ave,       // 4
  	   L_ave,        // 5
  	   sigma[XX],    // 6
  	   sigma[YY],    // 7
  	   sigma[ZZ],    // 8
  	   sigma[XY],    // 9
  	   sigma[YZ],    // 10
  	   sigma[ZX],    // 11
  	   Z_min,        // 12
  	   Z_max,        // 13
  	   L_min,        // 14
  	   L_max,        // 15
	   count_Z_zero, // 16
  	   SS[XX],       // 17
  	   SS[YY],       // 18
  	   SS[ZZ],       // 19
  	   SS[XY],       // 20
  	   SS[YZ],       // 21
  	   SS[ZX]        // 22
  	   );

  fclose(fp);
  
  
  return(0);
}

//------------------------------------------------------------------------
int Write_Data( IOParameterSet &param,
 		int*           &Ze,
 		int***         &SL_pair,
 		double***      &SL_pos,
 		double***      &r,
 		double**       &s,
 		double*        &L,
		int             counter,
		int             ip )
{
  static const int N_polymer = param.int_value("N_polymer");
  static const int X=0, Y=1, Z=2;
  static const int XX=0, YY=1, ZZ=2, XY=3, YZ=4, ZX=5;
  static const int head=0, tail=1;
  static const int PC=0, SL=1;
  char filename[32];
  FILE *fp;
  //----------------------------------------------------------------------
  sprintf(filename, "./polymer_data/Time_Evolution/Other_data/Polym_Data_%.4d_time_%.4d.dat", ip, counter );
  fp = fopen(filename,"w");
  for( int p=0; p<N_polymer; p++ ){
    fprintf( fp, "%d\t%e\t%e\t%e\n",
	     Ze[p],
	     L[p],
	     s[p][head],
	     s[p][tail] );
  }
  fclose(fp);
  //----------------------------------------------------------------------
  sprintf(filename, "./polymer_data/Time_Evolution/SL_pos/SL_pos_%.4d_time_%.4d.dat", ip, counter );
  fp = fopen(filename,"w");
  for( int p=0; p<N_polymer; p++ ){
    //fprintf( fp, "# head =%e\ttail = %e\n",s[p][head],s[p][tail]); 
    double av_x =0, av_y=0, av_z=0;
    for( int k=0; k<Ze[p]; k++ ){
      av_x += SL_pos[p][k][X];
      av_y += SL_pos[p][k][Y];  
      av_z += SL_pos[p][k][Z];
    }
    av_x /= Ze[p];    av_y /= Ze[p];    av_z /= Ze[p];

    for( int k=0; k<Ze[p]; k++ ){
      fprintf( fp, "%e\t%e\t%e\n", 
 	       SL_pos[p][k][X]-av_x,
 	       SL_pos[p][k][Y]-av_y,
 	       SL_pos[p][k][Z]-av_z );
    }
    //fprintf( fp, "\n\n" );
  }
  fclose(fp);
  //----------------------------------------------------------------------
  /*
    sprintf(filename, "./polymer_data/r_ip_%.4d_%.4d.dat", ip, counter);
    fp = fopen(filename,"w");

    for( int p=0; p<N_polymer; p++ ){

    fprintf( fp, "\n# p = %d\thead =%e\ttail = %e\n", 
    p, s[p][head], s[p][tail] );
    for( int k=0; k<Ze[p]-1; k++ ){
    fprintf( fp, "%e\t%e\t%e\n", r[p][k][X], r[p][k][Y], r[p][k][Z] );
    }
    }

    fclose(fp);
  */
  //----------------------------------------------------------------------
  /*
    sprintf(filename, "./polymer_data/All_SL_pair_ip%.4d_%.4d.dat", ip, counter);
    fp = fopen(filename,"w");
    for( int p=0; p<N_polymer; p++ ){
    for( int k=0; k<Ze[p]; k++ ){
    fprintf( fp, "( %d , %d )----( %d , %d )\n",
    p, k, SL_pair[p][k][PC], SL_pair[p][k][SL] );
    }
    fprintf( fp, "\n" );
    }
    fclose(fp);
  */
  //----------------------------------------------------------------------
  return(0);
}
//-----------------------------------------------------------------------
int Write_Data_end( IOParameterSet &param,
		    int**          &N_hook,
		    int*           &Ze,
		    int***         &SL_pair,
		    double***      &SL_pos,
		    double**       &s,
		    double*        &ave_s,
		    double*        &dis_s,		    
		    int             ip,
		    int             real_ip,
		    int            &counter_analyze )
{
  static int N_polymer = param.int_value("N_polymer");
  static const int X=0, Y=1, Z=2;
  static const int head=0, tail=1;
  static const int PC=0, SL=1;
  static const double strain_rate = param.value("strain_rate");
  char filename[64];
  FILE *fp;

  //--------------------------------------------------------------
  double N_hook_ave   = 0.0;
  double N_dehook_ave = 0.0;
  double N_hook_dis   = 0.0;
  double N_dehook_dis = 0.0;
  for( int p=0; p<N_polymer; p++ ){
    double n_hook   = double(N_hook[0][p]) / double(counter_analyze);
    double n_dehook = double(N_hook[1][p]) / double(counter_analyze);
    N_hook_ave   += n_hook;
    N_dehook_ave += n_dehook;
  }
  N_hook_ave   /= N_polymer;
  N_dehook_ave /= N_polymer;

  for( int p=0; p<N_polymer; p++ ){
    double n_hook   = double(N_hook[0][p]) / double(counter_analyze);
    double n_dehook = double(N_hook[1][p]) / double(counter_analyze);
    double dn_hook    = n_hook   - N_hook_ave;
    double dn_dehook  = n_dehook - N_dehook_ave;
    N_hook_dis   += dn_hook*dn_hook;
    N_dehook_dis += dn_dehook*dn_dehook;
  }
  N_hook_dis   /= N_polymer;
  N_dehook_dis /= N_polymer;

  sprintf(filename, "./hook_and_s%.4d.dat", real_ip);
  fp = fopen(filename,"w");
  
  fprintf( fp, "%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",
	   double(real_ip+1)*strain_rate, 
	   N_hook_ave,
	   sqrt(N_hook_dis),
	   N_dehook_ave,
	   sqrt(N_dehook_dis),
	   ave_s[0]/double(counter_analyze),
	   sqrt(dis_s[0]/double(counter_analyze)/double(N_polymer)),
	   ave_s[1]/double(counter_analyze),
	   sqrt(dis_s[1]/double(counter_analyze)/double(N_polymer)) );
  
  fclose(fp);
  //--------------------------------------------------------------
  sprintf(filename, "./polymer_data/SL_data_ip_%.4d.dat", real_ip);
  fp = fopen(filename,"w");

  for( int p=0; p<N_polymer; p++ ){

    for( int k=0; k<Ze[p]; k++ ){

      fprintf( fp, "%d\t%d\t%d\t%d\t%e\t%e\t%e\n",
	       p, 
	       k,
	       SL_pair[p][k][PC],
	       SL_pair[p][k][SL],
	       SL_pos[p][k][X],
	       SL_pos[p][k][Y],
	       SL_pos[p][k][Z] );
    }
  }
  fclose(fp);
  //--------------------------------------------------------------
  sprintf(filename, "./polymer_data/Ze_data_ip_%.4d.dat", real_ip );
  fp = fopen(filename,"w");

  for( int p=0; p<N_polymer; p++ ){
    fprintf( fp, "%d\t%d\t%e\t%e\t%d\t%d\n",p,Ze[p],s[p][head],s[p][tail],N_hook[0][p],N_hook[1][p] );
  }

  fclose(fp);
  return(0);
}
//-----------------------------------------------------------------------
int Write_D_Data( IOParameterSet &param,
		  double***      &D )
{
  static const int N_particle = param.int_value("N_particle");
  static const int SPACE_DIM  = 3;
  static const int X=0, Y=1, Z=2;
  char filename[32];
  FILE *fp;

  sprintf( filename, "D_Data.dat" );
  fp = fopen(filename, "w" );

  for( int ip=0; ip<N_particle; ip++ ){
    fprintf( fp, "%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",
	     D[ip][X][X],
	     D[ip][X][Y],
	     D[ip][X][Z],
	     D[ip][Y][X],
	     D[ip][Y][Y],
	     D[ip][Y][Z],
	     D[ip][Z][X],
	     D[ip][Z][Y],
	     D[ip][Z][Z] );
  }
  fclose(fp);
  return(0);
}
//----------------------------------------------------------------------
int MY_MPI_scatter( IOParameterSet &param,
		    double*        &D_all_1D,
		    double*        &D_local_1D,
		    double***      &D,
		    int*           &N_in_PE,
		    int*           &N_disp,
		    int*           &N_in_PE_D,
		    int*           &N_disp_D )
{ 
  static const int SPACE_DIM = 3;
  int rank;

  MPI_Comm_rank( MCW, &rank );
  
  MPI_Scatterv( &D_all_1D[0], &N_in_PE_D[0], &N_disp_D[0],
		MPI_DOUBLE, &D_local_1D[0], N_in_PE_D[rank],
		MPI_DOUBLE, 0, MCW );
  

  for( int ip=0; ip<N_in_PE[rank]; ip++ ){
    for( int dim1=0; dim1<SPACE_DIM; dim1++ ){
      for( int dim2=0; dim2<SPACE_DIM; dim2++ ){
	D[ip][dim1][dim2] 
	  = double((ip+1)+N_disp[rank])*
	  D_local_1D[ ip*SPACE_DIM*SPACE_DIM+dim1*SPACE_DIM+dim2 ];
      }
    }
  }

  return(0);
}
//-----------------------------------------------------------------------
int MY_MPI_gather( IOParameterSet &param,
		   int             rank,
		   double**       &sigma,
		   double**       &SS,
		   double***      &D,
		   double**       &sigma_all,
		   double***      &D_all,
		   int*           &N_in_PE,
		   int*           &N_disp,
		   double*        &D_local_1D,
		   double*        &D_all_1D,
		   double*        &sigma_local_1D,
		   double*        &sigma_all_1D,
		   double*        &SS_local_1D,
		   double*        &SS_all_1D,
		   int*           &N_in_PE_D,
		   int*           &N_disp_D,
		   int*           &N_in_PE_sigma,
		   int*           &N_disp_sigma,
		   int*           &N_in_PE_SS,
		   int*           &N_disp_SS )
{
  static const int SPACE_DIM  = 3;
  static const int MM         = SPACE_DIM*((SPACE_DIM - 1)/2 + 1);
 
  for( int ip=0; ip<N_in_PE[rank]; ip++ ){
    for( int dim1=0; dim1<SPACE_DIM; dim1++ ){
      for( int dim2=0; dim2<SPACE_DIM; dim2++ ){
	D_local_1D[ ip*SPACE_DIM*SPACE_DIM+dim1*SPACE_DIM+dim2 ] 
	  = D[ip][dim1][dim2];
      }
    }
    for( int mm=0; mm<MM; mm++ ){
      sigma_local_1D[ ip*MM+mm ] = sigma[ip][mm];
      SS_local_1D[ ip*MM+mm ]    = SS[ip][mm];
    }
  }
  MPI_Gatherv( &D_local_1D[0], N_in_PE_D[rank],
	       MPI_DOUBLE, &D_all_1D[0], &N_in_PE_D[rank], &N_disp_D[0],
	       MPI_DOUBLE, 0, MCW );
  MPI_Gatherv( &sigma_local_1D[0], N_in_PE_sigma[rank],
	       MPI_DOUBLE, &sigma_all_1D[0], &N_in_PE_sigma[rank], &N_disp_sigma[0],
	       MPI_DOUBLE, 0, MCW );
  MPI_Gatherv( &SS_local_1D[0], N_in_PE_SS[rank],
	       MPI_DOUBLE, &SS_all_1D[0], &N_in_PE_SS[rank], &N_disp_SS[0],
	       MPI_DOUBLE, 0, MCW );
  
  return(0);
}
//-----------------------------------------------------------------------
int Substitution( IOParameterSet &param,
		  const int      &step,
		  double**       &sigma_all,
		  double**       &SS_all,
		  double***      &D_all,
		  double*        &sigma_all_1D,
		  double*        &SS_all_1D,
		  double*        &D_all_1D )
{
  static const int N_particle = param.int_value("N_particle");
  static const int SPACE_DIM  = 3;
  static const int MM         = SPACE_DIM*( (SPACE_DIM-1)/2 + 1 );
  static const int MAXSTEP    = param.value("MAXSTEP");
  static const int Equilibration_step  = param.int_value("Equilibration_step");
  static const int Data_step           = param.int_value("Data_step");
  static const double ave_data         = param.value("ave_data");
  static const int Analysis_start_step = (int)( Equilibration_step+ave_data*Data_step+0.1);
  static bool first_flag = true;
  static double **sigma_temp;
  static int counter = 0;
  
  if(first_flag == true){
    ALLOC_2D( sigma_temp, N_particle, MM );
    for(int i=0;i<N_particle;i++){
      for(int j=0;j<MM;j++){
	sigma_temp[i][j] = 0.0;
      }
    }
    first_flag = false;
  }
  
  for( int ip=0; ip<N_particle; ip++ ){
    for(int a=0;a<SPACE_DIM;a++){
      for(int b=0;b<SPACE_DIM;b++){
	D_all[ip][a][b] = D_all_1D[ip*SPACE_DIM*SPACE_DIM+a*SPACE_DIM+b];
      }
    }
    for( int mm=0; mm<MM; mm++ ){
      sigma_all[ip][mm] = sigma_all_1D[ ip*MM+mm ];
      SS_all   [ip][mm] = SS_all_1D   [ ip*MM+mm ];
    }
  }

  if(step >= Analysis_start_step){
    counter++;
    for(int ip=0;ip<N_particle;ip++){
      for(int mm=0;mm<MM;mm++){
	sigma_temp[ip][mm] += sigma_all[ip][mm];
      }
    }
  }
  if(step == (MAXSTEP-1)){
    double inv_step = 1/double(counter);
    cout<<"inv_step == "<<inv_step<<endl;
    for(int ip=0;ip<N_particle;ip++){
      for(int mm=0;mm<MM;mm++){
	sigma_temp[ip][mm] *= inv_step;
      }
    }
    char filename[64];
    FILE *fp;

    sprintf(filename, "./sigma_and_D.dat");
    fp = fopen(filename,"w");
    fprintf( fp, 
	     "#--------------------------------------------------------"
	     "--------------------------\n" );
    fprintf( fp, 
	     "#  (1)D_xx,  (2)D_xy,  (3)D_xz,  "
	     "(4)D_yx,  (5)D_yy,  (6)D_yz,  "
	     "(7)D_zx,  (8)D_zy,  (9)D_zz,"
	     "(10)sigmaxx,  (11)sigmayy,  (12)sigmazz,  "
	     "(13)sigmaxy,  (14)sigmayz,  (15)sigmazx,  "
	     "\n" );
    fprintf( fp,
	     "#----------------------------------------------------"
	     "------------------------------\n\n" );
  
    for( int ip=0; ip<N_particle; ip++ ){
      fprintf( fp, "%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t%e\t"
	       "%e\t%e\t%e\t%e\t%e\t%e\n",
	       D_all[ip][0][0],
	       D_all[ip][0][1],
	       D_all[ip][0][2],
	       D_all[ip][1][0],
	       D_all[ip][1][1],
	       D_all[ip][1][2],
	       D_all[ip][2][0],
	       D_all[ip][2][1],
	       D_all[ip][2][2],
	       sigma_temp[ip][0],
	       sigma_temp[ip][1],
	       sigma_temp[ip][2],
	       sigma_temp[ip][3],
	       sigma_temp[ip][4],
	       sigma_temp[ip][5]);
    }
    fclose(fp);
    
  }
  
  return(0);
}

//-----------------------------------------------------------------------
