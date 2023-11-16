#include <adios2.h>
#include <mpi.h>
#include <string>
#include <iostream>
#include <ctime>
#include <vector>

int lx1, ly1, lz1, lx2, ly2, lz2;
int lxyz1, lxyz2; 
int nelv;
int nelgv, nelgt;
int nelbv, nelbt;

std::size_t total1, start1, count1;
std::size_t total2, start2, count2;
std::size_t total3, start3, count3;
std::size_t init_total, init_start, init_count;
adios2::ADIOS adios;
adios2::IO io;
adios2::Engine writer;
adios2::Variable<int> init_int_const;
adios2::Variable<double> init_double_const;
std::vector<int> vINT;
std::vector<double> vDOUBLE;
adios2::Variable<double> p;
adios2::Variable<double> vx;
adios2::Variable<double> vy;
adios2::Variable<double> vz;
adios2::Variable<double> t;

adios2::Variable<int> vstep;
adios2::Variable<int> ADIOS_connectivity;
adios2::Variable<float> ADIOS_points;
std::vector<int> connectivity;
std::vector<float> points;
int step; 

/* DATA STREAMER */
int ifile;
int ifilew;
int ifstream;
int decide_stream_global;
adios2::IO io_head;
adios2::IO io_asynchronous;
adios2::Engine writer_head;
adios2::Engine writer_st;
adios2::Variable<double> bm1;
adios2::Variable<int> lglelw;
adios2::Variable<double> p_st;
adios2::Variable<double> vx_st;
adios2::Variable<double> vy_st;
adios2::Variable<double> vz_st;
adios2::Variable<double> bm1_st;
adios2::Variable<int> lglelw_st;
adios2::Variable<double> x;
adios2::Variable<double> y;
adios2::Variable<double> z;
/* DATA STREAMER */

double dataTime = 0.0;
std::clock_t startT;
std::clock_t startTotal;
int rank, size, i;
bool if3d;
bool firstStep;

extern "C" void adios2_setup_(
    const int *lx1_in,
    const int *ly1_in,
    const int *lz1_in,
    const int *nelv_in,
    const int *nelgv_in,
    const int *nelgt_in,
    const double *v,
    const double *u,
    const double *w,
    const double *pr,
    const int *comm_int,
    const double *x_in,
    const double *y_in,
    const double *z_in,
    const double *t_in,
    const double *t_start_in,
    const double *dt_in,
    const int *iostep_in
){
    startTotal = std::clock();
    std::string configFile="config/config.xml";
    MPI_Comm comm = MPI_Comm_f2c(*comm_int);
    adios = adios2::ADIOS(configFile, comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    lx1 = *lx1_in;
    ly1 = *ly1_in;
    lz1 = *lz1_in;

    lx2 = *lx1_in;
    ly2 = *ly1_in;
    lz2 = *lz1_in;

    lxyz1 = lx1 * ly1 * lz1;

    nelgt = *nelgt_in;
    nelgv = *nelgv_in;
    nelv = *nelv_in;
    unsigned int nelt = static_cast<unsigned int>((*nelv_in));
    std::vector<int>temp(size);
    std::vector<int>temp1(size);
    MPI_Allgather(&nelv, 1, MPI_INT, temp.data(), 1,MPI_INT, comm);
    MPI_Allgather(&nelt, 1, MPI_INT, temp1.data(), 1,MPI_INT, comm);
    nelbt = 0;
    nelbv = 0;
    for(i=0;i<rank;++i){
        nelbv += temp[i];
        nelbt += temp1[i];
    }
    total1 = static_cast<std::size_t>(lxyz1 * nelgv);
    start1 = static_cast<std::size_t>(lxyz1 * nelbv);
    count1 = static_cast<std::size_t>(lxyz1 * nelv);
    total3 = static_cast<std::size_t>(lxyz1 * nelgv);
    start3 = static_cast<std::size_t>(lxyz1 * nelbv);
    count3 = static_cast<std::size_t>(lxyz1 * nelv);
    io = adios.DeclareIO("writer");
    writer = io.Open("globalArray", adios2::Mode::Write);
    if (!io.InConfigFile())
    {
        // if not defined by user, we can change the default settings
        // BPFile is the default engine
        io.SetEngine("BPFile");
        io.SetParameters({{"num_threads", "1"}});

        // ISO-POSIX file output is the default transport (called "File")
        // Passing parameters to the transport
    }
    
    vINT_CONST[0]=lx1;    
    vINT_CONST[1]=ly1;    
    vINT_CONST[2]=lz1;    
    vINT_CONST[3]=lx2;    
    vINT_CONST[4]=ly2;    
    vINT_CONST[5]=lz2; 
    vINT_CONST[6]=nelv;    
    vINT_CONST[7]=nelt;    
    std::vector<int>temp(size);
    MPI_Allgather(&nelv, 1, MPI_INT, temp.data(), 1, MPI_INT, comm);
    nelbv=0;
    for(i=0;i<rank;++i){
        nelbv+=temp[i];
    }
    MPI_Allgather(&nelt, 1, MPI_INT, temp.data(), 1, MPI_INT, comm);
    nelbt=0;
    for(i=0;i<rank;++i){
        nelbt+=temp[i];
    }
    std::size_t total, start, count;
    count = static_cast<std::size_t>(8);
    total = count * static_cast<std::size_t>(size);
    start = count * static_cast<std::size_t>(rank);
    init_int_vec1 = io.DefineVariable<int>("INT_CONST", {total}, {start}, {count});
    
    /*adios variables definition*/
    count = static_cast<std::size_t>(lx1*ly1*lz1*nelv);
    start = static_cast<std::size_t>(lx1*ly1*lz1*nelbv);
    total = static_cast<std::size_t>(lx1*ly1*lz1*nelgv);
    vx = io.DefineVariable<double>("VX",{total}, {start}, {count});
    vy = io.DefineVariable<double>("VY",{total}, {start}, {count});
    vz = io.DefineVariable<double>("VZ",{total}, {start}, {count});
    p = io.DefineVariable<double>("P", {total}, {start}, {count});
    t = io.DefineVariable<double>("T", {total}, {start}, {count});

    writer = io.Open("globalArray", adios2::Mode::Write);
    writer.BeginStep();
    writer.Put<int>(init_int_vec1, vINT_CONST.data());
    writer.EndStep();
    if(!rank) std::cout << "In-Situ setting done" << std::endl;
    std::cout << "Nek rank: " << rank << " count: " << nelt << " , start: " << nelbt << " , total: " << nelgt << std::endl;
}

extern "C" void adios2_update_(
    const double *v,
    const double *u,
    const double *w,
    const double *pr,
    const double *t_in
){
    startT = std::clock();
    writer.BeginStep();

    writer.Put<double>(vx, v);
    writer.Put<double>(vy, u);
    writer.Put<double>(vz, w);
    writer.Put<double>(p, pr);
    writer.Put<double>(t, t_in);
    ++step;
    writer.Put<int>(vstep, step);
    writer.EndStep();
    dataTime += (std::clock() - startT) / (double) CLOCKS_PER_SEC;
}

extern "C" void adios2_finalize_(){
    writer.Close();
    std::cout <<  "rank: " << rank << " sin-situ time: " << dataTime << "s, total time: " << (std::clock() - startTotal) / (double) CLOCKS_PER_SEC << "s. " << std::endl;
}

extern "C" void adios2_stream_(
    const int *lglel,
    const double *pr,
    const double *u,
    const double *v,
    const double *w,
    const double *mass1,
    const double *temp
){
    startT = std::clock();
    // Begin a step of the writer
    writer_st.BeginStep();
    writer_st.Put<double>(p_st, pr);
    writer_st.Put<double>(vx_st, u);
    writer_st.Put<double>(vy_st, v);
    writer_st.Put<double>(vz_st, w);
    writer_st.Put<double>(bm1_st, mass1);
    writer_st.Put<int>(lglelw_st, lglel);
    writer_st.EndStep();
    dataTime += (std::clock() - startT) / (double) CLOCKS_PER_SEC;
}

extern "C" void adios2_setup_data_streamer_(
    const int *nval,
    const int *nelvin,
    const int *nelb,
    const int *nelgv,
    const int *nelgt,
    const double *xml,
    const double *yml,
    const double *zml,
    const int *if_asynchronous,
    const int *comm_int
){
    std::string configFile="adios2_config/config.xml";
    MPI_Comm comm = MPI_Comm_f2c(*comm_int);
    adios = adios2::ADIOS(configFile, comm);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    // Compressor writer.
    io = adios.DeclareIO("writer");
    // Mesh writer.
    io_head = adios.DeclareIO("writer0");
    // Asynchronous writer.
    io_asynchronous = adios.DeclareIO("writerISMPI");

    // Determine if asyncrhonous operation will be needed for this set up
    unsigned int decide_stream = static_cast<unsigned int>((*if_asynchronous));
    decide_stream_global = decide_stream;
    
    // Number of elements in my rank.
    unsigned int nelv = static_cast<unsigned int>((*nelvin));

    // Determine where my rank writes in the global array according to number of element in previous ranks
    unsigned int start = static_cast<unsigned int>(*nelb);
    start *= static_cast<unsigned int>(*nval);

    // n is count, i.e number of entries in the array in my rank
    unsigned int n = static_cast<unsigned int> (*nval) * nelv;
    // gn is the total size of the arrays, not per io rank 
    unsigned int gn = static_cast<unsigned int>((*nelgv)*(*nval));
    std::cout << rank << ": " << gn << ", " << start << "," << n << std::endl;

    // Create the adios2 variables for writer that depend on the current start and n
    p = io.DefineVariable<double>("P_OUT", {gn}, {start}, {n});
    vx = io.DefineVariable<double>("VX_OUT", {gn}, {start}, {n});
    vy = io.DefineVariable<double>("VY_OUT", {gn}, {start}, {n});
    vz = io.DefineVariable<double>("VZ_OUT", {gn}, {start}, {n});
    bm1 = io.DefineVariable<double>("BM1_OUT", {gn}, {start}, {n});

    // Create the adios2 variables for writer0
    x = io_head.DefineVariable<double>("X", {gn}, {start}, {n});
    y = io_head.DefineVariable<double>("Y", {gn}, {start}, {n});
    z = io_head.DefineVariable<double>("Z", {gn}, {start}, {n});

    // If the process is asynchronous, define the relevant variables for writer_st
    if (decide_stream == 1){
	    p_st = io_asynchronous.DefineVariable<double>("P", {gn}, {start}, {n});
	    vx_st = io_asynchronous.DefineVariable<double>("VX", {gn}, {start}, {n});
	    vy_st = io_asynchronous.DefineVariable<double>("VY", {gn}, {start}, {n});
	    vz_st = io_asynchronous.DefineVariable<double>("VZ", {gn}, {start}, {n});
	    bm1_st = io_asynchronous.DefineVariable<double>("BM1", {gn}, {start}, {n});
    }


    // Do everything again for the global indices 
    nelv = static_cast<unsigned int>((*nelvin));
    start = static_cast<unsigned int>(*nelb);
    n = static_cast<unsigned int> (nelv);
    gn = static_cast<unsigned int>((*nelgv));
    // Define variable for compression writer
    lglelw = io.DefineVariable<int>("LGLEL_OUT", {gn}, {start}, {n});
    // Define variable for asyncrhonous writet
    if (decide_stream == 1){
    	lglelw_st = io_asynchronous.DefineVariable<int>("LGLEL", {gn}, {start}, {n});
    }

    // Write the mesh information only once (Currently commented out).
    //writer_head = io_head.Open("geo.bp", adios2::Mode::Write);
    //writer_head.Put<double>(x, xml);
    //writer_head.Put<double>(y, yml);
    //writer_head.Put<double>(z, yml);
    //writer_head.Close();
    //if(!rank)
	//std::cout << "geo.bp written" << std::endl;
    
    // If asyncrhonous execution, open the global array
    if (decide_stream == 1){
	std::cout << "create global array" << std::endl;
    	writer_st = io_asynchronous.Open("globalArray", adios2::Mode::Write);
    }

    // Initialize global variables for writing. This could be done in global definition    
    ifile = 0 ;
    ifilew = 0 ;
}

extern "C" void adios2_finalize_data_streamer_(){
    if (decide_stream_global == 1){
	std::cout << "Close global array" << std::endl;
    	writer_st.Close();
    	std::cout <<  "rank: " << rank << " in-situ time: " << dataTime << "s." << std::endl;
    }

}
