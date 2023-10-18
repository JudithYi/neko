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

double dataTime = 0.0;
std::clock_t startT;
std::clock_t startTotal;
int rank, size, i;
bool if3d;
bool firstStep;

void convert_points_connectivity(
    const double *x_in,
    const double *y_in,
    const double *z_in,
    float *points_out,
    int *connectivity_out,
    const int lx1_in,
    const int ly1_in,
    const int lz1_in,
    const int nelt_in,
    const int nelbt_in,
    const bool if3d_in)
{
    /**
     * @param x_in, y_in, z_in the position of points in x, y, z directions
     * @param lx_in, ly_in, lz_in the number of points in one element in x, y, z  directions
     * @param nelt_in the number of elements assigned to this rank
     * @param nelbt_in the index of the first elements assigned to this rank
     * @param if3d_in the bool indicate if the simulation case is 3d
     * @param points_out all the positions of the points
     * @param connectivity_out the index of points each cell connected to
     */
    int idx, ii, jj, kk;
    int points_offset, start_index, index;
    int lxyz1_ = lx1_in * ly1_in * lz1_in;
    int lxy1_ = lx1_in * ly1_in;
    int lx3_ = lx1_in - 1;
    int ly3_ = ly1_in - 1;
    int lz3_ = lz1_in - 1;
    if (!if3d_in)
        lz3_ = lz1_in;
    int lyz3_ = ly3_ * lz3_;
    int lxyz3_ = lx3_ * lyz3_;
    if (if3d_in)
    {
        for (idx = 0; idx < nelt_in; ++idx)
        {
            /* Global point index offset would be:
                points_offset = lxyz1_ * (idx + nelbt_in);
            */
            points_offset = lxyz1_ * idx;
            start_index = lxyz3_ * idx * 8;
            for (ii = 0; ii < lx3_; ++ii)
            {
                for (jj = 0; jj < ly3_; ++jj)
                {
                    for (kk = 0; kk < lz3_; ++kk)
                    {
                        index = start_index + (ii * lyz3_ + jj * lz3_ + kk) * 8;
                        connectivity_out[index] = kk * lxy1_ + jj * lx1_in + ii + points_offset;
                        connectivity_out[index + 1] = kk * lxy1_ + jj * lx1_in + (ii + 1) + points_offset;
                        connectivity_out[index + 2] = kk * lxy1_ + (jj + 1) * lx1_in + (ii + 1) + points_offset;
                        connectivity_out[index + 3] = kk * lxy1_ + (jj + 1) * lx1_in + ii + points_offset;
                        connectivity_out[index + 4] = (kk + 1) * lxy1_ + jj * lx1_in + ii + points_offset;
                        connectivity_out[index + 5] = (kk + 1) * lxy1_ + jj * lx1_in + (ii + 1) + points_offset;
                        connectivity_out[index + 6] = (kk + 1) * lxy1_ + (jj + 1) * lx1_in + (ii + 1) + points_offset;
                        connectivity_out[index + 7] = (kk + 1) * lxy1_ + (jj + 1) * lx1_in + ii + points_offset;
                    }
                }
            }
        }
    }
    else
    {
        for (idx = 0; idx < nelt_in; ++idx)
        {
            points_offset = lxyz1_ * (idx + nelbt_in);
            start_index = lxyz3_ * idx * 4;
            for (ii = 0; ii < lx3_; ++ii)
            {
                for (jj = 0; jj < ly3_; ++jj)
                {
                    index = start_index + (ii * ly3_ + jj) * 4;
                    connectivity_out[index] = jj * lx1_in + ii + points_offset;
                    connectivity_out[index + 1] = jj * lx1_in + (ii + 1) + points_offset;
                    connectivity_out[index + 2] = (jj + 1) * lx1_in + (ii + 1) + points_offset;
                    connectivity_out[index + 3] = (jj + 1) * lx1_in + ii + points_offset;
                }
            }
        }
    }
    for (idx = 0; idx < nelt_in * lxyz1_; ++idx)
    {
        points_out[idx * 3] = static_cast<float>(x_in[idx]);
        points_out[idx * 3 + 1] = static_cast<float>(y_in[idx]);
        points_out[idx * 3 + 2] = static_cast<float>(z_in[idx]);
    }
}

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
    io = adios.DeclareIO("writer");
    /*
     *  Fides schema
     */
    /* VTK connectivity list is always a 1D array, a contiguous enumeration of all points  */
    io.DefineAttribute<std::string>("Fides_Data_Model", "unstructured_single");

    io.DefineAttribute<std::string>("Fides_Cell_Type", "quad");
    ADIOS_connectivity = io.DefineVariable<int>("connectivity", {total3 * 4}, {start3 * 4}, {count3 * 4});
    connectivity.resize(count3 * 4);

    /* VTK points is one table with the coordinates as columns OR separate 1D variables for each coordinate */
    ADIOS_points = io.DefineVariable<float>("points", {total1, 3}, {start1, 0}, {count1, 3});
    points.resize(count1 * 3);
    convert_points_connectivity(x_in, y_in, z_in, points.data(), connectivity.data(), lx1, ly1, lz1, nelv, nelbv, true);

    io.DefineAttribute<std::string>("Fides_Coordinates_Variable", "points");
    io.DefineAttribute<std::string>("Fides_Connectivity_Variable", "connectivity");
    // io.DefineAttribute<std::string>("Fides_Time_Variable", "step");

    std::vector<std::string> varList = {"P", "T", "VX", "VY", "VZ"};
    std::vector<std::string> assocList = {"points", "points", "points", "points", "points"};
    io.DefineAttribute<std::string>("Fides_Variable_List", varList.data(), varList.size());
    io.DefineAttribute<std::string>("Fides_Variable_Associations", assocList.data(), assocList.size());
    step = 0;

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
    
    //writer0.BeginStep();
    writer.BeginStep();
    
    init_count = 12;
    init_total = init_count*static_cast<std::size_t>(size);
    init_start = init_count*static_cast<std::size_t>(rank);
    vINT.resize(init_count);
    vINT[0] = lx1;
    vINT[1] = ly1;
    vINT[2] = lz1;
    vINT[3] = lx2;
    vINT[4] = ly2;
    vINT[5] = lz2;
    vINT[6] = nelv;
    vINT[7] = nelt;
    vINT[8] = nelgv;
    vINT[9] = nelgt;
    vINT[10] = *iostep_in;
    vINT[11] = 1;
    init_int_const = io.DefineVariable<int>("INT_CONST", {init_total}, {init_start}, {init_count});

    init_count = 2;
    init_total = init_count*static_cast<std::size_t>(size);
    init_start = init_count*static_cast<std::size_t>(rank);
    vDOUBLE.resize(init_count);
    vDOUBLE[0] = *t_start_in;
    vDOUBLE[1] = *dt_in;
    init_double_const = io.DefineVariable<double>("DOUBLE_CONST", {init_total}, {init_start}, {init_count});

    total1 = static_cast<std::size_t>(lxyz1 * nelgv);
    start1 = static_cast<std::size_t>(lxyz1 * nelbv);
    count1 = static_cast<std::size_t>(lxyz1 * nelv);
    start3 = static_cast<std::size_t>(lxyz1 * nelbt);

    vx = io.DefineVariable<double>("VX", {total1}, {start1}, {count1});
    vy = io.DefineVariable<double>("VY", {total1}, {start1}, {count1});
    vz = io.DefineVariable<double>("VZ", {total1}, {start1}, {count1});
    p = io.DefineVariable<double>("P", {total1}, {start1}, {count1});
    t = io.DefineVariable<double>("T", {total1}, {start1}, {count1});
    vstep = io.DefineVariable<int>("step");
    writer.Put<int>(ADIOS_connectivity, connectivity.data());
    writer.Put<float>(ADIOS_points, points.data());
    writer.Put<int>(init_int_const, vINT.data());
    writer.Put<double>(init_double_const, vDOUBLE.data());

    writer.EndStep();


    /* End of Fides schema */
    writer.BeginStep();
    /*
    But vtk related information such as points and connectivity are communicated in this step
    Also the initail values of p (pressure), vx (velocity in x direction), vy (velocity in y direction), vz (velocity in z direction), and t (temperature) are communicated.
    */
    writer.Put<double>(vx, v);
    writer.Put<double>(vy, u);
    writer.Put<double>(vz, w);
    writer.Put<double>(p, pr);
    writer.Put<double>(t, t_in);
    writer.Put<int>(vstep, step);
    writer.EndStep();

    firstStep=true;
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



