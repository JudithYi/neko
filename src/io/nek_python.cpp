#include <mpi.h>
#include <string>
#include <iostream>
#include <ctime>
#include <vector>

/*Python*/
#include <Python.h>
#include <stdio.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <map>
/*Python*/

int lx1, ly1, lz1, lx2, ly2, lz2;
int lxyz1, lxyz2; 
int nelv;
int nelgv, nelgt;
int nelbv, nelbt;
double dataTime = 0.0;
std::clock_t startT,startTotal;
int rank, size;
double setupTime = 0.0;
double closeTime = 0.0;

/*Python*/
PyObject *pName, *pModule, *pDict, *pClass, *pArgs, *pInst;
PyObject *py_Func, *py_Args_x;
PyObject *py_currentStep, *py_vx;
PyObject *py_uncertOpts;
PyObject *py_sample_freq_val;
PyObject *py_max_acf_lag_val;
PyObject *py_output_freq_val;
PyObject *py_compute_uncert_estim;
PyObject *py_name;
PyObject *py_var_mu;
PyObject *py_comm_int;
PyObject *py_nelv;

std::vector<double>vVX;
int total_size, local_size, nval_local;
int currentStep;
npy_intp dims[1];
/*Python*/


extern "C" void python_setup_(
    const int *lx1_in,
    const int *ly1_in,
    const int *lz1_in,
    const int *nelv_in,
    const int *nelgv_in,
    const int *nelgt_in,
    const int *comm_int,
    const int *iostep_in
){
    startTotal = std::clock();
    MPI_Comm comm = MPI_Comm_f2c(*comm_int);
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
    total_size = nelgv * lxyz1;
    local_size = nelv * lxyz1;
    char* pPath;
    nval_local = lxyz1;
    vVX.resize(nelv);
    
    //setenv("PYTHONPATH",".",1);
    Py_Initialize ();
    pName = PyUnicode_FromString ("IMACF");
    if (!pName)
    {
	std::cout << "!pName" << std::endl;
        return;
    }
    pModule = PyImport_Import(pName);
    if (!pModule)
    {
            std::cout << "!pModule" << std::endl;
	    return;
    }
    
    pDict = PyModule_GetDict(pModule);

    if (!pDict)
    {
            std::cout << rank << "!pDict" << std::endl;
            return;
    }

    if(PyArray_API == NULL){import_array();}
    
    py_sample_freq_val = PyLong_FromLong(1);
    py_uncertOpts = PyDict_New();
    PyDict_SetItemString(py_uncertOpts, (char*)"sample_freq", py_sample_freq_val);

    py_max_acf_lag_val = PyLong_FromLong(50);
    PyDict_SetItemString(py_uncertOpts, (char*)"max_acf_lag", py_max_acf_lag_val);

    py_output_freq_val = PyLong_FromLong(50);
    PyDict_SetItemString(py_uncertOpts, (char*)"output_freq", py_output_freq_val);

    PyDict_SetItemString(py_uncertOpts, (char*)"compute_uncert_estim", Py_True);

    pArgs = PyTuple_New (3);
    PyTuple_SetItem (pArgs, 0, py_uncertOpts);
    py_comm_int = PyLong_FromLong(*comm_int);
    PyTuple_SetItem (pArgs, 1, py_comm_int);
    py_nelv = PyLong_FromLong(vVX.size());
    PyTuple_SetItem (pArgs, 2, py_nelv);

    
    pClass = PyDict_GetItemString (pDict, (char*)"IMACF");
    if (!pClass)
    {
            std::cout << "!pClass" << std::endl;
            return;
    }

    if(!rank) std::cout << "init IMACF" << std::endl;
    pInst = PyObject_CallObject(pClass, pArgs);
    if (!pInst)
    {
	    PyErr_Print();
            std::cout << "!pInst" << std::endl;
            return;
    }
    if(!rank) std::cout << "IMACF done" << std::endl;
    dims[0]=nelv;
    
    py_vx = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, vVX.data());
    py_name = PyUnicode_FromString((char*)"update_stats");
    if (!py_name)
    {
            PyErr_Print();
            std::cout << "!py_name" << std::endl;
            return;
    }

    currentStep = 0;
    /*Python end*/
    setupTime += (std::clock() - startT) / (double) CLOCKS_PER_SEC;
    if(!rank) std::cout << "setup done" << std::endl;
}

extern "C" void python_update_(
    const double *v,
    const double *u,
    const double *w,
    const double *pr,
    const double *t_in
){
    startT = std::clock();
    py_currentStep=PyLong_FromLong(currentStep);        
    double nval_double=static_cast<double>(nval_local);

    for(int i = 0; i < nelv; ++i){
        vVX[i] = v[i*nval_local] / nval_double;
	for(int j = 1; j < nval_local; ++j){
	   vVX[i] += v[i*nval_local+j] / nval_double;
	}
    }
    py_var_mu=PyObject_CallMethodObjArgs(pInst, py_name, py_currentStep, py_vx, NULL);
    if (!py_var_mu)
    {
            PyErr_Print();
            std::cout << "!py_var_mu" << std::endl;
            return;
    }

    ++currentStep;
    dataTime += (std::clock() - startT) / (double) CLOCKS_PER_SEC;
}

extern "C" void python_finalize_(){
    startT = std::clock();
    /*Python start*/
    Py_DECREF(pName);                            
    Py_DECREF(pModule);
    Py_DECREF(pDict);
    Py_DECREF(pClass);
    Py_DECREF(pInst);
    Py_DECREF(pArgs);
    Py_DECREF(py_uncertOpts);
    Py_DECREF(py_sample_freq_val);
    Py_DECREF(py_max_acf_lag_val);
    Py_DECREF(py_output_freq_val);
    Py_DECREF(py_currentStep);
    Py_DECREF(py_vx);
    Py_DECREF(py_name);
    Py_DECREF(py_var_mu);

    /* PyObject checklist
     * PyObject *pName, *pModule, *pDict, *pFunc, *pArgs;
     * PyObject *py_totalSize, *py_comm, *py_currentStep;
     * PyObject *py_Func, *py_Args_x;
     * PyObject *py_currentsnap_x, *py_U_1t_x, *py_D_1t_x, *py_V_1t_x;
    */
    Py_Finalize ();   
    /*Python end*/

    closeTime += (std::clock() - startT) / (double) CLOCKS_PER_SEC;
    std::cout <<  "rank: " << rank << " sin-situ time: " << dataTime << "s, total time: " << (std::clock() - startTotal) / (double) CLOCKS_PER_SEC << "s. " << std::endl;
}
