import ctypes
from ctypes import c_int, c_char_p, POINTER, byref, c_double
import numpy as np
import math
from numpy.ctypeslib import ndpointer
import argparse

# Load the shared library (adjust the path/name as needed)
lib = ctypes.CDLL("./cpp_wrapper.so")

##############################################
############## WRAPPER FOR init ##############
##############################################

lib.wrapper_init.argtypes = [POINTER(c_int), POINTER(c_int), c_char_p]
lib.wrapper_init.restype = None

def py_wrapper_init(Nmax, rmax, output_file):

    Nmax_c = c_int(Nmax)
    rmax_c = c_int(rmax)

    output_file_c = output_file.encode('utf-8')

    lib.wrapper_init(byref(Nmax_c), byref(rmax_c), output_file_c)

###############################################
############## WRAPPER FOR GetNc ##############
###############################################

lib.wrapper_GetNc_cll.argtypes = [POINTER(c_int), POINTER(c_int)]
lib.wrapper_GetNc_cll.restype = c_int

def py_wrapper_GetNc_cll(Nmax, rmax):

    # Convert Python ints to c_int
    Nmax_c = c_int(Nmax)
    rmax_c = c_int(rmax)
    
    # Call the function
    return lib.wrapper_GetNc_cll(byref(Nmax_c), byref(rmax_c))

################################################
############## WRAPPER FOR TN_cll ##############
################################################

lib.wrapper_TN_cll.argtypes = [
    ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"),  # TN
    ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"),  # TNuv
    ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"),  # MomInv
    ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"),  # mass2
    ctypes.POINTER(ctypes.c_int),                          # Nn
    ctypes.POINTER(ctypes.c_int),                          # R
    ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),      # TNerr
    ctypes.POINTER(ctypes.c_int)                           # N
]
lib.wrapper_TN_cll.restype = None 

def py_wrapper_TN_cll(TN, TNuv, MomInv, mass2, Nn, R, TNerr, N):

    Nn_c = ctypes.c_int(Nn)
    R_c  = ctypes.c_int(R)
    N_c  = ctypes.c_int(N)
    
    lib.wrapper_TN_cll(TN, TNuv, MomInv, mass2,
                        ctypes.byref(Nn_c), ctypes.byref(R_c),
                        TNerr, ctypes.byref(N_c))
    
    return TN, TNuv


###############################################
############## WRAPPER FOR GetNt ##############
###############################################

lib.wrapper_GetNt_cll.argtypes = [POINTER(c_int)]
lib.wrapper_GetNc_cll.restype = c_int

def py_wrapper_GetNt_cll(rmax):

    # Convert Python ints to c_int
    rmax_c = c_int(rmax)
    
    # Call the function
    return lib.wrapper_GetNt_cll(byref(rmax_c))


###############################################
############## WRAPPER FOR TNten ##############
###############################################

lib.wrapper_TNten_cll.argtypes = [
    ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"),  # TNten
    ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"),  # TNtenuv
    ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"),  # MomVec
    ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"),  # MomInv
    ndpointer(dtype=np.complex128, flags="C_CONTIGUOUS"),  # mass2
    ctypes.POINTER(ctypes.c_int),                          # Nn
    ctypes.POINTER(ctypes.c_int),                          # R
    ndpointer(dtype=np.double, flags="C_CONTIGUOUS"),      # TNtenerr
    ctypes.POINTER(ctypes.c_int)                           # N
]
lib.wrapper_TNten_cll.restype = None 

def py_wrapper_TNten_cll(TNten, TNtenuv, MomVec, MomInv, mass2, Nn, R, TNtenerr, N):

    Nn_c = ctypes.c_int(Nn)
    R_c  = ctypes.c_int(R)
    N_c  = ctypes.c_int(N)
    
    lib.wrapper_TNten_cll(TNten, TNtenuv, MomVec, MomInv, mass2,
                        ctypes.byref(Nn_c), ctypes.byref(R_c),
                        TNtenerr, ctypes.byref(N_c))
    
    return TNten, TNtenuv


#######################################################
############## WRAPPERS FOR IR FUNCTIONS ##############
#######################################################

lib.wrapper_GetDeltaIR_cll.argtypes = [POINTER(c_double),POINTER(c_double)]
lib.wrapper_GetDeltaIR_cll.restype = None

def py_wrapper_GetDeltaIR_cll(delta1,delta2):
    delta1_c = c_double(delta1)
    delta2_c = c_double(delta2)
    return lib.wrapper_GetDeltaIR_cll(byref(delta1_c),byref(delta2_c))

lib.wrapper_GetMuIR2_cll.argtypes = [POINTER(c_double)]
lib.wrapper_GetMuIR2_cll.restype = None

def py_wrapper_GetMuIR2_cll(mu):
    mu_c = c_double(mu)
    return lib.wrapper_GetMuIR2_cll(byref(mu_c))

lib.wrapper_SetDeltaIR_cll.argtypes = [POINTER(c_double),POINTER(c_double)]
lib.wrapper_SetDeltaIR_cll.restype = None

def py_wrapper_SetDeltaIR_cll(delta1,delta2):
    delta1_c = c_double(delta1)
    delta2_c = c_double(delta2)
    return lib.wrapper_SetDeltaIR_cll(byref(delta1_c),byref(delta2_c))

lib.wrapper_SetMuIR2_cll.argtypes = [POINTER(c_double)]
lib.wrapper_SetMuIR2_cllrestype = None

def py_wrapper_SetMuIR2_cll(mu):
    mu_c = c_double(mu)
    return lib.wrapper_SetMuIR2_cll(byref(mu_c))


#######################################################
############## WRAPPERS FOR UV FUNCTIONS ##############
#######################################################

lib.wrapper_GetDeltaUV_cll.argtypes = [POINTER(c_double)]
lib.wrapper_GetDeltaUV_cll.restype = None

def py_wrapper_GetDeltaUV_cll(delta):
    delta_c = c_double(delta)
    return lib.wrapper_GetDeltaUV_cll(byref(delta_c))

lib.wrapper_GetMuUV2_cll.argtypes = [POINTER(c_double)]
lib.wrapper_GetMuUV2_cll.restype = None

def py_wrapper_GetMuUV2_cll(delta):
    delta_c = c_double(delta)
    return lib.wrapper_GetMuUV2_cll(byref(delta_c))

lib.wrapper_SetDeltaUV_cll.argtypes = [POINTER(c_double)]
lib.wrapper_SetDeltaUV_cll.restype = None

def py_wrapper_SetDeltaUV_cll(delta):
    delta_c = c_double(delta)
    return lib.wrapper_SetDeltaUV_cll(byref(delta_c))

lib.wrapper_SetMuUV2_cll.argtypes = [POINTER(c_double)]
lib.wrapper_SetMuUV2_cll.restype = None

def py_wrapper_SetMuUV2_cll(delta):
    delta_c = c_double(delta)
    return lib.wrapper_SetMuUV2_cll(byref(delta_c))





def mink_square(v):
    return v[0]**2-v[1]**2-v[2]**2-v[3]**2

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Parse one string input from the command line.")
    parser.add_argument("input_string", nargs="?", default=None, help="add one of the following arguments (default=None): ")
    args = parser.parse_args()

    if args.input_string is not None:
        if args.input_string=="test_scalar_triangle":

            N=3
            R=0
            py_wrapper_init(N,R,"gg_to_H")
            Nt=py_wrapper_GetNt_cll(R)

            TNten = np.array([0+0j] * (Nt+1), dtype=np.complex128)
            TNtenuv = np.array([0+0j] * (Nt+1), dtype=np.complex128)
            TNtenerr = np.array([0] * (Nt+1), dtype=np.double)

            p1=np.array([1+0j,0+0j,0+0j,1+0j], dtype=np.complex128)
            p2=np.array([1+0j,0+0j,0+0j,-1+0j], dtype=np.complex128)
            q=-p1-p2

            Momvec = np.concatenate((p1,p2,q),axis=None)

            MomInv = np.array([mink_square(p1),mink_square(p2),mink_square(q),mink_square(p1+p2),mink_square(p2+q),mink_square(p1+q)], dtype=np.complex128)

            mass2=np.array([0+0j,0+0j,0+0j], dtype=np.complex128)

            py_wrapper_SetMuIR2_cll(1)

            py_wrapper_SetDeltaIR_cll(0,0)

            new_TNten, new_TNtenuv = py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, N, R, TNtenerr, N)

            print("---- DeltaIR1 = 0, DeltaIR2 = 0 ----")
            print("results from collier")
            print(new_TNten)
            print(new_TNtenuv)
            print("my benchmark")
            print(-1.4047075598891257241388639034827-1.0887930451518010652503444491188j)
            print(0)

            deltauv = 0
            deltair1 = 1
            deltair2 = 0

            py_wrapper_SetDeltaIR_cll(deltair1,deltair2)

            new_TNten, new_TNtenuv = py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, N, R, TNtenerr, N)

            print("---- DeltaIR1 = 1, DeltaIR2 = 0 ----")
            print("results from collier")
            print(new_TNten)
            print(new_TNtenuv)

            print("my benchmark")
            print(-1.7512811501690983788474799642118-0.3033948817543527556346836032989j)
            print(0)

            

            deltauv = 0
            deltair1 = 0
            deltair2 = 1

            py_wrapper_SetDeltaIR_cll(deltair1,deltair2)

            new_TNten, new_TNtenuv = py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, N, R, TNtenerr, N)

            print("---- DeltaIR1 = 0, DeltaIR2 = 1 ----")
            print("results from collier")
            print(new_TNten)
            print(new_TNtenuv)
            print("my benchmark")
            print(-1.1547075598891257241388639034827-1.0887930451518010652503444491188j)
            print(0)









