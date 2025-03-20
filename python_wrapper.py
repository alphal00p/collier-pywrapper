import ctypes
from ctypes import c_int, c_char_p, POINTER, byref
import numpy as np
import math
from numpy.ctypeslib import ndpointer

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

def mink_square(v):
    return v[0]**2-v[1]**2-v[2]**2-v[3]**2

if __name__ == "__main__":
    """ py_wrapper_init(2,2,"output_folder")
    print("Nc and Nt")
    print(py_wrapper_GetNc_cll(2,2))
    print(py_wrapper_GetNt_cll(2))

    size = 5  # Adjust as needed.
    
    # Create example NumPy arrays for the four complex inputs.
    TN     = np.array([1+0j, 1+0j, 1+0j, 1+0j], dtype=np.complex128)
    TNuv   = np.array([1+0j, 1+0j, 1+0j, 1+0j], dtype=np.complex128)
    MomInv = np.array([3+0j], dtype=np.complex128)
    mass2  = np.array([4+0j, 6+0j], dtype=np.complex128)
    
    # Create an example TNerr array (double precision).
    TNerr = np.array([0, 0], dtype=np.double)
    
    # Initial values for the integer parameters.
    Nn = 2
    R  = 2
    N  = 2

    new_TN, new_TNuv = py_wrapper_TN_cll(TN, TNuv, MomInv, mass2, Nn, R, TNerr, N)
    print("Updated parameters:")
    print("TN =", new_TN, "TNuv =", new_TNuv)

    print(py_wrapper_GetNt_cll(R))

    TNten     = np.array([1+0j, 1+0j, 1+0j, 1+0j,1+0j, 1+0j, 1+0j, 1+0j,1+0j, 1+0j, 2+0j, 1+0j, 1+0j, 1+0j, 1+0j], dtype=np.complex128)
    TNtenuv   = np.array([1+0j, 1+0j, 1+0j, 1+0j,1+0j, 1+0j, 1+0j, 1+0j,1+0j, 1+0j, 1+0j, 1+0j, 1+0j, 1+0j, 1+0j], dtype=np.complex128)
    TNtenerr = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.double)
    Momvec = np.array([math.sqrt(3+1+0.25)+0j,0,0.5+0j,1+0j], dtype=np.complex128)

    new_TNten, new_TNtenuv = py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, Nn, R, TNtenerr, N)
    print("Updated parameters:")
    print("TNten =", new_TNten)
    print("TNtenuv =", new_TNtenuv) """

    N=3
    R=3
    py_wrapper_init(N,R,"gg_to_H")
    Nt=py_wrapper_GetNt_cll(R)

    TNten = np.array([0+0j] * Nt, dtype=np.complex128)
    TNtenuv = np.array([0+0j] * Nt, dtype=np.complex128)
    TNtenerr = np.array([0] * Nt, dtype=np.double)

    p1=np.array([1+0j,0+0j,0+0j,1+0j], dtype=np.complex128)
    p2=np.array([1+0j,0+0j,0+0j,-1+0j], dtype=np.complex128)
    q=-p1-p2

    Momvec = np.concatenate((p1,p2,q),axis=None)

    MomInv = np.array([mink_square(p1-p2),mink_square(p2-q),mink_square(q-p1)], dtype=np.complex128)

    mass2=np.array([1/2+0j,1/2+0j,1/2+0j], dtype=np.complex128)

    new_TNten, new_TNtenuv = py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, N, R, TNtenerr, N)

    print(new_TNten)
    print(new_TNtenuv)





