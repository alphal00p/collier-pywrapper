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

    return


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
    
    return


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


#######################################
############## UTILITIES ##############
#######################################

def mink_square(v):
    return v[0]**2-v[1]**2-v[2]**2-v[3]**2


##########################################
############## GET IR POLES ##############
##########################################

def get_tensor(Momvec, MomInv, mass2, N, R, NN):


    py_wrapper_SetMuIR2_cll(1)

    Nt=py_wrapper_GetNt_cll(R)
    #py_wrapper_SetDeltaUV_cll(1)
    py_wrapper_SetDeltaIR_cll(0,0)

    TNten1 = np.array([0+0j] * (Nt), dtype=np.complex128)
    TNtenuv1 = np.array([0+0j] * (Nt), dtype=np.complex128)
    TNtenerr1 = np.array([0] * (Nt), dtype=np.double)

    py_wrapper_TNten_cll(TNten1, TNtenuv1, Momvec, MomInv, mass2, N, R, TNtenerr1, NN)

    py_wrapper_SetDeltaIR_cll(1,0)

    TNten2 = np.array([0+0j] * (Nt), dtype=np.complex128)
    TNtenuv2 = np.array([0+0j] * (Nt), dtype=np.complex128)
    TNtenerr2 = np.array([0] * (Nt), dtype=np.double)

    py_wrapper_TNten_cll(TNten2, TNtenuv2, Momvec, MomInv, mass2, N, R, TNtenerr2, NN)
    
    singlep=TNten2-TNten1


    py_wrapper_SetDeltaIR_cll(0,1)
    TNten3 = np.array([0+0j] * (Nt), dtype=np.complex128)
    TNtenuv3 = np.array([0+0j] * (Nt), dtype=np.complex128)
    TNtenerr3 = np.array([0] * (Nt), dtype=np.double)

    py_wrapper_TNten_cll(TNten3, TNtenuv3, Momvec, MomInv, mass2, N, R, TNtenerr3, NN)

    doublep=TNten3-TNten1

    result={"finite": TNten1, 
            "epsIR^(-1)": singlep, 
            "epsIR^(-2)": doublep,
            "epsUV^{-1}": TNtenuv1}
    
    return result

def get_tensor_coefficients(MomInv, mass2, N, R, NN):

    py_wrapper_SetMuIR2_cll(1)

    Nc=py_wrapper_GetNc_cll(N,R)
    #py_wrapper_SetDeltaUV_cll(1)
    py_wrapper_SetDeltaIR_cll(0,0)

    TN1 = np.array([0+0j] * (Nc), dtype=np.complex128)
    TNuv1 = np.array([0+0j] * (Nc), dtype=np.complex128)
    TNerr1 = np.array([0] * (Nc), dtype=np.double)

    py_wrapper_TN_cll(TN1, TNuv1, MomInv, mass2, N, R, TNerr1, NN)

    py_wrapper_SetDeltaIR_cll(1,0)

    TN2 = np.array([0+0j] * (Nc), dtype=np.complex128)
    TNuv2 = np.array([0+0j] * (Nc), dtype=np.complex128)
    TNerr2 = np.array([0] * (Nc), dtype=np.double)

    py_wrapper_TN_cll(TN2, TNuv2, MomInv, mass2, N, R, TNerr2, NN)
    
    singlep=TN2-TN1


    py_wrapper_SetDeltaIR_cll(0,1)
    TN3 = np.array([0+0j] * (Nc), dtype=np.complex128)
    TNuv3 = np.array([0+0j] * (Nc), dtype=np.complex128)
    TNerr3 = np.array([0] * (Nc), dtype=np.double)

    py_wrapper_TN_cll(TN3, TNuv3, MomInv, mass2, N, R, TNerr3, NN)

    doublep=TN3-TN1

    result={"finite": TN1, 
            "epsIR^(-1)": singlep, 
            "epsIR^(-2)": doublep,
            "epsUV^{-1}": TNuv1}
    
    return result



###################################
############## TESTS ##############
###################################

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Parse one string input from the command line.")
    parser.add_argument("input_string", nargs="?", default=None, help="add one of the following arguments (default=None): ")
    args = parser.parse_args()

    if args.input_string is not None:
        if args.input_string=="test_scalar_one_mass_triangle":

            N=3
            R=0
            py_wrapper_init(N,R,"scalar_one_mass_triangle")
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

            py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, N, R, TNtenerr, N)

            print("---- DeltaIR1 = 0, DeltaIR2 = 0 ----")
            print("results from collier")
            print(TNten)
            print(TNtenuv)
            print("my benchmark")
            print(-1.4047075598891257241388639034827-1.0887930451518010652503444491188j)
            print(0)

            deltauv = 0
            deltair1 = 1
            deltair2 = 0

            py_wrapper_SetDeltaIR_cll(deltair1,deltair2)

            py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, N, R, TNtenerr, N)

            print("---- DeltaIR1 = 1, DeltaIR2 = 0 ----")
            print("results from collier")
            print(TNten)
            print(TNtenuv)

            print("my benchmark")
            print(-1.7512811501690983788474799642118-0.3033948817543527556346836032989j)
            print(0)

            

            deltauv = 0
            deltair1 = 0
            deltair2 = 1

            py_wrapper_SetDeltaIR_cll(deltair1,deltair2)

            py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, N, R, TNtenerr, N)

            print("---- DeltaIR1 = 0, DeltaIR2 = 1 ----")
            print("results from collier")
            print(TNten)
            print(TNtenuv)
            print("my benchmark")
            print(-1.1547075598891257241388639034827-1.0887930451518010652503444491188j)
            print(0)

            resultwithpoles=get_tensor(Momvec, MomInv, mass2, N, R, N)
            print(resultwithpoles)


        if args.input_string=="test_rank3_one_mass_bubble":

            N=2
            R=1
            py_wrapper_init(N,R,"rank3_one_mass_bubble")
            Nt=py_wrapper_GetNt_cll(R)

            TNten = np.array([0+0j] * (Nt), dtype=np.complex128)
            TNtenuv = np.array([0+0j] * (Nt), dtype=np.complex128)
            TNtenerr = np.array([0] * (Nt), dtype=np.double)

            p=np.array([1+0j,0+0j,0+0j,0+0j], dtype=np.complex128)

            #Momvec = np.concatenate((p1,p2,q),axis=None)

            Momvec = np.concatenate((p),axis=None)

            MomInv = np.array([mink_square(p)], dtype=np.complex128)

            mass2=np.array([0+0j,0+0j], dtype=np.complex128)

            py_wrapper_SetMuIR2_cll(1)

            py_wrapper_SetDeltaIR_cll(0,0)

            py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, N, R, TNtenerr, N)

            print("---- DeltaIR1 = 0, DeltaIR2 = 0 ----")
            print(TNten[0])
            print(TNten)
            print("----")
            print(get_tensor(Momvec, MomInv, mass2, N, R, N))



        if args.input_string=="test_rank3_one_mass_triangle":

            N=3
            R=2
            py_wrapper_init(N,R,"rank3_one_mass_triangle")
            Nt=py_wrapper_GetNt_cll(R)
            Nc=py_wrapper_GetNc_cll(N,R)

            TNten = np.array([0+0j] * (Nt), dtype=np.complex128)
            TNtenuv = np.array([0+0j] * (Nt), dtype=np.complex128)
            TNtenerr = np.array([0] * (Nt), dtype=np.double)

            TN = np.array([0+0j] * (Nc), dtype=np.complex128)
            TNuv = np.array([0+0j] * (Nc), dtype=np.complex128)

            p1=np.array([1+0j,0+0j,0+0j,1+0j], dtype=np.complex128)
            p2=np.array([1+0j,0+0j,0+0j,-1+0j], dtype=np.complex128)
            #p2=np.array([-2+0j,0+0j,0+0j,0+0j], dtype=np.complex128)
            zero=np.array([0+0j,0+0j,0+0j,0+0j], dtype=np.complex128)
            q=-p1-p2

            #Momvec = np.concatenate((p1,p2,q),axis=None)

            Momvec = np.concatenate((p1,-p2),axis=None)

            MomInv = np.array([mink_square(p1),mink_square(q),mink_square(p2)], dtype=np.complex128)

            mass2=np.array([0+0j,0+0j,0+0j], dtype=np.complex128)

            py_wrapper_SetMuIR2_cll(1)

            py_wrapper_SetDeltaIR_cll(0,0)

            py_wrapper_TNten_cll(TNten, TNtenuv, Momvec, MomInv, mass2, N, R, TNtenerr, N)

            py_wrapper_TN_cll(TN, TNuv, MomInv, mass2, N, R, TNtenerr, N)



            print("---- DeltaIR1 = 0, DeltaIR2 = 0 ----")
            print("---- rank 0 ----")
            print(TNten[0])
            print("---- rank 1 ----")
            print(TNten[1:5])
            print("---- rank 2 ----")
            print(TNten[5:16])
            print(get_tensor(Momvec, MomInv, mass2, N, R, N))
            print(get_tensor_coefficients(MomInv, mass2, N, R, N))


            

           