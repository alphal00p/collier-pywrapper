#include <iostream>
#include <stdio.h>
#include <complex.h>

extern "C" {
    void wrapper_init(int* Nmax, int* rmax, char* output_file);
    int wrapper_GetNc_cll(int* N, int* rank);
    void wrapper_TN_cll(const double _Complex* TN, const double _Complex* TNuv, const double _Complex* MomInv, const double _Complex* mass2, int* Nn, int* R, const double* TNerr, int*N);
    int wrapper_GetNt_cll(int* rank);
    void wrapper_TNten_cll(const double _Complex* TNten, const double _Complex* TNtenuv, const double _Complex* Momvec, const double _Complex* MomInv, const double _Complex* mass2, int* Nn, int* R, const double* TNtenerr, int*N);
}

int main() {
    int Nmax = 10;
    int rmax = 5;
    int N=2;
    int rank=2;
    

    
    //printf("Result: %f + %fi\n", creal(result), cimag(result));
    char output_file[] = "output_folder";
    wrapper_init(&Nmax, &rmax, output_file);

    int result=wrapper_GetNc_cll(&N,&rank);
    std::cout<<result<<std::endl;

    double _Complex TN [2]={1,1};
    double _Complex TNuv [2]={1,1};
    double _Complex MomInv [1]={};
    double _Complex mass2 [1]={2};

    int Nn=1;
    int R=2;
    double TNerr[2]={1,1};
    N=1;

    wrapper_TN_cll(TN, TNuv, MomInv, mass2,&Nn,&R,TNerr,&N);

    std::cout<<"heeeere"<<std::endl;

    std::cout<<TN[0]<<" "<<TN[1]<<std::endl;



    double _Complex TNten [15]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    double _Complex TNtenuv [15]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    double _Complex momVec [4]={1,0,0,0};
    double TNtenerr[15]={1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    N=1;



    wrapper_TNten_cll(TNten, TNtenuv, momVec, MomInv, mass2,&Nn,&R,TNtenerr,&N);

    std::cout<<TNten[0]<<" "<<TNten[1]<<std::endl;

    std::cout<<"finished :)"<<std::endl;

    return 0;
}