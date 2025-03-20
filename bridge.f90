module collier_wrapper
  use iso_c_binding, only: c_char, c_int, c_null_char, c_ptr, c_f_pointer, c_double_complex, c_loc, c_double
  use COLLIER  
  implicit none

contains


  ! WRAPPER FOR INIT FUNCTION

  subroutine wrapper_init(Nmax, rmax, output_file) bind(c, name="wrapper_init")
    integer(c_int), intent(in) :: Nmax, rmax
    type(c_ptr), value :: output_file   ! Accept a C pointer (i.e. char*)
    character(len=256) :: fortran_output_file

    fortran_output_file = c_f90string(output_file)

    call Init_cll(Nmax, rmax, fortran_output_file)

  end subroutine wrapper_init

  ! FUNCTION THAT CONVERTS C STRINGS TO FORTRAN STRINGS

  function c_f90string(c_str_ptr) result(f90_str)
    type(c_ptr), value :: c_str_ptr
    character(len=256) :: f90_str
    integer :: i
    character(c_char), pointer :: c_chars(:)

    call c_f_pointer(c_str_ptr, c_chars, [256])
    f90_str = ' '  ! initialize f90_str to spaces

    do i = 1, 256
      if (c_chars(i) == c_null_char) exit  ! stop at null terminator
      f90_str(i:i) = c_chars(i)
    end do

    if (i > 1) then
      f90_str = f90_str(:i-1)  ! trim the Fortran string to actual length
    else
      f90_str = ''
    end if
  end function c_f90string

  ! WRAPPER TO GET_NC FUNCTION

  function wrapper_GetNc_cll(N,rank) result(nc) bind(c, name='wrapper_GetNc_cll')
    integer(c_int), intent(in) :: N, rank
    integer(c_int) :: nc
    nc = GetNc_cll(N,rank)
  end function wrapper_GetNc_cll

  ! WRAPPER TO TN_CLL FUNCTION

  subroutine wrapper_TN_cll(TN,TNuv,MomInv,mass2,Nn,R,TNerr,N) bind(c, name='wrapper_TN_cll')
    
    complex(c_double_complex), intent(in), target :: TN(*), TNuv(*), MomInv(*), mass2(*)
    real(c_double), intent(in), target :: TNerr(*)
    integer(c_int), intent(in) :: Nn, R, N

    complex(c_double_complex), pointer :: fTN(:), fTNuv(:), fMomInv(:), fmass2(:)
    real(c_double), pointer :: fTNerr(:)

    integer(c_int) :: nc
    integer(c_int) :: np
    nc=GetNc_cll(N,R)
    np=N*(N-1)/2

    call c_f_pointer(c_loc(TN(1)), fTN, [nc])
    call c_f_pointer(c_loc(TNuv(1)), fTNuv, [nc])
    call c_f_pointer(c_loc(MomInv(1)), fMomInv, [np])
    call c_f_pointer(c_loc(mass2(1)), fmass2, [N])
    call c_f_pointer(c_loc(TNerr(1)), fTNerr, [R])

    if (N == 1) then
      call TN_cll(fTN,fTNuv,fmass2,Nn,R,fTNerr)
    else
      call TN_cll(fTN,fTNuv,fMomInv,fmass2,Nn,R,fTNerr)
    end if

  end subroutine wrapper_TN_cll

  ! WRAPPER TO GET_NT FUNCTION

  function wrapper_GetNt_cll(rank) result(nt) bind(c, name='wrapper_GetNt_cll')
    integer(c_int), intent(in) :: rank
    integer(c_int) :: nt
    nt = GetNt_cll(rank)
  end function wrapper_GetNt_cll

  ! WRAPPER TO TNTEN_CLL FUNCTION

  subroutine wrapper_TNten_cll(TNten,TNtenuv,Momvec,MomInv,mass2,Nn,R,TNtenerr,N) bind(c, name='wrapper_TNten_cll')

    ! Input vectors as assumed-shape arrays:
    complex(c_double_complex), intent(in), target :: TNten(*)
    complex(c_double_complex), intent(in), target :: TNtenuv(*)
    complex(c_double_complex), intent(in), target :: MomInv(*)
    complex(c_double_complex), intent(in), target :: mass2(*)
    ! Input matrix:
    complex(c_double_complex), intent(in), target :: Momvec(*)
    ! Integer inputs:
    integer(c_int), intent(in) :: Nn, R, N
    ! Output vector:
    complex(c_double), intent(in), target :: TNtenerr(*)



    complex(c_double_complex), pointer :: fTNten(:), fTNtenuv(:), fMomInv(:), fmass2(:)
    complex(c_double_complex), pointer :: fMomvec(:,:)
    real(c_double), pointer :: fTNtenerr(:)

    integer(c_int) :: nc
    integer(c_int) :: np
    integer(c_int) :: nt


    nc=GetNc_cll(N,R)
    nt=GetNt_cll(R)
    np=N*(N-1)/2

    

    call c_f_pointer(c_loc(TNten(1)), fTNten, [nt])
    call c_f_pointer(c_loc(TNtenuv(1)), fTNtenuv, [nt])
    call c_f_pointer(c_loc(mass2(1)), fmass2, [N])
    call c_f_pointer(c_loc(TNtenerr(1)), fTNtenerr, [R])



    
    if (N == 1) then
      call TNten_cll(fTNten,fTNtenuv,fmass2,Nn,R,fTNtenerr)
    else
      call c_f_pointer(c_loc(MomInv(1)), fMomInv, [np])
      call c_f_pointer(c_loc(Momvec(1)), fMomvec, [4,N])
      call TNten_cll(fTNten,fTNtenuv,fMomvec,fMomInv,fmass2,Nn,R,fTNtenerr)
    end if

  end subroutine wrapper_TNten_cll

end module collier_wrapper
