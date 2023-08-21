!-------------------------------------------------------------------------------------------------
! Fortran source code for module dot.f90_impl
!-------------------------------------------------------------------------------------------------
! Remarks:
!   . Enter Python documentation for this module in .
!     You might want to check the f2py output for the interfaces of the C-wrapper functions.
!     It will be autmatically included in the dot documentation.
!   . Documument the Fortran routines in this file. This documentation will not be included
!     in the dot documentation (because there is no recent sphinx
!     extension for modern fortran).

real*8 function dot(a,b,n)
  ! Compute the dot product of two 1d arrays.
  !
    implicit none
  !-------------------------------------------------------------------------------------------------
    integer*4              , intent(in)    :: n
    real*8   , dimension(n), intent(in)    :: a,b
  ! real*8                 , intent(out)   :: dot
    ! intent is inout because we do not want to return an array to avoid needless copying
  !-------------------------------------------------------------------------------------------------
  ! declare local variables
    integer*4 :: i
  !-------------------------------------------------------------------------------------------------
    dot = 0.0
    do i=1,n
        dot = dot + a(i) * b(i)
    end do
end function dot
