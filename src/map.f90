!> NEKTON map
!! @todo Figure out a better name for this module
module map
  implicit none

  !> NEKTON vertex mapping
  type :: map_t
     integer :: nel, nlv
     integer, allocatable :: imap(:)
     integer, allocatable :: vertex(:,:)
  end type map_t

contains

  subroutine map_init(m, nel, nlv)
    type(map_t), intent(inout) :: m
    integer, intent(in) :: nel
    integer, intent(in) :: nlv

    call map_free(m)

    m%nel = nel
    m%nlv = nlv
    
    allocate(m%imap(m%nel))
    
    allocate(m%vertex(m%nlv, m%nel))
    
  end subroutine map_init

  subroutine map_free(m)
    type(map_t), intent(inout) :: m

    if (allocated(m%imap)) then
       deallocate(m%imap)
    end if

    if (allocated(m%vertex)) then
       deallocate(m%vertex)
    end if
  end subroutine map_free
  
end module map