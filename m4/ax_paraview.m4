AC_DEFUN([AX_PARAVIEW],[
	AC_ARG_WITH([paraview],
	AS_HELP_STRING([--with-paraview=DIR],
	[Directory for PARAVIEW]),
	[
	if test -d "$withval"; then
	   ac_paraview_path="$withval";
	fi		
	],[with_paraview=no])

	if test "x${with_paraview}" != xno; then
	   PATH_SAVED="$PATH"
	   if test -d "$ac_paraview_path"; then
	      PATH="$ac_paraview_path/bin:$PATH"
	   fi

	   AC_CHECK_PROG(PARAVIEWCONF,paraview-config,yes)

	   if test x"${PARAVIEWCONF}" == x"yes"; then
	      PARAVIEW_CPPFLAGS=`paraview-config -f -c Catalyst`
	      CPPFLAGS="$PARAVIEW_CPPFLAGS $CPPFLAGS"

	      PARAVIEW_LDFLAGS=`paraview-config -l -c Catalyst`
              LDFLAGS="$PARAVIEW_LDFLAGS $LDFLAGS"
      	      with_paraview=yes
	      have_paraview=yes
	      AC_SUBST(have_paraview)
	    else
	      with_paraview=no
	    fi
            PATH="$PATH_SAVED"
	    
	fi
])
	

