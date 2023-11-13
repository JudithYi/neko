AC_DEFUN([AX_PYTHON],[
	AC_ARG_WITH([python],
	AS_HELP_STRING([--with-python=DIR],
	[Directory for PYTHON]),
	[
	if test -d "$withval"; then
	    ac_python_path="$withval";
        AS_IF([test -d "$ac_python_path/lib64"],
          	             [suffix="64"],[suffix=""])
		PYTHON_LDFLAGS="-L$ac_python_path/lib$suffix"
       	PYTHON_CPPFLAGS="-I$ac_python_path/lib$suffix/python3.8/site-packages/numpy/core/include -I$ac_python_path/include/python3.8/ -fpermissive  "
        PYTHON_LIB="-lpython3.8"
	fi		
	],[with_paraview=no])

	if test "x${with_python}" != xno; then
	   if test -d "$ac_python_path"; then
            PATH="$ac_python_path/bin:$PATH"
            CPPFLAGS="$PYTHON_CPPFLAGS $CPPFLAGS"
            LDFLAGS="$PYTHON_LDFLAGS $LDFLAGS"
            export CPPFLAGS
            export LDFLAGS
	   fi	    
		LIBS="$PYTHON_LIB $LIBS"
	fi
])
	
