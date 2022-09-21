# First using this conda in a new path, need to run this shell to initialize conda.
#
# Make sure the new path for python exists.
#
# You can execute twice and look at the second print to make sure the changed correctly.

CONDA_DIR=./conda
CONDA_DIR_full_path=$(readlink -f $CONDA_DIR)

docu=$CONDA_DIR/bin/conda
ab_full_path=$(readlink -f $docu)
ab_path=$(dirname $ab_full_path)
Python_Exe_Path=$ab_path/python
fir_line=$(sed -n 1p $docu)
echo "old python path: $fir_line"
sed -i "s+$fir_line+#!$Python_Exe_Path+" $docu
echo -e  "new python path: $(sed -n 1p $docu)\n"

docu=$CONDA_DIR/bin/conda-env
fir_line=$(sed -n 1p $docu)
echo "old python path: $fir_line"
sed -i "s+$fir_line+#!$Python_Exe_Path+" $docu
echo -e "new python path: $(sed -n 1p $docu)\n"

docu=$CONDA_DIR/bin/pip
fir_line=$(sed -n 1p $docu)
echo "old python path: $fir_line"
sed -i "s+$fir_line+#!$Python_Exe_Path+" $docu
echo -e "new python path: $(sed -n 1p $docu)\n"

docu=$CONDA_DIR/condabin/conda
fir_line=$(sed -n 1p $docu)
echo "old python path: $fir_line"
sed -i "s+$fir_line+#!$Python_Exe_Path+" $docu
echo -e "new python path: $(sed -n 1p $docu)\n"


docu=$CONDA_DIR/bin/activate
sed_line=$(sed -n 2p $docu)
echo "old path for conda directory: $sed_line"
new_path=\_CONDA\_ROOT\=\"$CONDA_DIR_full_path\"
sed -i "s+$sed_line+$new_path+" $docu
echo -e "new path for conda directory: $(sed -n 2p $docu)\n"


docu=$CONDA_DIR/bin/deactivate
sed_line=$(sed -n 2p $docu)
echo "old path for conda directory: $sed_line"
new_path=\_CONDA\_ROOT\=\"$CONDA_DIR_full_path\"
sed -i "s+$sed_line+$new_path+" $docu
echo -e "new path for conda directory: $(sed -n 2p $docu)\n"

docu=$CONDA_DIR/bin/qt.conf
echo -e  "old path in qt:\n$(sed -n '2,$p' $docu)\n"
_first_line=$(sed -n 2p $docu)
_sed_line=$(sed -n 3p $docu)
_third_line=$(sed -n 4p $docu)
_fourth_line=$(sed -n 5p $docu)

first_line="Prefix \= $CONDA_DIR_full_path"
sed_line="Binaries \= $CONDA_DIR_full_path/bin"
third_line="Libraries \= $CONDA_DIR_full_path/lib"
fourth_line="Headers \= $CONDA_DIR_full_path/include/qt"

sed -i "s+$_first_line+$first_line+" $docu
sed -i "s+$_sed_line+$sed_line+" $docu
sed -i "s+$_third_line+$third_line+" $docu
sed -i "s+$_fourth_line+$fourth_line+" $docu
echo -e  "new path in qt:\n$(sed -n '2,$p' $docu)\n"

docu=$CONDA_DIR/bin/tensorboard
first_line=$(sed -n 1p $docu)
echo "old python path:$first_line"
new_path="#!$Python_Exe_Path"
sed -i "s+$first_line+$new_path+" $docu
echo -e "new python path: $(sed -n 1p $docu)\n"

docu=$CONDA_DIR/etc/profile.d/conda.csh
echo -e  "old path in conda.csh :\n$(sed -n '1,4p' $docu)\n"
_first_line=$(sed -n 1p $docu)
_sed_line=$(sed -n 2p $docu)
_third_line=$(sed -n 3p $docu)
_fourth_line=$(sed -n 4p $docu)


first_line="setenv CONDA_EXE\ '$CONDA_DIR_full_path/bin/conda'"
sed_line="setenv _CONDA_ROOT\ '$CONDA_DIR_full_path'"
third_line="setenv _CONDA_EXE\ '$CONDA_DIR_full_path/bin/conda'"
fourth_line="setenv CONDA_PYTHON_EXE\ '$CONDA_DIR_full_path/bin/python'"

sed -i "s+$_first_line+$first_line+" $docu
sed -i "s+$_sed_line+$sed_line+" $docu
sed -i "s+$_third_line+$third_line+" $docu
sed -i "s+$_fourth_line+$fourth_line+" $docu
echo -e  "new path in conda.csh:\n$(sed -n '1,4p' $docu)\n"

docu=$CONDA_DIR/etc/profile.d/conda.sh
echo -e  "old path in conda.sh :\n$(sed -n '1,4p' $docu)\n"
_first_line=$(sed -n 1p $docu)
_sed_line=$(sed -n 2p $docu)
_third_line=$(sed -n 3p $docu)
_fourth_line=$(sed -n 4p $docu)


first_line="export CONDA_EXE=\'$CONDA_DIR_full_path/bin/conda'"
fourth_line="export CONDA_PYTHON_EXE=\'$CONDA_DIR_full_path/bin/python'"

sed -i "s+$_first_line+$first_line+" $docu
sed -i "s+$_fourth_line+$fourth_line+" $docu
echo -e  "new path in conda.sh:\n$(sed -n '1,4p' $docu)\n"
