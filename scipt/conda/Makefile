.PHONY: clean clone create_env create_env_from_file remove_env env_list list
en=iobjectspy
v=3.7
d=requirements.yml
ep=/root/iobjectspy
create_env:
	conda create -n ${en} python=${v}
create_env_from_file:
	conda env create -f ${d}
clean:
	conda clean -p && conda clean -y --all
remove_env:
	conda remove -n ${en} --all
clone:
	conda create -n ${en} --clone ${ep}
env_list:
	conda env list
list:
	conda list
