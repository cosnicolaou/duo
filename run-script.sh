#!/usr/bin/env bash


script="$1"
if [[ -z "$script" ]]; then
	echo "specify a script"
	exit 1
fi
scriptfile="scripts/${script}.sh"
if [[ ! -f "${scriptfile}" ]]; then
	echo "script ${scriptfile} does not exist"
	exit 1
fi
shift
ds=$(date '+%Y-%m-%d:%H:%M:%S')
mkdir -p logs
(time bash ${scriptfile} "$@") 2>&1 | tee logs/${script}-${ds}.log
rm -f logs/${script}-latest.log
ln -s ${script}-${ds}.log logs/${script}-latest.log
