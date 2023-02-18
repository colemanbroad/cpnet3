mkdir inspect_data

for alldirs in /Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/*; do
	isbiname=$(basename $alldirs)
	echo $isbiname
	basedir="/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/$isbiname/data/png/"
	for img in $basedir/*; do
		name="inspect_data/$isbiname-$(basename $img)"
		printf "$name \n"
		cp $img $name
	done;
done;

