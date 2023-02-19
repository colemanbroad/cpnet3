mkdir inspect_data
rm -rf inspect_data/*

for alldirs in /Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/*
do
	isbiname=$(basename $alldirs)
	echo $isbiname
	basedir="/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/$isbiname/data/png/"
	
	## if directory exists and isn't empty
	if [ -d $basedir ] && [ ! -z "$(ls -A $basedir)" ]
	then
		targetdir="inspect_data/$isbiname/"
		mkdir -p $targetdir
		for img in $basedir/*
		do
			name=$targetdir/$(basename $img)
			printf "$name \n"
			cp $img $name
		done
	fi
done

