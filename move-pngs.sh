
function move() {

	newdir=$1
	oldext=$2

	mkdir $newdir
	rm -rf $newdir/*

	for alldirs in /Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/*
	do
		isbiname=$(basename $alldirs)
		echo $isbiname
		basedir="/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/$isbiname/$oldext"
		
		## if directory exists and isn't empty
		if [ -d $basedir ] && [ ! -z "$(ls -A $basedir)" ]
		then
			targetdir="$newdir/$isbiname/"
			mkdir -p $targetdir
			for img in $basedir/*
			do
				name=$targetdir/$(basename $img)
				printf "$name \n"
				cp $img $name
			done
		fi
	done	
}

move "data-png" "data/png/"
move "movies" "track/png/"
move "pred" "predict/pred/"
