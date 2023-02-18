
mkdir movies

for alldirs in /Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/*; do
	isbiname=$(basename $alldirs)
	echo $isbiname
	basedir="/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/$isbiname/track/png/"
	vf="pad=ceil(iw/2)*2:ceil(ih/2)*2,setpts=2.0*PTS,format=yuv420p" ## makes height & width divisible by 2
	ffmpeg -y -pattern_type glob -i $basedir/'img*.png' -c:v libx264 -vf $vf movies/$isbiname.mp4
done;

