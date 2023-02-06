
# mkdir -p movies

for alldirs in /Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/*; do
	isbiname=$(basename $alldirs)
	echo $isbiname
	basedir="/Users/broaddus/Desktop/mpi-remote/project-broaddus/cpnet3/cpnet-out/$isbiname/track/png/"
	ffmpeg -y -pattern_type glob -i $basedir/'img*.png' -c:v libx264 -vf "setpts=2.0*PTS,format=yuv420p" movies/$isbiname.mp4
done;

