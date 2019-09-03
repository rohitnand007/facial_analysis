# ./video_runner.sh "${ARR[@]}"
ARR=( "$@" )
videos=()
for i in "${ARR[@]}"
do
	:
	#videos=$(find /na/homes/ryerramsetty/../../../export/research/analysis/human/kkiehl/media/ -name "*$i**.wmv")
	# printf "ARR array contains %d elements: " ${#videos[@]}
	# printf ' ->%s\n' "${videos[@]}"
	while IFS=  read -r -d $'\0'; do
		videos+=("$REPLY")
	done < <(find /na/homes/ryerramsetty/../../../export/research/analysis/human/kkiehl/media/ -name "*$i**.wmv" -print0)
	
	for j in "${videos[@]}"
	do
	 	:
	 	printf "this video runs here: " ${#j}
	 done
done
printf ' ->%s\n' "${videos[@]}"	
printf "ARR array contains %d elements: " ${#videos[@]}

