# ./video_runner.sh "${ARR[@]}"
ARR=( "$@" )
videos=()
for i in "${ARR[@]}"
do
	:
	while IFS=  read -r -d $'\0'; do
		videos+=("$REPLY")
	done < <(find /na/homes/ryerramsetty/../../../export/research/analysis/human/kkiehl/media/ -name "*$i**.wmv" -print0)
	
	for j in "${videos[@]}"
	do
	 	:
	 	python blink_tester_franklin.py --shape-predictor ~/file_transfer_bucket/shape_predictor_68_face_landmarks.dat --video $j
	 done
done
printf ' ->%s\n' "${videos[@]}"	
printf "ARR array contains %d elements: " ${#videos[@]}

