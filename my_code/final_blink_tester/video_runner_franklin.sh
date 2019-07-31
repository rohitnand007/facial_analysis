# ./video_runner.sh "${ARR[@]}"
ARR=( "$@" )

for i in "${ARR[@]}"
do
	:
	#videos=$(find /na/homes/ryerramsetty/../../../export/research/analysis/human/kkiehl/media/ -name "*$i**.wmv")
	# printf "ARR array contains %d elements: " ${#videos[@]}
	# printf ' ->%s\n' "${videos[@]}"
	videos=()
	while IFS=  read -r -d $'\0'; do
    videos+=("$REPLY")
	done < <(find /na/homes/ryerramsetty/../../../export/research/analysis/human/kkiehl/media/ -name "*$i**.wmv" -print0)
	printf "ARR array contains %d elements: " ${#videos[@]}
	printf ' ->%s\n' "${videos[@]}"
	# for j in "${videos[@]}"
	# do
	# 	:
	# 	python blink_tester_franklin.py --shape-predictor ~/file_transfer_bucket/shape_predictor_68_face_landmarks.dat --video $j
	# done
done	
