# ./video_runner.sh "${ARR[@]}"
# bash array should be in the format of array=("Rohit" "nand") and not a=["rohit","nand"]
ARR=( "$@" )
videos=()
printf "Input ARR contains %d elements.\n" ${#ARR[@]}
for i in "${ARR[@]}"
do
	:
	search_array=(`find /na/homes/ryerramsetty/../../../export/research/analysis/human/kkiehl/media/adult_06408/ /na/homes/ryerramsetty/../../../export/research/analysis/human/kkiehl/media/action_09542/ -name '*.wmv' -print | grep "$i"`)
	for ele in "${search_array[@]}"; do videos+=("${ele}"); done
		
done
for j in "${videos[@]}"
do
	:
	 	python image_segregation_franklin.py --shape-predictor ~/file_transfer_bucket/shape_predictor_68_face_landmarks.dat --video "$j"
done
printf ' ->%s\n' "${videos[@]}"	
printf "Final array contains %d elements...EOM\n " ${#videos[@]}

