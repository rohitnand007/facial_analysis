# ./video_runner.sh "${ARR[@]}"
# bash array should be in the format of array=("Rohit" "nand") and not a=["rohit","nand"]
ARR=( "$@" )
videos=()
printf "Input ARR contains %d elements.\n" ${#ARR[@]}
for i in "${ARR[@]}"
do
	:
	search_array=(`find /na/homes/ryerramsetty/../../../export/research/analysis/human/kkiehl/media/BBP_20150/Assessment_Videos/NewMexico/Adult_Incarcerated/Male -name '*.wmv' -print | grep "$i"`)
	for ele in "${search_array[@]}"; do videos+=("${ele}"); done
		
done
for j in "${videos[@]}"
do
	:
	 	python blink_tester_franklin.py --shape-predictor ~/file_transfer_bucket/shape_predictor_68_face_landmarks.dat --video "$j"
done
printf ' ->%s\n' "${videos[@]}"	
printf "Final array contains %d elements...EOM\n " ${#videos[@]}

