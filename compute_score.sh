function public_score()
{
	target_dir="/str/users/angus/Codes/yang/SPiCe_Offline/targets"
	ranking_dir='/str/users/angus/Codes/yang/tr_b_30_15_2_'
	models=("mlp" "cnn" "lstm")
	s=1
	n=16
    for model in ${models[@]}
	do
        sum=0.0
		for ((i=$s;i<$n;i=$i+1))
		do
			ranking_file="${ranking_dir}${model}/${i}.ranking"
			target_file="${target_dir}/${i}.spice.target.public"
			score_file="${ranking_dir}${model}/${i}_public.score"
			python2.7 score_computation.py $ranking_file $target_file > ${score_file}
            score=$(cat ${score_file})
			echo "${i}:${score}"
            sum=$(echo "$sum + $score" | bc)
		done
       echo "Final public score of ${model} is ${sum}."
	done
}
public_score
