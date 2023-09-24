for m in 16 24 32 40 48 56 64 72 96 128 160 192 224 256
do
		for d in 5 10 20
		do
			echo $t $d
			python fitar.py --dim $d --sigma 0.3 --signal 0.95 --dmodel $m --order 1 --seed 0 --record_dir logs/ --dropout 0 --T 5000
	done
done