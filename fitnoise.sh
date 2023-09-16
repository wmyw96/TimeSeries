for t in 500 1000 2000 4000 6000 8000 10000 12000 14000 16000 18000 20000
do
		for d in 5 10 20 30 40
		do
			echo $t $d
			python fitar.py --dim d --sigma 0.3 --signal 0.95 --T T --order 1
	done
done