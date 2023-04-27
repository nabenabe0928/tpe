for n_workers in 1 2 4 8 16
do
    for num in `seq 1 ${n_workers}`; do
        python run_neps_worker.py --seed 0 --dataset_id 0 --bench_name jahs --n_workers $n_workers &
        pids[${num}]=$!
        echo "Start Proc. $num"
    done

    for pid in ${pids[*]}; do
        wait $pid
        echo "Finish Proc. $pid"
    done
done
