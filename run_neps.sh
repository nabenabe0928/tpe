for num in `seq 0 3`; do
    python run_neps_worker.py &
    pids[${num}]=$!
    echo "Start Proc. $num"
done

for pid in ${pids[*]}; do
    wait $pid
    echo "Finish Proc. $pid"
done
