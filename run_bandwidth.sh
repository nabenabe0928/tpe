echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_benchmarks --dim 5

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_benchmarks --dim 10

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_benchmarks --dim 30

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpolib

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpobench

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_jahs --dataset_id 0

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_jahs --dataset_id 1

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_jahs --dataset_id 2
