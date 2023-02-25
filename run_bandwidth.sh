echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_benchmarks --dim 5

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_benchmarks --dim 10

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_benchmarks --dim 30

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpolib

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpobench --dataset_id 0

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpobench --dataset_id 1

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpobench --dataset_id 2

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpobench --dataset_id 3

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpobench --dataset_id 4

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpobench --dataset_id 5

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpobench --dataset_id 6

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_hpobench --dataset_id 7

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_jahs --dataset_id 0

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_jahs --dataset_id 1

echo `date '+%y/%m/%d %H:%M:%S'`
python -m scripts.bandwidth.run_jahs --dataset_id 2
