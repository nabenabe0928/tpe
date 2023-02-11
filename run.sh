echo `date '+%y/%m/%d %H:%M:%S'`
python run_benchmarks.py --dim 5

echo `date '+%y/%m/%d %H:%M:%S'`
python run_benchmarks.py --dim 10

echo `date '+%y/%m/%d %H:%M:%S'`
python run_benchmarks.py --dim 30

echo `date '+%y/%m/%d %H:%M:%S'`
python run_hpolib.py

echo `date '+%y/%m/%d %H:%M:%S'`
python run_hpobench.py

echo `date '+%y/%m/%d %H:%M:%S'`
python run_jahs.py --dataset_id 0

echo `date '+%y/%m/%d %H:%M:%S'`
python run_jahs.py --dataset_id 1

echo `date '+%y/%m/%d %H:%M:%S'`
python run_jahs.py --dataset_id 2
