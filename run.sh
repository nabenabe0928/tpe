while getopts ":m:" o; do
    case "${o}" in
        m) mode=${OPTARG};;
    esac
done

if [[ "$mode" == "jahs" ]]
then
    echo "This process handles JAHS"
fi

python run_benchmarks.py --dim 10

python run_benchmarks.py --dim 5

python run_benchmarks.py --dim 30

python run_tabular.py

if [[ "$mode" == "jahs" ]]
then
    python run_jahs.py
fi
