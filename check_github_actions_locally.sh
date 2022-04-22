run_check() {
    proc_name=${1}
    cmd=${2}
    echo "### Start $proc_name ###"
    echo $cmd
    $cmd
    echo "### Finish $proc_name ###"
    printf "\n\n"
}

target="tpe"  # target must be modified accordingly
run_check "pre-commit" "pre-commit run --all-files"
run_check "pytest" "python -m pytest -W ignore --cov-report term-missing --cov=$target --cov-config=.coveragerc"
run_check "black" "black test/ $target/"
