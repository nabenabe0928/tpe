#!/bin/bash
USER=$(whoami)
CWD=$(dirname $0)


echo $USER:~$CWD$ rm -r evaluation/CNN/000
rm -r evaluation/CNN/000
echo $USER:~$CWD$ rm log/CNN/000/*
rm log/CNN/000/*
