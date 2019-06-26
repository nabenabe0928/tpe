#!/bin/bash
USER=$(whoami)
CWD=$(dirname $0)


echo $USER:~$CWD$ rm -r evaluation/square/000
rm -r evaluation/square/000
echo $USER:~$CWD$ rm log/square/000/*
rm log/square/000/*
