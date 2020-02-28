#!/bin/bash

for file in ./version/*
do
	make clean
	make version=${file}    
	./conv2d
done

