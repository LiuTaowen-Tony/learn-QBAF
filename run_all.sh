#!/bin/bash
source venv/bin/activate
for d in iris mushroom adult; do
  for s in {1..5}; do
    for j in {0..2}; do 
      for dd in direct no; do
        for ff in fuzzy no; do
          python run_exp.py $d $j $s $dd sp $ff
        done
      done
    done
  done
done

for d in iris mushroom adult; do
  for s in {1..5}; do
    python run_exp_multigbag.py $d $s sp no
  done
done