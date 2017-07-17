#!/bin/bash

dir="FastXML_PfastreXML"
#dataset=$1
#data_dir="data/$dataset"

trn_ft_file="trn_X_Xf.txt"
trn_lbl_file="trn_X_Y.txt"
tst_ft_file="tst_X_Xf.txt"
tst_lbl_file="tst_X_Y.txt"
inv_prop_file="inv_prop.txt"

perl $dir/Tools/convert_format.pl train $trn_ft_file $trn_lbl_file
perl $dir/Tools/convert_format.pl $1 $tst_ft_file $tst_lbl_file


# performance evaluation 
matlab -nodesktop -nodisplay -r "cd('$PWD'); addpath(genpath('$dir/Tools')); trn_X_Y = read_text_mat('$trn_lbl_file'); tst_X_Y = read_text_mat('$tst_lbl_file'); wts = inv_propensity(trn_X_Y,0.55,1.5);score_mat = read_text_mat('$2');get_all_metrics(score_mat, tst_X_Y, wts); exit;"

