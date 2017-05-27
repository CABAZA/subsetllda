#!/bin/bash

dir="FastXML_PfastreXML"
dataset=$1
data_dir="data/$dataset"

mkdir $dir/Results/ ;

trn_lbl_file="${data_dir}/trn_X_Y.txt"
tst_lbl_file="${data_dir}/tst_X_Y.txt"

perl $dir/Tools/convert_format.pl $data_dir/train $trn_ft_file $trn_lbl_file
perl $dir/Tools/convert_format.pl $data_dir/test $tst_ft_file $tst_lbl_file


# performance evaluation 
matlab -nodesktop -nodisplay -r "cd('$PWD'); addpath(genpath('$dir/Tools')); trn_X_Y = read_text_mat('$trn_lbl_file'); tst_X_Y = read_text_mat('$tst_lbl_file'); wts = inv_propensity(trn_X_Y,0.55,1.5); score_mat = read_text_mat('$2'); get_all_metrics(score_mat, tst_X_Y, wts); exit;"

