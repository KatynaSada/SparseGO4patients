#!/bin/bash

# CHANGE THIS - parameters to modify
epoch=300


# These parameters can also be tunned in train_gpu_wb.py using Weights & Biases (W&B) ----- THEY ARE
# batchsize=1000 # take into account number of GPUS
# lr=0
# decay_rate=0
# cudaid=0
# -----



projectname="Hyper_PDCs2018_expression_transfer_LELO" # CHANGE THIS - name you want to give to your W&B project
foldername="PDCs2018_expression_LELO"
pretrainedmodel="../results/CL_PDCs2018_expression_brain/allsamples/last_model.pt"
mkdir "../results/"$foldername"_pretrained/"

tags="quartile,brain"
# The loop was created to facilitate cross-validation, more than 1 folder can be provided to create different models using different samples for training and testing
source activate /scratch/ksada/envs/SparseGO # CHANGE THIS - your environment
for samples in "samples1"
#"samples2" "samples3" "samples4" "samples5" "allsamples"  # CHANGE THIS - folder(s) where you have the data
do
  inputdir="../data/"$foldername"/"$samples"/" # CHANGE THIS - folder where you have the folder(s) of data
  modeldir="../results/"$foldername"_pretrained/"$samples"/" # CHANGE THIS - folder to store results
  mkdir $modeldir

  type="_PDCs2018" # CHANGE THIS - add something if files have different endings

  gene2idfile=$inputdir"gene2ind"$type".txt"
  cell2idfile=$inputdir"cell2ind"$type".txt"
  drug2idfile=$inputdir"drug2ind"$type".txt"
  traindatafile=$inputdir"sparseGO_train.txt"
  valdatafile=$inputdir"sparseGO_val.txt"
  drugfile=$inputdir"drug2fingerprint"$type".txt"
  drugs_names=$inputdir"compound_names"$type".txt"

  ontfile=$inputdir"sparseGO_ont"$type".txt" # CHANGE THIS - ontology file
  # ontfile=$inputdir"ontology.txt"

  mutationfile=$inputdir"cell2expression"$type"_quartile.txt" # CHANGE THIS - expression/mutation file
  #mutationfile=$inputdir"cell2expression"$type".txt"

  testdatafile=$inputdir"sparseGO_test.txt"

  wandb login b1f6d1cea53bb6557df3c1c0c0530b53cadeed3d # CHANGE THIS - your W&B account
  python -u "../code/train_transfer.py" -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -train $traindatafile -val $valdatafile -modeldir $modeldir -genotype $mutationfile -fingerprint $drugfile -drugs_names $drugs_names -epoch $epoch -predict $testdatafile -result $modeldir -project $projectname -pretrained $pretrainedmodel -sweep_name $samples -tags $tags > $modeldir"train_correlation.log"
done
#conda info --envs
# ----- Create the plots/graphs/final metrics
type="_PDCs2018"
input_folder="../data/"$foldername"/"
output_folder="../results/"$foldername"_pretrained/"
model_name="best_model_s.pt"
predictions_name="ModelSpearman_test_predictions.txt"
labels_name="sparseGO_test.txt"
ontology_name="sparseGO_ont"$type".txt"
genomics_name="cell2mutation"$type".txt"
txt_type="_PDCs2018"
# samples_folders and txt_type has to be changed in the python file (for now)

# python -u "../code/per_drug_correlation.py" -input_folder $input_folder -output_folder $output_folder -model_name $model_name -predictions_name $predictions_name -labels_name $labels_name -ontology_name $ontology_name -genomics_name $genomics_name -project $projectname -txt_type $txt_type >$modeldir"metrics.log"

