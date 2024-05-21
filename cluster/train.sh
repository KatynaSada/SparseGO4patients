#!/bin/bash

# CHANGE THIS - parameters to modify
epoch=300


# These parameters can also be tunned in train_gpu_wb.py using Weights & Biases (W&B) ----- THEY ARE
batchsize=1000 # take into account number of GPUS
lr=0
decay_rate=0
num_neurons_per_GO=6
num_neurons_final_GO=6
num_neurons_final=6
drug_neurons='100,50,6'
# -----

projectname="CL_PDCs2018_expression_quartile" # CHANGE THIS - name you want to give to your W&B project
foldername="CL_PDCs2018_expression_brain" 
mkdir "../results/"$foldername

# The loop was created to facilitate cross-validation, more than 1 folder can be provided to create different models using different samples for training and testing
source activate /scratch/ksada/envs/SparseGO # CHANGE THIS - your environment
for samples in "allsamples" "samples1" "samples2" "samples3" "samples4" "samples5"
# "samples1" "samples2" "samples3" "samples4" "samples5" "allsamples" 
# "allsamples"  # CHANGE THIS - folder(s) where you have the data
do
  inputdir="../data/"$foldername"/"$samples"/" # CHANGE THIS - folder where you have the folder(s) of data
  modeldir="../results/"$foldername"/"$samples"/" # CHANGE THIS - folder to store results
  mkdir $modeldir

  type="" # CHANGE THIS - add something if files have different endings

  gene2idfile=$inputdir"gene2ind"$type".txt"
  cell2idfile=$inputdir"cell2ind"$type".txt"
  drug2idfile=$inputdir"drug2ind"$type".txt"
  traindatafile=$inputdir"sparseGO_train.txt"
  valdatafile=$inputdir"sparseGO_val.txt"
  #drugfile=$inputdir"drug2fingerprint"$type".txt"
  drugfile=$inputdir"drug2fingerprint"$type".txt"
  drugs_names=$inputdir"compound_names"$type".txt"

  ontfile=$inputdir"sparseGO_ont"$type".txt" # CHANGE THIS - ontology file

  #mutationfile=$inputdir"cell2mutation"$type".txt" # CHANGE THIS - expression/mutation file
  mutationfile=$inputdir"cell2expression_quartile"$type".txt"

  testdatafile=$inputdir"sparseGO_test.txt"

  wandb login b1f6d1cea53bb6557df3c1c0c0530b53cadeed3d # CHANGE THIS - your W&B account
  python -u "../code/train.py" -train $traindatafile -val $valdatafile -onto $ontfile -gene2id $gene2idfile -drug2id $drug2idfile -cell2id $cell2idfile -genotype $mutationfile -fingerprint $drugfile -drugs_names $drugs_names -epoch $epoch -lr $lr -decay_rate $decay_rate -batchsize $batchsize -modeldir $modeldir -num_neurons_per_GO $num_neurons_per_GO -num_neurons_final_GO $num_neurons_final_GO -drug_neurons $drug_neurons -num_neurons_final $num_neurons_final -predict $testdatafile -result $modeldir -project $projectname  -sweep_name $samples > $modeldir"train_correlation.log"

done

# Create the plots/graphs/final metrics
type=""
input_folder="../data/"$foldername"/"
output_folder="../results/"$foldername"/"
model_name="best_model_d.pt"
predictions_name="ModelPerDrug_test_predictions.txt"
labels_name="sparseGO_test.txt"
ontology_name="sparseGO_ont"$type".txt"
genomics_name="cell2expression_quartile"$type".txt"
druginput_name="drug2fingerprint"$type".txt"
gene2id_name="gene2ind"$type".txt"
#txt_type="PDCs2018" #  -txt_type $txt_type 
# samples_folders and txt_type has to be changed in the python file (for now)

python -u "../code/per_drug_correlation.py" -gene2id $gene2id_name -input_folder $input_folder -output_folder $output_folder -model_name $model_name -predictions_name $predictions_name -labels_name $labels_name -ontology_name $ontology_name -genomics_name $genomics_name -project $projectname -druginput_name $druginput_name >$modeldir"metrics.log"

