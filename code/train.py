"""
    This script trains a SparseGO network and makes predictions using the trained network.
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.utils.data as du
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import util
from util import *
from network_dropout import sparseGO_nn
import argparse
import time
import wandb
import random

#  Ensure that training is as reproducible as possible
seed=123
torch.manual_seed(seed) # Random seed for CPU tensors
torch.cuda.manual_seed_all(seed) # Seed for generating random numbers for the current GPU
torch.backends.cudnn.deterministic = True #  Is done to ensure that the computations are deterministic, meaning they will produce the same results given the same input.
torch.backends.cudnn.benchmark = False
random.seed(seed)
np.random.seed(seed)

since0 = time.time()
parser = argparse.ArgumentParser(description='Train SparseGO')
mac = "/Users/katyna/Library/CloudStorage/OneDrive-Tecnun/"
windows = "C:/Users/ksada/OneDrive - Tecnun/"
manitou = "/manitou/pmg/users/ks4237/" 
computer = manitou # CHANGE

inputdir=computer+"SparseGO_code/data/PDCs2018_expression_zasinh_LELO/samples1/" # CHANGE
modeldir=computer+"SparseGO_code/results/PDCs2018_expression_zasinh_LELO/samples1/" # PUEDO CAMBIAR EL NOMBRE PARA GUARDAR EN OTRO SITIO

parser.add_argument('-sweep_name', help='W&B project name', type=str, default="samples1")
parser.add_argument('-train', help='Training dataset', type=str, default=inputdir+"sparseGO_train.txt")
parser.add_argument('-val', help='Validation dataset', type=str, default=inputdir+"sparseGO_val.txt")

parser.add_argument('-onto', help='Ontology file used to guide the neural network', type=str, default=inputdir+"sparseGO_ont_PDCs2018.txt")
parser.add_argument('-gene2id', help='Gene to ID mapping file', type=str, default=inputdir+"gene2ind_PDCs2018.txt")
parser.add_argument('-drug2id', help='Drug to ID mapping file', type=str, default=inputdir+"drug2ind_PDCs2018.txt")
parser.add_argument('-cell2id', help='Cell to ID mapping file', type=str, default=inputdir+"cell2ind_PDCs2018.txt")
parser.add_argument('-genotype', help='Mutation information for cell lines', type=str, default=inputdir+"cell2expression_PDCs2018.txt")
parser.add_argument('-fingerprint', help='Morgan fingerprint representation for drugs', type=str, default=inputdir+"drug2selfieslatent200_PDCs2018.txt")
parser.add_argument('-drugs_names', help='Drugs names and SMILEs', type=str, default=inputdir+"compound_names_PDCs2018.txt")

parser.add_argument('-epoch', help='Training epochs for training', type=int, default=100)
parser.add_argument('-lr', help='Learning rate', type=float, default=0.1)
parser.add_argument('-decay_rate', help='Learning rate decay', type=float, default=0.001)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=500)
parser.add_argument('-modeldir', help='Folder for trained models', type=str, default=modeldir)
parser.add_argument('-cuda_id', help='Specify GPU', type=int, default=0)

parser.add_argument('-num_neurons_per_GO', help='Mapping for the number of neurons in each term in genotype parts', type=int, default=6)
parser.add_argument('-num_neurons_final_GO', help='Mapping for the number of neurons in the root term', type=int, default=24)
parser.add_argument('-drug_neurons', help='Mapping for the number of neurons in each layer', type=str, default='100,50,6')
parser.add_argument('-num_neurons_final', help='The number of neurons in the top layer (before the output and after concatenating)', type=int, default=18)

parser.add_argument('-predict', help='Dataset to be predicted', type=str, default=inputdir+"sparseGO_test.txt")
parser.add_argument('-result', help='Result file name', type=str, default=modeldir)

parser.add_argument('-project', help='W&B project name', type=str, default="PDCs2018_expression_LELO_final")

# call functions
opt = parser.parse_args()
torch.set_printoptions(precision=5)

def train_model(run,config,model, optimizer, criterion, train_data, cell_features, drug_features, batch_size, model_dir, device, num_epochs,decay_rate):
     print('\nTraining started...')

     since = time.time()

     # initialize metrics
     max_corr_pearson = -1
     max_corr_spearman = -1
     max_corr_per_drug = -1
     min_loss = 1000

    # data for train and validation
     train_feature, train_label, val_feature, val_label = train_data

     # data to  corresponding device
     train_label_device = train_label.to(device, non_blocking=True)
     val_label_device = val_label.to(device, non_blocking=True).detach()
     print("\nTraining samples: ",train_label_device.shape[0])
     print("Val samples: ",val_label_device.shape[0])
     print('-----------------')

     train_loader = du.DataLoader(du.TensorDataset(train_feature,train_label), batch_size=batch_size, shuffle=False)
     val_loader = du.DataLoader(du.TensorDataset(val_feature,val_label), batch_size=batch_size, shuffle=False)

     optimizer.zero_grad()
     lr0 = optimizer.param_groups[0]['lr'] # extract learning rate

     for epoch in range(config.epochs):
         epoch_start_time = time.time()
         train_cum_loss = 0

         model.train() # Tells the model that you are training it. So effectively layers which behave different on the train and test procedures know what is going on
         train_predict = torch.zeros(0, 1, device=device) # initialize training results tensor

         # Learning rate decay
         optimizer.param_groups[0]['lr'] = lr0*(1/(1+decay_rate*epoch)) # or epoch*epoch

         # Training epoch
         for i, (inputdata, labels) in enumerate(train_loader):
             # Convert torch tensor to Variable
             features = build_input_vector(inputdata, cell_features, drug_features)
             
             device_features = features.to(device)
             device_labels = labels.to(device)

             # Forward pass & statistics
             out = model(device_features)
             train_predict = torch.cat([train_predict, out])

             normal_loss = criterion(out, device_labels)
             #corr_per_drug_batch = per_drug_corr(inputdata[:,1], out, device_labels)
             #penalty = (1/torch.sigmoid(torch.tensor(corr_per_drug_batch, dtype=torch.float32)))**3 # penalize based on the accuracy
             #penalty_nonan = torch.where(torch.isnan(penalty), torch.zeros_like(penalty), penalty)
             #loss_per_drug = per_drug_loss(criterion, inputdata[:,1], out, device_labels) # Takes into account predictions per drug (can also change it to per sample)
             #loss = normal_loss*2 + penalty_nonan
             loss = normal_loss
             train_cum_loss += float(loss.item())

             # Backwards pass & update
             optimizer.zero_grad() # zeroes the grad attribute of all the parameters passed to the optimizer
             loss.backward() # Computes the sum of gradients of given tensors with respect to graph leaves.
             optimizer.step() # Performs a parameter update based on the current gradient
             torch.cuda.empty_cache()

         train_corr = pearson_corr(train_predict, train_label_device) # compute pearson correlation
         train_corr_spearman = spearman_corr(train_predict.cpu().detach().numpy(), train_label_device.cpu()) # compute spearman correlation
         corr_per_drug = per_drug_corr(train_feature[:,1], train_predict, train_label_device)

         print('Epoch %d' % (epoch + 1))
         print("L.r. ", round(optimizer.param_groups[0]['lr'],6))
         print('Train Loss: {:.4f}; Train Corr (pear.): {:.5f}; Train Corr (sp.): {:.5f}; Per drug corr.: {:.5f}'.format(train_cum_loss, train_corr,train_corr_spearman, corr_per_drug))
         print('Allocated after train:', round(torch.cuda.memory_allocated(0)/1024**3,3), 'GB')
         print("    ")

         val_cum_loss = 0

         #Validation: random variables in training mode become static
         model.eval()

         val_predict = torch.zeros(0, 1, device=device)

         # Val epoch
         with torch.no_grad():
             for i, (inputdata, labels) in enumerate(val_loader):
                  # Convert torch tensor to Variable
                  features = build_input_vector(inputdata, cell_features, drug_features)

                  device_features = features.to(device)

                  # Forward pass & statistics
                  out = model(device_features)
                  val_predict = torch.cat([val_predict, out]) # concatenate predictions
                  torch.cuda.empty_cache()

         val_cum_loss = criterion(val_predict, val_label_device)
         val_corr = pearson_corr(val_predict, val_label_device) # compute correlation
         val_corr_spearman = spearman_corr(val_predict.cpu().detach().numpy(), val_label_device.cpu())
         val_corr_per_drug = per_drug_corr(val_feature[:,1], val_predict, val_label_device)
         val_corr_low = low_corr(val_predict, val_label_device,0.2)

         print('Val Loss: {:.4f}; Val Corr (pear.): {:.5f}; Val Corr (sp.): {:.5f}; Per drug corr.: {:.5f}'.format(val_cum_loss, val_corr,val_corr_spearman,val_corr_per_drug))
         print('Allocated after val:', round(torch.cuda.memory_allocated(0)/1024**3,3), 'GB')

         epoch_time_elapsed = time.time() - epoch_start_time
         print('Epoch time: {:.0f}m {:.0f}s'.format(
         epoch_time_elapsed // 60, epoch_time_elapsed % 60))

         print('-----------------')

         wandb.log({"pearson_val": val_corr,
                    "spearman_val": val_corr_spearman,
                    "corr_per_drug_val": val_corr_per_drug,
                    "corr_low": val_corr_low,
                    "loss_val": val_cum_loss
                    })
         
         if epoch > 100:
             if val_corr >= max_corr_pearson:
                 max_corr_pearson = val_corr
                 best_model_p = epoch+1
                 print("pearson: ",epoch+1)
        #         torch.save(model, model_dir + '/best_model_p.pt')
    
             if val_corr_spearman >= max_corr_spearman:
                 max_corr_spearman = val_corr_spearman
                 best_model_s = epoch+1
                 print("spearman: ",epoch+1)
        #         torch.save(model, model_dir + '/best_model_s.pt')
             
             if val_corr_per_drug >= max_corr_per_drug:
                 max_corr_per_drug = val_corr_per_drug
                 best_model_d = epoch+1
                 print("per drug: ",epoch+1)
                 torch.save(model, model_dir + '/best_model_d.pt')

             if val_cum_loss <= min_loss:
                 min_loss = val_cum_loss
                 best_model_l = epoch+1
                 print("loss: ",epoch+1)
        #         torch.save(model, model_dir + '/best_model_l.pt')

     wandb.log({"max_pearson_val": max_corr_pearson,
                "max_spearman_val": max_corr_spearman,
                "max_perdrug_val": max_corr_per_drug,
                "min_loss_val": min_loss,
                })

     torch.save(model, model_dir + '/last_model.pt')

     print("Best performed model (loss) (epoch)\t%d" % best_model_l,'loss: {:.6f}'.format(min_loss))

     print("Best performed model (pearson) (epoch)\t%d" % best_model_p, 'corr: {:.6f}'.format(max_corr_pearson))
     
     print("Best performed model (per drug) (epoch)\t%d" % best_model_d,'corr: {:.6f}'.format(max_corr_per_drug))

     print("Best performed model (spearman) (epoch)\t%d" % best_model_s,'corr: {:.6f}'.format(max_corr_spearman))
     

     artifact = wandb.Artifact("Last_model",type="model")
     artifact.add_file(model_dir + '/last_model.pt')
     run.log_artifact(artifact)

    #  artifact = wandb.Artifact("Loss_model",type="model")
    #  artifact.add_file(model_dir + '/best_model_l.pt')
    #  run.log_artifact(artifact)

    #  artifact = wandb.Artifact("Pearson_model",type="model")
    #  artifact.add_file(model_dir + '/best_model_p.pt')
    #  run.log_artifact(artifact)

    #  artifact = wandb.Artifact("Spearman_model",type="model")
    #  artifact.add_file(model_dir + '/best_model_s.pt')
    #  run.log_artifact(artifact)
     
     artifact = wandb.Artifact("Per_drug_model",type="model")
     artifact.add_file(model_dir + '/best_model_d.pt')
     run.log_artifact(artifact)


     time_elapsed = time.time() - since
     print('\nTraining complete in {:.0f}m {:.0f}s'.format(
         time_elapsed // 60, time_elapsed % 60))


def predict(statistic,run,criterion,predict_data, gene_dim, drug_dim, model_file, batch_size, result_file, cell_features, drug_features, device):
    model = torch.load(model_file)
    model.to(device)

    predict_feature, predict_label = predict_data

    predict_label_device = predict_label.to(device, non_blocking=True)

    model.eval()

    test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)
    test_cum_loss = 0
    #Test
    test_predict = torch.zeros(0, 1, device=device)
    with torch.no_grad():
        for i, (inputdata, labels) in enumerate(test_loader):
            # Convert torch tensor to Variable
            features = build_input_vector(inputdata, cell_features, drug_features)
            device_features = features.to(device)

            # make prediction for test data
            out = model(device_features)
            test_predict = torch.cat([test_predict, out])

    test_cum_loss=criterion(test_predict, predict_label_device)
    test_corr = pearson_corr(test_predict, predict_label_device)
    test_corr_spearman = spearman_corr(test_predict.cpu().detach().numpy(), predict_label_device.cpu())
    test_corr_per_drug = per_drug_corr(predict_feature[:,1], test_predict, predict_label_device)
    test_corr_low = low_corr(test_predict, predict_label_device,-0.1)

    print('Test loss: {:.4f}'.format(test_cum_loss))
    print('Test Corr (p): {:.4f}'.format(test_corr))
    print('Test Corr (s): {:.4f}'.format(test_corr_spearman))
    print('Per drug corr.: {:.4f}'.format(test_corr_per_drug))
    print('Low corr.: {:.4f}'.format(test_corr_low))

    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')

    wandb.log({"pearson_test_"+statistic: test_corr,
               "spearman_test_"+statistic: test_corr_spearman,
                "low_test_"+statistic: test_corr_low,
               "loss_test_"+statistic: test_cum_loss})

    np.savetxt(result_file+statistic+'_test_predictions.txt', test_predict.cpu().detach().numpy(),'%.5e')
    # artifact2 = wandb.Artifact(statistic+"_predictions",type="predictions")
    # artifact2.add_file(result_file+statistic+'_test_predictions.txt')
    # run.log_artifact(artifact2)
    
def load_select_data(file_name, cell2id, drug2id,selected_drug): # only selected samples of chosen drug 
    feature = []
    label = []

    with open(file_name, 'r') as fi: # ,encoding='UTF-16'
        for line in fi:
            tokens = line.strip().split('\t')
            if tokens[1] == selected_drug:
                feature.append([cell2id[tokens[0]], drug2id[tokens[1]]])
                label.append([float(tokens[2])])
    return feature, label

def get_compound_names(file_name):
    compounds = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            compounds.append([tokens[1],tokens[2]])

    return compounds

def predict_drug(predict_data, model, batch_size, cell_features, drug_features,device):

    predict_feature, predict_label = predict_data

    predict_label_device = predict_label.to(device, non_blocking=True).detach()

    model.eval()

    test_loader = du.DataLoader(du.TensorDataset(predict_feature,predict_label), batch_size=batch_size, shuffle=False)

    #Test
    test_predict = torch.zeros(0, 1, device=device)
    with torch.no_grad():
        for i, (inputdata, labels) in enumerate(test_loader):
            features = build_input_vector(inputdata, cell_features, drug_features)
            features = features.to(device)

            # make prediction for test data
            out = model(features)

            test_predict = torch.cat([test_predict, out])

    test_corr = pearson_corr(test_predict, predict_label_device)
    test_corr_spearman = spearman_corr(test_predict.cpu().detach().numpy(), predict_label_device.cpu())

    return test_corr, test_corr_spearman

# Configure the sweep – specify the parameters to search through, the search strategy, the optimization metric et all.
# sweep_config = {
#     'method': 'bayes', #bayes, random, grid

#     'metric': {
#       'name': 'corr_per_drug_val',
#       'goal': 'maximize'
#     },

#     'parameters': {
#         'epochs': {
#             'value': opt.epoch
#         },
#         'batch_size': {
#             'values': [1000,5000,10000,15000,25000] # no poner mas de 25000
#         },
#         'learning_rate': {
#             'values': [0.2,0.3,0.4,0.5]
#         },
#         'optimizer': {
#             'values': ['sgd']
#         },
#         'decay_rate': {
#             'values': [0.001,0.0001]
#         },
#         'criterion': {
#             'value': 'MSELoss'
#         },
#         'momentum': {
#             'values':[0.85,0.91,0.88,0.8]
#         },
#         'num_neurons_per_GO': {
#             'values': [5,6,7,8]
#         },
#         'num_neurons_final_GO': {
#             'values': [6,12,24]
#         },
#         'drug_neurons': {
#             #'value': list(map(int, opt.drug_neurons.split(',')))
#             'values': [list(map(int, '100,50,6'.split(','))),list(map(int, '100,30'.split(','))),list(map(int, '100,50,25'.split(','))),list(map(int, '50,100,50'.split(',')))] 
#         },
#         'num_neurons_final': {
#             'values': [12,18,16,20]
#         },
#         'p_drop_genes': {
#             'values': [0,0.05,0.1]
#         },
#         'p_drop_terms': {
#             'values': [0,0.05,0.1]
#         },
#         'p_drop_drugs': {
#             'values': [0,0.05,0.1]
#         },
#         'p_drop_final': {
#             'values': [0,0.05]
#         },
#         # 'p_drop_genes': {
#         #     'value': 0
#         # },
#         # 'p_drop_terms': {
#         #     'value': 0
#         # },
#         # 'p_drop_drugs': {
#         #     'value': 0
#         # },
#         # 'p_drop_final': {
#         #     'value': 0
#         # },       

#     }
# }
sweep_config = {
    'method': 'bayes', #bayes, random, grid

    'metric': {
      'name': 'corr_per_drug_val',
      'goal': 'maximize'
    },

    'parameters': {
        'epochs': {
            'value': opt.epoch
        },
        'batch_size': {
            'values': [10000]
        },
        'learning_rate': {
            'value': 0.5
        },
        'optimizer': {
            'value': 'sgd'
        },
        'decay_rate': {
            'value': 0.0001
        },
        'criterion': {
            'value': 'MSELoss'
        },
        'momentum': {
            'value':0.88
        },
        'num_neurons_per_GO': {
            #'value': opt.num_neurons_per_GO
            'value': 7
        },
        'num_neurons_final_GO': {
            #'value': opt.num_neurons_final_GO
            'value': 24
        },
        'drug_neurons': {
            #'value': list(map(int, opt.drug_neurons.split(',')))
            'value':list(map(int, '100,50,25'.split(',')))
        },
        'num_neurons_final': {
            'value': 20
        },
        'p_drop_genes': {
            'value': 0.1
        },
        'p_drop_terms': {
            'value': 0.05
        },
        'p_drop_drugs': {
            'value': 0
        },
        'p_drop_final': {
            'value': 0
        },

    }
}

# Initialize a new sweep
# Arguments:
#     – sweep_config: the sweep config dictionary defined above
#     – entity: Set the username for the sweep
#     – project: Set the project name for the sweep
sweep_id = wandb.sweep(sweep_config, entity="katynasada", project=opt.project)

# Load ontology: create the graph of connected GO terms
since1 = time.time()
gene2id_mapping = load_mapping(opt.gene2id)
dG, terms_pairs, genes_terms_pairs = load_ontology(opt.onto, gene2id_mapping)
time_elapsed1 = time.time() - since1
print('Load ontology complete in {:.0f}m {:.0f}s'.format(
 time_elapsed1 // 60, time_elapsed1 % 60))
####

# Layer connections contains the pairs on each layer (including virtual nodes)
since2 = time.time()
sorted_pairs, level_list, level_number = sort_pairs(genes_terms_pairs, terms_pairs, dG, gene2id_mapping)
layer_connections = pairs_in_layers(sorted_pairs, level_list, level_number)
time_elapsed2 = time.time() - since2
print('\nLayer connections complete in {:.0f}m {:.0f}s'.format(
 time_elapsed2 // 60, time_elapsed2 % 60))
####

# Load cell/drug features
cell_features = np.genfromtxt(opt.genotype, delimiter=',')
drug_features = np.genfromtxt(opt.fingerprint, delimiter=',')
drug_dim = len(drug_features[0,:])


since4 = time.time()
train_data = prepare_train_data(opt.train, opt.val, opt.cell2id, opt.drug2id)
time_elapsed4 = time.time() - since4
print('\nTrain data was ready in {:.0f}m {:.0f}s'.format(
 time_elapsed4 // 60, time_elapsed4 % 60))
####

# PREDICT/TEST DATA!
predict_data, cell2id_mapping, drug2id_mapping = prepare_predict_data(opt.predict, opt.cell2id, opt.drug2id)

num_cells = len(cell2id_mapping)
num_drugs = len(drug2id_mapping)
num_genes = len(gene2id_mapping)

def pipeline():
    # Initialize a new wandb run
    run = wandb.init(settings=wandb.Settings(start_method="thread"),name=opt.sweep_name, tags=["fingerprint"])

    config = wandb.config # config is a variable that holds and saves hyperparameters and inputs
    
    # Add notes to the run
    #wandb.run.notes = ""

    ## Training
    since3 = time.time()
    model = sparseGO_nn(layer_connections, config.num_neurons_per_GO, config.num_neurons_final_GO, config.drug_neurons, config.num_neurons_final, drug_dim, gene2id_mapping,config.p_drop_final,config.p_drop_genes,config.p_drop_terms,config.p_drop_drugs)
    # xavier_initialize(model)
    time_elapsed3 = time.time() - since3
    print('\nModel created in {:.0f}m {:.0f}s'.format(
     time_elapsed3 // 60, time_elapsed3 % 60))
    ####

    device = torch.device(f"cuda:{opt.cuda_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!") # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    model.to(device)

    if config.optimizer=='sgd':
        momentum=config.momentum
        print("Momentum: ", momentum)
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=momentum)
    elif config.optimizer=='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.99), eps=1e-05)

    if config.criterion=='MSELoss':
        criterion = nn.MSELoss()
        test_model = '/best_model_d.pt' # Test model is spearman
    elif config.criterion=='L1Loss':
        criterion = nn.L1Loss()
        test_model = '/best_model_p.pt' # Test model is pearson

    batch_size = config.batch_size
    decay_rate = config.decay_rate

    print("Decay rate: ", decay_rate)

    train_model(run,config, model, optimizer, criterion, train_data, cell_features, drug_features, batch_size, opt.modeldir, device, config.epochs, decay_rate)

    since_test = time.time()
    predict("ModelSpearman",run, criterion, predict_data, num_genes, drug_dim, opt.modeldir + '/best_model_s.pt', batch_size, opt.result, cell_features, drug_features, device)
    predict("ModelLoss",run, criterion, predict_data, num_genes, drug_dim, opt.modeldir + '/best_model_l.pt', batch_size, opt.result, cell_features, drug_features, device)
    predict("ModelPearson",run, criterion, predict_data, num_genes, drug_dim, opt.modeldir + '/best_model_p.pt', batch_size, opt.result, cell_features, drug_features, device)
    predict("ModelPerDrug",run, criterion, predict_data, num_genes, drug_dim, opt.modeldir + '/best_model_d.pt', batch_size, opt.result, cell_features, drug_features, device)

    # -------------------------
    # PREDICTIONS PER DRUG
    names = get_compound_names(opt.drugs_names)
    names.pop(0)
    model = torch.load(opt.modeldir + test_model, map_location=device)
    predictions = {}
    predictions_spearman = {}
    for selected_drug_data in names:
        selected_drug = selected_drug_data[0]
        features, labels = load_select_data(opt.predict, cell2id_mapping, drug2id_mapping,selected_drug)
        predict_data_drug = (torch.Tensor(features), torch.FloatTensor(labels))
        pearson, spearman = predict_drug(predict_data_drug, model, batch_size, cell_features, drug_features,device)
        predictions[selected_drug_data[1]]=pearson.item()
        predictions_spearman[selected_drug_data[1]]=spearman.item()
        print(selected_drug_data[1]+" "+str(spearman.item()))
    
    predictions_dataframe_sp = pd.DataFrame(predictions_spearman.items(),columns=['Name', 'Spearman'])
    predictions_dataframe_per = pd.DataFrame(predictions.items(),columns=['Name', 'Pearson'])

    mean_spearman = predictions_dataframe_sp["Spearman"].mean()
    print("Average spearman correlations of all drugs: ",mean_spearman)
    mean_pearson = predictions_dataframe_per["Pearson"].mean()
    print("Average pearson correlations of all drugs: ",mean_pearson)

    wandb.log({"av_pearson_drugs_spmodel": mean_pearson,
               "av_spearman_drugs_spmodel": mean_spearman})
    # -------------------------

    time_elapsed_test = time.time() - since_test
    print('\nTest complete in {:.0f}m {:.0f}s'.format(
    time_elapsed_test // 60, time_elapsed_test % 60))

    time_elapsed0 = time.time() - since0
    print('\nTotal run time {:.0f}m {:.0f}s'.format(
        time_elapsed0 // 60, time_elapsed0 % 60))

    wandb.log({"time": '{:.0f}m {:.0f}s'.format(time_elapsed0 // 60, time_elapsed0 % 60)})

# RUUUUUUUN
wandb.agent(sweep_id, pipeline,count=1)

####
