import sys
import torch
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import numpy as np
import pandas as pd
import shutil
import scipy.stats as ss # for spearman_corr
import torch.nn as nn
import random

random.seed(0)
torch.seed()
np.random.seed(0)

# The next block codes between "###" are performed outside the main code. 
# We are normalizing-scaling the input data of cell lines and PDC expression using a quantile normalization 
# calculating the median of the CL expressions and doing rankings) and standard normalization.

def read_data_cl_pdc():
    """
    This function reads the gene expression data for cell lines and patient-derived cells from text files.
    
    Returns:
    tuple: Two DataFrames containing the gene expression data for cell lines (df_cl_genes) and 
           patient-derived cells (df_pdc_genes).

    The data is read from the following files:
    - Cell line gene expression data from 'cell2expression.txt'
    - Patient-derived cell gene expression data from 'cell2expression_PDCs2018.txt'
    - Gene names from 'gene2ind.txt'

    The function also assigns appropriate column names (gene names) and index names (sample identifiers).
    """
    df_cl_genes = pd.read_csv('../data/CL_PDCs2018_expression/allsamples/cell2expression.txt', sep=',', header=None)
    df_pdc_genes = pd.read_csv('../data/PDCs2018_expression_LELO/samples1/cell2expression_PDCs2018.txt', sep=',', header=None)
    gene_names = pd.read_csv('../data/CL_PDCs2018_expression/allsamples/gene2ind.txt', sep="\t", header=None)[1].values.tolist()
    
    df_cl_genes.columns = gene_names
    df_cl_genes.index = ['cl' + str(i+1) for i in range(len(df_cl_genes.index))]
    df_pdc_genes.columns = gene_names
    df_pdc_genes.index = ['p' + str(i+1) for i in range(len(df_pdc_genes.index))]
    return df_cl_genes, df_pdc_genes


def do_transformation(df_orig, df_median, gene_names):
    """
    This function transforms the original gene expression data to match the quantile normalized distribution.

    Parameters:
    df_orig (DataFrame): Original gene expression data.
    df_median (DataFrame): DataFrame containing median values and ranks.
    gene_names (list): List of gene names.

    Returns:
    DataFrame: Transformed gene expression data.
    
    The transformation process involves:
    1. Sorting each row of the original data in descending order.
    2. Ranking the sorted values.
    3. Merging the ranked values with the median ranks.
    4. Replacing original values with the corresponding median values.
    """
    df_genes_norm = pd.DataFrame(columns=gene_names)
    for i in range(0, df_orig.shape[0]):
        # Select row i of the original data
        aux = df_orig.iloc[i, :].sort_values(ascending=False)

        # Create DataFrame with original values and ranks
        df_aux = pd.DataFrame({'Origin_Value': aux.values, 'Rank': range(0, len(aux))}, index=aux.index)
        df_aux = df_aux.sort_index() 

        # Add gene column to median and auxiliary dataframes
        df_median['genes'] = df_median.index
        df_aux['genes'] = df_aux.index
        df_median.reset_index(drop=True, inplace=True)
        df_aux.reset_index(drop=True, inplace=True)

        # Merge based on Rank
        df_merge = pd.merge(df_aux, df_median, on='Rank', suffixes=('_orig', '_median'))

        # Assign the median values to the normalized data
        df_merge = df_merge[['genes_orig', 'Median']]
        df_merge.set_index('genes_orig', inplace=True)

        # Store the normalized values in the result DataFrame
        df_genes_norm.loc[i] = df_merge.values.flatten()
    return df_genes_norm

def quantile_normalization():
    """
    This function performs quantile normalization on gene expression data from two dataframes: 
    one containing cell line gene expression data (df_cl_genes) and another containing patient-derived 
    cell gene expression data (df_pdc_genes). The steps involved are:

    1. Read data from the files using the read_data_cl_pdc() function.
    2. Calculate the median expression value for each gene across all cell lines.
    3. Rank these median values in descending order and create a DataFrame to hold these values along with their ranks.
    4. Normalize the patient-derived cell gene data and the cell line gene data by transforming them using the do_transformation() function.
    5. Save the normalized data to CSV files.

    The quantile normalization ensures that the distribution of gene expression values is the same for both the cell lines and the patient-derived cells.
    """
    df_cl_genes, df_pdc_genes = read_data_cl_pdc()
    gene_names = df_cl_genes.columns.values.tolist()

    # Calculate median and rank the genes
    median_genes = df_cl_genes.median(axis=0)
    median_genes = median_genes.sort_values(ascending=False)

    # Create DataFrame with genes ordered by median values and their ranks
    df_cl_genes_median = pd.DataFrame({'Median': median_genes.values, 'Rank': range(0, len(median_genes))}, index=median_genes.index)
    df_cl_genes_median = df_cl_genes_median.sort_index() 

    # Transform the original dataframes using the calculated medians
    df_pdc_genes_norm = do_transformation(df_orig=df_pdc_genes, df_median=df_cl_genes_median.copy(), gene_names=gene_names)
    df_cl_genes_norm = do_transformation(df_orig=df_cl_genes, df_median=df_cl_genes_median.copy(), gene_names=gene_names)

    # Save normalized data to CSV files
    df_pdc_genes_norm.to_csv('cell2expression_pdc_genes_norm.txt', index=False, header=False, sep=',')
    df_cl_genes_norm.to_csv('cell2expression_cl_genes_norm.txt', index=False, header=False, sep=',')

def standard_scale():
    """
    This function standardizes the gene expression data from two DataFrames: 
    one containing cell line gene expression data (df_cl_genes) and the other containing patient-derived cell 
    gene expression data (df_pdc_genes).
    The standardization is done using the mean and standard deviation calculated from each DataFrame independently.
    The resulting DataFrames are saved to CSV files.
    """
    df_cl_genes, df_pdc_genes = read_data_cl_pdc()

    # Calculate the mean and standard deviation of df_cl_genes
    mean_cl_genes = df_cl_genes.mean()
    std_cl_genes = df_cl_genes.std()

    # Calculate the mean and standard deviation of df_pdc_genes
    mean_pdc_genes = df_pdc_genes.mean()
    std_pdc_genes = df_pdc_genes.std()

    # Standardize df_cl_genes using its own mean and standard deviation
    df_cl_genes_scaled = (df_cl_genes - mean_cl_genes) / std_cl_genes
    # Standardize df_pdc_genes using its own mean and standard deviation
    df_pdc_genes_scaled = (df_pdc_genes - mean_pdc_genes) / std_pdc_genes

    # Save the standardized DataFrames to CSV files
    df_cl_genes_scaled.to_csv('../data/CL_PDCs2018_expression/allsamples/cell2expression_scaled.txt', index=False, header=False, sep=',')
    for i in ['allsamples', 'samples1', 'samples2', 'samples3', 'samples4', 'samples5']:
        df_pdc_genes_scaled.to_csv(f'../data/PDCs2018_expression_LELO/{i}/cell2expression_scaled.txt', index=False, header=False, sep=',')


################################################################

def load_mapping(mapping_file):
    """
    Opens a txt file with two columns and saves the second column as the key of the dictionary and the first column as a value.

        Parameters
        ----------
        mapping_file: str, path to txt file

        Output
        ------
        mapping: dic

        Notes: used to read gene2ind.txt, drug2ind.txt

    """
    mapping = {} # dictionary of values on required txt

    file_handle = open(mapping_file) # function opens a file, and returns it as a file object.

    for line in file_handle:
        line = line.rstrip().split() # quitar espacios al final del string y luego separar cada elemento (en gene2ind hay dos elementos 3007	ZMYND8, los pone en una lista ['3007', 'ZMYND8'] )
        mapping[line[1]] = int(line[0]) # en gene2ind el nombre del gen es el key del dictionary y el indice el valor del diccionario

    file_handle.close()

    return mapping

def load_ontology(ontology_file, gene2id_mapping):
    """
    Creates the directed graph of the GO terms and stores the connected elements in arrays.

        Output
        ------
        dG: networkx.classes.digraph.DiGraph
            Directed graph of all terms

        terms_pairs: numpy.ndarray
            Store the connection between a term and a term

        genes_terms_pairs: numpy.ndarray
            Store the connection between a gene and a term
    """

    dG = nx.DiGraph() # Directed graph class

    file_handle = open(ontology_file) #  Open the file that has genes and go terms

    terms_pairs = [] # store the pairs between a term and a term
    genes_terms_pairs = [] # store the pairs between a gene and a term

    gene_set = set() # create a set (elements can't repeat)
    term_direct_gene_map = {}
    term_size_map = {}


    for line in file_handle:

        line = line.rstrip().split() # delete spaces and transform to list, line has 3 elements

        # No me hace falta el if, no tengo que separar las parejas
        if line[2] == 'default': # si el tercer elemento es default entonces se conectan los terms en el grafo
            dG.add_edge(line[0], line[1]) # Add an edge between line[0] and line[1]
            terms_pairs.append([line[0], line[1]]) # Add the pair to the list
        else:
            if line[1] not in gene2id_mapping: # se salta el gen si no es parte de los que estan en gene2id_mapping
                print(line[1])
                continue

            genes_terms_pairs.append([line[0], line[1]]) # add the pair

            if line[0] not in term_direct_gene_map: # si el termino todavia no esta en el diccionario lo agrega
                term_direct_gene_map[ line[0] ] = set() # crea un set

            term_direct_gene_map[line[0]].add(gene2id_mapping[line[1]]) # añadimos el gen al set de ese term

            gene_set.add(line[1]) # añadimos el gen al set total de genes

    terms_pairs = np.array(terms_pairs) # convert to 2d array
    genes_terms_pairs = np.array(genes_terms_pairs) # convert to 2d array

    file_handle.close()

    print('There are', len(gene_set), 'genes')

    for term in dG.nodes(): # hacemos esto para cada uno de los GO terms

        term_gene_set = set() # se crea un set

        if term in term_direct_gene_map:
            term_gene_set = term_direct_gene_map[term] # genes conectados al term

        deslist = nxadag.descendants(dG, term) #regresa todos sus GO terms descendientes (biological processes tiene 2085 descendientes, todos menos el mismo)

        for child in deslist:
            if child in term_direct_gene_map: # añadir los genes de sus descendientes
                term_gene_set = term_gene_set | term_direct_gene_map[child] # union of both sets, ahora tiene todos los genes los suyos y los de sus descendientes

        if len(term_gene_set) == 0:
            print('There is empty terms, please delete term:', term)
            sys.exit(1)
        else:
            # por ahora esta variable no me hace falta
            term_size_map[term] = len(term_gene_set) # cantidad de genes en ese term  (tomando en cuenta sus descendientes)

    leaves = [n for n in dG.nodes if dG.in_degree(n) == 0] # buscar la raiz
    #leaves = [n for n,d in dG.in_degree() if d==0]

    uG = dG.to_undirected() # Returns an undirected representation of the digraph
    connected_subG_list = list(nxacc.connected_components(uG)) #list of all GO terms

    # Verify my graph makes sense...
    print('There are', len(leaves), 'roots:', leaves[0])
    print('There are', len(dG.nodes()), 'terms')
    print('There are', len(connected_subG_list), 'connected components')
    if len(leaves) > 1:
        print('There are more than 1 root of ontology. Please use only one root.')
        sys.exit(1)
    if len(connected_subG_list) > 1:
        print( 'There are more than connected components. Please connect them.')
        sys.exit(1)

    return dG, terms_pairs, genes_terms_pairs

def sort_pairs(genes_terms_pairs, terms_pairs, dG, gene2id_mapping):
    """
    Function concatenates the pairs and orders them, the parent term goes first.

        Output
        ------
        level_list: list
            Each array of the list stores the elements on a level of the hierarchy

        level_number: dict
            Has the gene and GO terms with their corresponding level number

        sorted_pairs: numpy.ndarray
            Contains the term-gene or term-term pairs with the parent element on the first column
    """

    all_pairs = np.concatenate((genes_terms_pairs,terms_pairs))
    graph = dG.copy() #  Copy the graph to avoid modifying the original

    level_list = []   # level_list stores the elements on each level of the hierarchy
    level_list.append(list(gene2id_mapping.keys())) # add the genes

    while True:
        leaves = [n for n in graph.nodes() if graph.out_degree(n) == 0]

        if len(leaves) == 0:
            break

        level_list.append(leaves) # add the terms on each level
        graph.remove_nodes_from(leaves)

    level_number = {} # Has the gene and GO terms with their corresponding level number
    for i, layer in enumerate(level_list):
        for _,item in enumerate(layer):
            level_number[item] = i

    sorted_pairs = all_pairs.copy() # order pairs based on their level
    for i, pair in enumerate(sorted_pairs):
        level1 = level_number[pair[0]]
        level2 = level_number[pair[1]]
        if level2 > level1:  # the parent term goes first
            sorted_pairs[i][1] = all_pairs[i][0]
            sorted_pairs[i][0] = all_pairs[i][1]

    return sorted_pairs, level_list, level_number


def pairs_in_layers(sorted_pairs, level_list, level_number):
    """
    This function divides all the pairs of GO terms and genes by layers and adds the virtual nodes

        Output
        ------
        layer_connections: numpy.ndarray
            Contains the pairs that will be part of each layer of the model.
            Not all terms are connected to a term on the level above it. "Virtual nodes" are added to establish the connections between non-subsequent levels.

    """
    total_layers = len(level_list)-1 # Number of layers that the model will contain
    # Will contain the GO terms connections and gene-term connections by layers
    layer_connections = [[] for i in range(total_layers)]

    for i, pair in enumerate(sorted_pairs):
       parent = level_number[pair[0]]
       child = level_number[pair[1]]

       # Add the pair to its corresponding layer
       layer_connections[child].append(pair)

       # If the pair is not directly connected virtual nodes have to be added
       dif = parent-child # number of levels in between
       if dif!=1:
          virtual_node_layer = parent-1
          for j in range(dif-1): # Add the necessary virtual nodes
              layer_connections[virtual_node_layer].append([pair[0],pair[0]])
              virtual_node_layer = virtual_node_layer-1

    # Delete pairs that are duplicated (added twice on the above step)
    for i,_ in enumerate(layer_connections):
        layer_connections[i] = np.array(layer_connections[i]) # change list to array
        layer_connections[i] = np.unique(layer_connections[i], axis=0)

    return layer_connections

def create_index(array):
    unique_array = pd.unique(array)

    index = {}
    for i, element in enumerate(unique_array):
        index[element] = i

    return index

def load_train_data(file_name, cell2id_mapping, drug2id_mapping):
    feature = []
    label = []

    with open(file_name, 'r') as fi:
        for line in fi:
            tokens = line.strip().split('\t')
            
            if len(tokens)==3:
                feature.append([cell2id_mapping[tokens[0]], drug2id_mapping[tokens[1]]])
                label.append([float(tokens[2])])
            elif len(tokens)==4: # if there are 2 output neurons in the network 
                feature.append([cell2id_mapping[tokens[0]], drug2id_mapping[tokens[1]]])
                label.append([float(tokens[2]), float(tokens[3])]) # keep both labels

    return feature, label

def prepare_train_data(train_file, test_file, cell2id_mapping_file, drug2id_mapping_file):

    # load mapping files
    cell2id_mapping = load_mapping(cell2id_mapping_file)
    drug2id_mapping = load_mapping(drug2id_mapping_file)

    train_feature, train_label = load_train_data(train_file, cell2id_mapping, drug2id_mapping)
    test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

    print('\nTotal number of cell lines = %d' % len(cell2id_mapping))
    print('Total number of drugs = %d' % len(drug2id_mapping))

    return (torch.Tensor(train_feature), torch.FloatTensor(train_label), torch.Tensor(test_feature), torch.FloatTensor(test_label))

def build_input_vector(inputdata, cell_features, drug_features):
    # For training
    genedim = len(cell_features[0,:])
    drugdim = len(drug_features[0,:])
    feature = np.zeros((inputdata.size()[0], (genedim+drugdim)))

    for i in range(inputdata.size()[0]):
        feature[i] = np.concatenate((cell_features[int(inputdata[i,0])], drug_features[int(inputdata[i,1])]), axis=None)

    feature = torch.from_numpy(feature).float()
    return feature

def pearson_corr(x, y): # comprobado que esta bien (con R)
    xx = x - torch.mean(x)
    yy = y - torch.mean(y)

    return torch.sum(xx*yy) / (torch.norm(xx, 2)*torch.norm(yy,2))

def spearman_corr(x, y): # comprobado que esta bien (con R)
    spearman = pearson_corr(torch.from_numpy(ss.rankdata(x)*1.),torch.from_numpy(ss.rankdata(y)*1.))
    # train_corr_spearman = spearman_corr(train_predict.cpu().detach().numpy(), train_label_gpu.cpu()) # con gpu hay que detach
    return spearman

def prepare_predict_data(test_file, cell2id_mapping_file, drug2id_mapping_file):

	# load mapping files
	cell2id_mapping = load_mapping(cell2id_mapping_file)
	drug2id_mapping = load_mapping(drug2id_mapping_file)

	test_feature, test_label = load_train_data(test_file, cell2id_mapping, drug2id_mapping)

	return (torch.Tensor(test_feature), torch.Tensor(test_label)), cell2id_mapping, drug2id_mapping

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir +"/checkpoint.pt"
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_dir + "/best_model.pt"
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def per_drug_loss(criterion, drugs_ids, predictions, labels):
    """
    Calculate the mean squared loss per unique drug.

    Parameters:
    - drugs_ids (torch.Tensor): Tensor containing drug IDs for each prediction-label pair.
    - predictions (torch.Tensor): Tensor containing predicted values for each prediction-label pair.
    - labels (torch.Tensor): Tensor containing actual labels for each prediction-label pair.

    Returns:
    - float: Mean squared loss.
    """

    # Initialize the total loss
    loss = 0

    # Iterate over the unique drug IDs
    for chosen_id in torch.unique(drugs_ids):
        # Get the indices of the labels that have that drug id
        indices = (drugs_ids == chosen_id).nonzero(as_tuple=True)[0]

        # Get the predictions and labels for the corresponding drug
        grouped_tensor_predictions = predictions[indices]
        grouped_tensor_labels = labels[indices]

        # Calculate the mean squared loss for the current drug ID
        current_loss = criterion(grouped_tensor_predictions, grouped_tensor_labels)

        # Add the current loss to the total loss
        loss += current_loss

    # Return the mean squared error loss mean (because maybe we dont always have all drugs)
    return loss / torch.unique(drugs_ids).shape[0]

def per_drug_corr(drugs_ids, predictions, labels):
    """
    Calculate the correlation per drug.

    Parameters:
    - drugs_ids (torch.Tensor): Tensor containing drug IDs for each prediction-label pair.
    - predictions (torch.Tensor): Tensor containing predicted values for each prediction-label pair.
    - labels (torch.Tensor): Tensor containing actual labels for each prediction-label pair.

    Returns:
    - float: Mean squared loss.
    """

    corr = 0 

    # Iterate over the unique drug IDs
    for chosen_id in torch.unique(drugs_ids):
        # Get the indices of the labels that have that drug id
        indices = (drugs_ids == chosen_id).nonzero(as_tuple=True)[0]

        # Get the predictions and labels for the corresponding drug
        grouped_tensor_predictions = predictions[indices]
        grouped_tensor_labels = labels[indices]

        # Calculate the mean squared loss for the current drug ID
        current_accuracy = torch.nan_to_num(spearman_corr(grouped_tensor_predictions.cpu().detach().numpy(), grouped_tensor_labels.cpu()))

        # Add the current loss to the total loss
        corr += float(current_accuracy)

    # Return the corr mean 
    return corr / torch.unique(drugs_ids).shape[0]

def low_corr(predictions, labels, threshold):
    """
    Calculate the correlation for low AUDRC values.

    Parameters:
    - predictions (torch.Tensor): Tensor containing predicted values for each prediction-label pair.
    - labels (torch.Tensor): Tensor containing actual labels for each prediction-label pair.
    - threshold 

    Returns:
    - float: correlation
    """

    # Get the indices of the labels that have that drug id
    indices = (labels < threshold).nonzero(as_tuple=True)[0]

    # Get the predictions and labels for the corresponding drug
    low_tensor_predictions = predictions[indices]
    low_tensor_labels = labels[indices]

    corr = spearman_corr(low_tensor_predictions.cpu().detach().numpy(), low_tensor_labels.cpu())

    return corr 

def weighted_low_loss(predictions, labels):
    """
    Calculate the weighted mean squared loss

    Parameters:
    - predictions (torch.Tensor): Tensor containing predicted values for each prediction-label pair.
    - labels (torch.Tensor): Tensor containing actual labels for each prediction-label pair.

    Returns:
    - float: Mean squared loss.
    """

    # Calculate weights based on the labels (adjust as needed)
    weights = 1.0 / labels  # You can use any function to compute weights based on input values
    
    # Your MSE loss with custom weights
    criterion = nn.MSELoss(reduction='none') # ESTA MAAAL NO SE PUEDEN PONER WEIGHTS ASI 
    
    # Calculate individual losses for each sample
    individual_losses = criterion(predictions,labels)
    
    # Multiply individual losses by weights
    weighted_losses = weights * individual_losses

    return torch.mean(weighted_losses) 

def xavier_initialize(model):
    """
    This function initializes the weights of Linear layers in the model 
    using Xavier uniform initialization, and sets the bias to 0.

    Xavier initialization helps in keeping the scale of gradients roughly the same 
    in all layers, which makes the backpropagation process more stable. 
    It is a good practice when we use activation functions like Tanh or Sigmoid. 
    For ReLU or its variants, He initialization is recommended instead.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be initialized.

    Returns
    -------
    None
    
    """

    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None: 
                torch.nn.init.constant_(layer.bias, 0)