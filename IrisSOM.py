from minisom import MiniSom
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec




def readData(filename):
    print("Reading data ...\n")
    data = np.genfromtxt(filename, delimiter=',', usecols=(0, 1, 2, 3))
    # data normalization
    data = np.apply_along_axis(lambda x: x/np.linalg.norm(x), 1, data)
    return data

def readTarget(filename):
    target = np.genfromtxt(filename, delimiter=',', usecols=(4), dtype=str)
    return target

def mappingTargetWithDecimal(target):
    deciaml_t = np.zeros(len(target), dtype=int)
    deciaml_t[target == 'setosa'] = 0
    deciaml_t[target == 'versicolor'] = 1
    deciaml_t[target == 'virginica'] = 2
    return deciaml_t

def train(data, width_, height_, features, sigma_, learning_rate_, neighborhood_function_, iterations):
    # Initialization and training
    som = MiniSom(width_, height_, features, sigma=sigma_, learning_rate=learning_rate_,
                  neighborhood_function=neighborhood_function_, random_seed=10)
    #som.random_weights_init(data)
    som.pca_weights_init(data)
    print("Training...\n")

    # random training
    som.train_random(data, iterations)
    # batch training
    #som.train_batch(data, iterations)
    print("done!\n")
    return som

def setting_poltting_parameters(target, som, data):
    
    decimal_t = mappingTargetWithDecimal(target)
    labels_map = som.labels_map(data, target)
    label_names = np.unique(target)

    return decimal_t,labels_map,label_names

def plotDataMap(width, height, som, data, decimal_t):
    
    plt.figure(figsize=(width, height))
    # Plotting the response for each pattern in the iris dataset
    # plotting the distance map as background
    plt.pcolor(som.distance_map().T, cmap='bone_r')  
    plt.colorbar()
    
    # use different colors and markers for each label
    markers = ['s', '^', 'v']
    #colors = ['C4', 'C6', 'C7'] #desired colors
    colors = ['C0', 'C1', 'C2']
    for cnt, xx  in enumerate(data):
        w = som.winner(xx)  # getting the winner
        # place a marker on the winning position for the sample xx
        plt.plot(w[0]+.5, w[1]+.5, markers[decimal_t[cnt]], markerfacecolor='None',
                 markeredgecolor=colors[decimal_t[cnt]], markersize=5, markeredgewidth=2)

    plt.axis([0, width, 0, height])
    plt.show()
    

def plotClustering(width, height, labels_map, label_names):
    
    plt.figure(figsize=(width, height))
    the_grid = GridSpec(width, height)
    colors = ['C4', 'C6', 'C7']
    
    for position in labels_map.keys():
        label_fracs = [labels_map[position][l] for l in label_names]
        plt.subplot(the_grid[width-1-position[1], position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)
        
        
    plt.legend(label_names , ncol=3, loc='best')
    
    plt.show()


def calcError(som, data):
    error = som.quantization_error(data)
    return error


def main():
    
    #important variables
    learning_rate_ = 0.5
    sigma_= 4
    neighborhood_function_='gaussian' 
    iterations = 10000
    map_width = 7
    map_height = 7
    graph_width = 7
    graph_height = 7
    features = 4
    
    training_data = readData('Iris_data_training.txt')
    target_training = readTarget('Iris_data_training.txt')
    som = train(training_data, map_width, map_height, features, learning_rate_, sigma_, neighborhood_function_, iterations);
    
     
    #plotting stuff
    decimal_t_training,labels_map,label_names = setting_poltting_parameters(target_training, som, training_data)
    plotDataMap(graph_width, graph_height, som, training_data, decimal_t_training)
    plotClustering(graph_width, graph_height, labels_map, label_names)

    error = calcError(som, training_data)
    print('The error :',error,'\t')

if __name__ == "__main__":
  main()
