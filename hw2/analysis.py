# this file will be used to analyze the data from the h5py data files located in the data folder

import h5py
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.colors as colors

import os
import cv2

# function to read the data from the h5py file
def read_data(file_name):
    # open the file
    file = h5py.File(file_name, 'r')
    # get the data called "average_fitness"
    average_fitness = file['average_fitness'][:]
    # get the data called "best_fitness"
    best_fitness = file['best_fitness'][:]
    # get the data called "image_-1000"
    image_0 = file['image_-1000'][:]
    # get the data called "image_-999"
    image_1 = file['image_-999'][:]
    # get the data called "image_-998"
    image_2 = file['image_-998'][:]
    # get the data called "image_-997"
    image_3 = file['image_-997'][:]
    # get the data called "image_-996"
    image_4 = file['image_-996'][:]
    # get the data called "image_-995"
    image_5 = file['image_-995'][:]
    # get the data called "image_-994"
    image_6 = file['image_-994'][:]
    # get the data called "image_-993"
    image_7 = file['image_-993'][:]
    # get the data called "image_-992"
    image_8 = file['image_-992'][:]
    # get the data called "image_-991"
    image_9 = file['image_-991'][:]
    # put them into the data array
    data = [image_0, image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9]

    # return the data and labels
    return data, average_fitness, best_fitness

# get list of files with h5 extension in the data folder
data_folder = "hw2/out_sweep/" # 
files = os.listdir(data_folder)
files = [file for file in files if file.endswith(".h5")]


# loop through the files and read the data
for file in files:
    file = "outputFRAC_20_50_5_0.2_0.6_0.2_guided.h5"
    data, average_fitness, best_fitness = read_data(data_folder + "/" + file)
    # get the paremeter values from the file name
    parts = file.split("_")

    # get the population size
    num_individuals = parts[1]
    # get the number og genes
    num_genes = parts[2]
    # get the tournament size
    tournament_size = parts[3]
    # get the fraction of elites
    fraction_elites = parts[4]
    # get the fraction of parents
    fraction_parents = parts[5]
    # get the mutation probability
    mutation_probability = parts[6]
    # get the mutation type
    mutation_type = parts[7].split(".")[0]

    # create the text info
    text_info = "Population Size: " + num_individuals + "\nNumber of Genes: " + num_genes + "\nTournament Size: " + tournament_size + "\nFraction of Elites: " + fraction_elites + "\nFraction of Parents: " + fraction_parents + "\nMutation Probability: " + mutation_probability + "\nMutation Type: " + mutation_type
    # plot the average fitness
    # Plot the data on A5 paper use Arial font. Use 4:3 aspect ratio.
    fig = plt.figure(figsize=(8.3, 5.8))
    gs = gridspec.GridSpec(1, 1)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(average_fitness, linestyle='-', color='#5F8670', linewidth=3.0, alpha=1)
    ax0.set_xlabel('# Generations', fontsize=18, labelpad=10)
    ax0.set_ylabel('Fitness', fontsize=18, labelpad=10)
    ax0.grid()
    ax0.set_title('Average Fitness', fontsize=20, pad=20)
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.text(0.985, 0.225, text_info + "\n Final Score: " + str(average_fitness[-1]), horizontalalignment='right', verticalalignment='center', transform=ax0.transAxes, fontsize=14, fontweight='bold', color='black', alpha=0.75, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))
    plt.tight_layout()
    #plt.show()
    # save the figure
    fig.savefig("hw2/out_sweep/average_fitness_" + file.split(".h")[0] + ".png")

    # plot the best fitness
    # Plot the data on A5 paper use Arial font. Use 4:3 aspect ratio.
    fig = plt.figure(figsize=(8.3, 5.8))
    gs = gridspec.GridSpec(1, 1)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(best_fitness, linestyle='-', color='#B80000', linewidth=3.0, alpha=1)
    ax0.set_xlabel('# Generations', fontsize=18, labelpad=10)
    ax0.set_ylabel('Fitness', fontsize=18, labelpad=10)
    ax0.grid()
    ax0.set_title('Best Fitness', fontsize=20, pad=20)
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.text(0.985, 0.225, text_info + "\n Final Score: " + str(best_fitness[-1]), horizontalalignment='right', verticalalignment='center', transform=ax0.transAxes, fontsize=14, fontweight='bold', color='black', alpha=0.75, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))
    plt.tight_layout()
    # plt.show()
    fig.savefig("hw2/out_sweep/best_fitness_" + file.split(".h")[0] + ".png")


    # plot the best fitness
    # Plot the data on A5 paper use Arial font. Use 4:3 aspect ratio.
    fig = plt.figure(figsize=(8.3, 5.8))
    gs = gridspec.GridSpec(1, 1)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(range(1000, 10000), best_fitness[1000:], linestyle='-', color='#7570b3', linewidth=3.0, alpha=1)
    ax0.set_xlabel('# Generations', fontsize=18, labelpad=10)
    ax0.set_ylabel('Fitness', fontsize=18, labelpad=10)
    ax0.grid()
    ax0.set_title('Best Fitness from 1000th Gen', fontsize=20, pad=20)
    ax0.tick_params(axis='both', which='major', labelsize=16)
    ax0.text(0.985, 0.225, text_info + "\n Final Score: " + str(best_fitness[-1]), horizontalalignment='right', verticalalignment='center', transform=ax0.transAxes, fontsize=14, fontweight='bold', color='black', alpha=0.75, bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.1'))
    plt.tight_layout()
    # plt.show()
    fig.savefig("hw2/out_sweep/best_fitness_1000_" + file.split(".h")[0] + ".png")


    # plot the images for the 10 generations
    fig, axs = plt.subplots(2, 5, figsize=(8.3, 5.8))
    fig.suptitle("Evolution of Images", fontsize=20)
    for i in range(10):
        ax = axs[i//5, i%5]
        # convert the image BGR to RGB 
        data[i] = cv2.cvtColor(data[i], cv2.COLOR_BGR2RGB)

        ax.imshow(data[i])
        ax.set_title("#Gen" + str((i+1)*1000), fontsize=12, fontweight='bold')
        ax.axis('off')
    # add text to the figure
    fig.text(0.9, 0.845, text_info, ha='right', fontsize=8, fontweight='bold', color='black', alpha=0.75)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.show()
    fig.savefig("hw2/out_sweep/images_" + file.split(".h")[0] + ".png")

    print("Done")
