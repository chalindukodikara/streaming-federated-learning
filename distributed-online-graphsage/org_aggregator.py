
import socket
import pickle
import select
import time
import numpy as np
import pandas as pd
import sys
import logging
from timeit import default_timer as timer
import gc
import math
import os
import argparse
from multiprocessing.connection import Listener, Client
import warnings
warnings.filterwarnings("ignore")

# CONSTANTS
HEADER_LENGTH = 10
WEIGHT_FILE_PATH = "/var/tmp/jasminegraph-localstore/"

######## Our parameters ################
parser = argparse.ArgumentParser('Client')
parser.add_argument('--path_weights', type=str, default='./local_weights/', help='Weights path to be saved')
parser.add_argument('--path_nodes', type=str, default='./data/', help='Nodes path')
parser.add_argument('--path_edges', type=str, default='./data/', help='Edges Path')
parser.add_argument('--graph_id', type=int, default=1, help='Graph ID')
parser.add_argument('--ip', type=str, default='localhost', help='IP')
parser.add_argument('--port', type=int, default=5250, help='PORT')
parser.add_argument('--partition_algorithm', type=str, default='fennel', help='Partition algorithm')

######## Frequently configured #######
parser.add_argument('--dataset_name', type=str, default='dblp', help='Dataset name')
parser.add_argument('--partition_size', type=int, default=2, help='Partition size')
parser.add_argument('--num_orgs', type=int, default=2, help='Number of organizations')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

WEIGHTS_PATH = args.path_weights
NODES_PATH = args.path_nodes
EDGES_PATH = args.path_edges
GRAPH_ID = args.graph_id
IP = args.ip
PORT = args.port
DATASET_NAME = args.dataset_name
PARTITION_SIZE = args.partition_size
PARTITION_ALGORITHM = args.partition_algorithm
NUM_ORGS = args.num_orgs
######## Our parameters ################

# create data folder with the dataset name
folder_path_agg = "aggregator_weights"
if os.path.exists(folder_path_agg):
    pass
else:
    os.makedirs(folder_path_agg)

folder_path_final = "final_model"
if os.path.exists(folder_path_final):
    pass
else:
    os.makedirs(folder_path_final)


######## Setup logger ################
# create data folder with the dataset name
folder_path_logs = "logs"
if os.path.exists(folder_path_logs):
    pass
else:
    os.makedirs(folder_path_logs)


folder_path_process = folder_path_logs + "/meta_server"
if os.path.exists(folder_path_process):
    pass
else:
    os.makedirs(folder_path_process)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('logs/meta_server/{}_{}_{}_partition_{}_aggregator.log'.format(str(time.strftime('%m %d %H:%M:%S # %l:%M%p on %b %d, %Y')), DATASET_NAME, PARTITION_ALGORITHM, PARTITION_SIZE)),
        logging.StreamHandler(sys.stdout)
    ]
)
############################################
# Find the minimum batch number in partitions
min_timestamps = []
for i in range(NUM_ORGS):
    timestamps = []
    for partition in range(PARTITION_SIZE):
        path = 'data/' + str(i) + '/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(partition)
        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files]
        timestamps.append(int(max(paths, key=os.path.getctime).split('/')[-1].split('_')[0]))
    min_timestamps.append(min(timestamps))
NUM_TIMESTAMPS = min(min_timestamps)

class GlobalAggregator:

    def __init__(self, graph_id, num_orgs):

        # Parameters
        self.NUM_CLIENTS = num_orgs

        # Global model
        self.model_weights = None

        self.weights = []
        self.partition_sizes = []

        self.stop_flag = False

        self.listeners = []
        self.clients = []

        self.GLOBAL_META_WEIGHTS = None

        for i in range(NUM_ORGS):
            address = ('localhost', (7000 + i*100))  # family is deduced to be 'AF_INET'
            listener = Listener(address, authkey=b'secret password')
            self.listeners.append(listener.accept())

            address = ('localhost', (7000 + i * 100 + 1))
            while True:
                try:
                    client = Client(address, authkey=b'secret password')
                    self.clients.append(client)
                    break
                except:
                    pass



    def run(self):
        batch_number = 0
        while NUM_TIMESTAMPS != (batch_number-1):
        # while True:
            logging.info("___________________________________________________ Batch %s started ______________________________________________________", batch_number)
            weights = []
            num_examples = []
            for i in range(NUM_ORGS):
                msg = self.listeners[i].recv()
                weights.append(msg[0])
                num_examples.append(msg[1])
            # avg_weight = sum(weights) / NUM_ORGS
            for i in range(NUM_ORGS):
                if i == 0:
                    avg_weight = (weights[i] * num_examples[i]) / sum(num_examples)
                else:
                    avg_weight += (weights[i] * num_examples[i]) / sum(num_examples)

            self.GLOBAL_META_WEIGHTS = avg_weight
            batch_number += 1
            logging.info("############### Aggregation finished ###############")
            for i in range(NUM_ORGS):
                self.clients[i].send([self.GLOBAL_META_WEIGHTS, NUM_TIMESTAMPS])
            logging.info("############### SENT BACK TO ORGS ###############")

        accuracy_list = []
        accuracy_std_list = []
        accuracy_99th_list = []
        accuracy_90th_list = []

        recall_list = []
        recall_std_list = []
        recall_99th_list = []
        recall_90th_list = []

        auc_list = []
        auc_std_list = []
        auc_99th_list = []
        auc_90th_list = []

        f1_list = []
        f1_std_list = []
        f1_99th_list = []
        f1_90th_list = []

        precision_list = []
        precision_std_list = []
        precision_99th_list = []
        precision_90th_list = []

        mean_time_list = []
        mean_time_std_list = []
        mean_time_99th_list = []
        mean_time_90th_list = []

        total_time_list = []
        total_time_std_list = []

        for i in range(NUM_ORGS):
            msg = self.listeners[i].recv()
            accuracy_list.append(msg[0][0])
            accuracy_std_list.append(msg[0][1])
            accuracy_99th_list.append(msg[0][2])
            accuracy_90th_list.append(msg[0][3])

            recall_list.append(msg[1][0])
            recall_std_list.append(msg[1][1])
            recall_99th_list.append(msg[1][2])
            recall_90th_list.append(msg[1][3])

            auc_list.append(msg[2][0])
            auc_std_list.append(msg[2][1])
            auc_99th_list.append(msg[2][2])
            auc_90th_list.append(msg[2][3])

            f1_list.append(msg[3][0])
            f1_std_list.append(msg[3][1])
            f1_99th_list.append(msg[3][2])
            f1_90th_list.append(msg[3][3])

            precision_list.append(msg[4][0])
            precision_std_list.append(msg[4][1])
            precision_99th_list.append(msg[4][2])
            precision_90th_list.append(msg[4][3])

            mean_time_list.append(msg[5][0])
            mean_time_std_list.append(msg[5][1])
            mean_time_99th_list.append(msg[5][2])
            mean_time_90th_list.append(msg[5][3])

            total_time_list.append(msg[6][0])
            total_time_std_list.append(msg[6][1])

        logging.info(
            "______________________________________________________________________________________________________ Final Mean Values of all Organizations ______________________________________________________________________________________________________")
        logging.info(
            "##########################################################################################################################################################################################################################")

        logging.info(
            'Result report : Accuracy - %s (%s), Recall - %s (%s), AUC - %s (%s), F1 - %s (%s), Precision - %s (%s), Mean time for a batch - %s (%s) seconds, Total Time - %s (%s)',
            str(round(np.mean(accuracy_list), 4)), str(round(np.mean(accuracy_std_list), 4)),
            str(round(np.mean(recall_list), 4)), str(round(np.mean(recall_std_list), 4)),
            str(round(np.mean(auc_list), 4)), str(round(np.mean(auc_std_list), 4)),
            str(round(np.mean(f1_list), 4)), str(round(np.mean(f1_std_list), 4)),
            str(round(np.mean(precision_list), 4)), str(round(np.mean(precision_std_list), 4)),
            str(round(np.mean(mean_time_list), 4)), str(round(np.mean(mean_time_std_list), 4)),
            str(round(np.mean(total_time_list), 4)), str(round(np.std(total_time_list), 4))
        )
        logging.info(
            'Result report : Accuracy 99th - 90th (%s, %s), Recall 99th - 90th (%s, %s), AUC 99th - 90th (%s, %s), F1 99th - 90th (%s, %s), Precision 99th - 90th (%s, %s), Mean time for a batch 99th - 90th (%s, %s)',
            str(round(np.mean(accuracy_99th_list), 4)), str(round(np.mean(accuracy_90th_list), 4)),
            str(round(np.mean(recall_99th_list), 4)), str(round(np.mean(recall_90th_list), 4)),
            str(round(np.mean(auc_99th_list), 4)), str(round(np.mean(auc_90th_list), 4)),
            str(round(np.mean(f1_99th_list), 4)), str(round(np.mean(f1_90th_list), 4)),
            str(round(np.mean(precision_99th_list), 4)), str(round(np.mean(precision_90th_list), 4)),
            str(round(np.mean(mean_time_99th_list), 4)), str(round(np.mean(mean_time_90th_list), 4))
        )
        logging.info(
            "______________________________________________________________________________________________________ Final Mean Values of all Organizations ______________________________________________________________________________________________________")
        logging.info(
            "##########################################################################################################################################################################################################################")

        logging.info('Total time list: ' + str(total_time_list))
        logging.info('Accuracy list: ' + str(accuracy_list))
        logging.info('Recall list: ' + str(recall_list))
        logging.info('AUC list: ' + str(auc_list))
        logging.info('F1 list: ' + str(f1_list))
        logging.info('Precision list: ' + str(precision_list))

if __name__ == "__main__":

    arg_names = [
        'path_nodes',
        'path_edges',
        'graph_id',
        'partition_id',
        'num_orgs',
        'num_rounds',
        'IP',
        'PORT'
    ]

    args = dict(zip(arg_names, sys.argv[1:]))

    logging.info('####################################### New Training Session #######################################')
    logging.info('aggregator started , graph ID %s, number of organizations %s', GRAPH_ID,NUM_ORGS)



    logging.info('Model initialized')

    aggregator = GlobalAggregator(graph_id=GRAPH_ID, num_orgs=int(NUM_ORGS))

    logging.info('Federated training started between organizations!')

    start = timer()
    aggregator.run()
    end = timer()

    elapsed_time = end - start
    logging.info('Federated training done!')
    logging.info('Training report : Elapsed time %s seconds, number of organizations %s', elapsed_time, NUM_ORGS)
