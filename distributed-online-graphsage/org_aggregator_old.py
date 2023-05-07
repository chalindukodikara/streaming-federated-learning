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
parser.add_argument('--port', type=int, default=5150, help='PORT')
parser.add_argument('--partition_algorithm', type=str, default='hash', help='Partition algorithm')

######## Frequently configured #######
parser.add_argument('--dataset_name', type=str, default='wikipedia', help='Dataset name')
parser.add_argument('--partition_size', type=int, default=2, help='Partition size')
parser.add_argument('--num_orgs', type=int, default=1, help='Number of organizations')

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

######## Setup logger ################
# create data folder with the dataset name
folder_path_logs = "logs"
if os.path.exists(folder_path_logs):
    pass
else:
    os.makedirs(folder_path_logs)

folder_path_process = folder_path_logs + "/aggregator"
if os.path.exists(folder_path_process):
    pass
else:
    os.makedirs(folder_path_process)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s : [%(levelname)s]  %(message)s',
    handlers=[
        logging.FileHandler('logs/client/{}_{}_{}_partition_{}_aggregator.log'.format(str(time.strftime('%m %d %H:%M:%S # %l:%M%p on %b %d, %Y')), DATASET_NAME, PARTITION_ALGORITHM, PARTITION_SIZE)),
        logging.StreamHandler(sys.stdout)
    ]
)
############################################
# Find the minimum batch number in partitions
timestamps = []
for partition in range(PARTITION_SIZE):
    path = 'data/' + DATASET_NAME + '_' + str(PARTITION_SIZE) + '_' + str(partition)
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    timestamps.append(int(max(paths, key=os.path.getctime).split('/')[-1].split('_')[0]))
NUM_TIMESTAMPS = min(timestamps)

# for i in range(NUM_ORGS):
#     with open('aggregator_weights/' + str(i) + '_flag.txt', 'w') as f:
#         f.write('0')
#     with open('aggregator_weights/' + str(i) + '_num_examples.txt', 'w') as f:
#         f.write('0')

class GlobalAggregator:

    def __init__(self, graph_id, num_orgs):

        # Parameters
        self.NUM_CLIENTS = num_orgs

        # Global model
        self.model_weights = None

        self.weights = []
        self.partition_sizes = []

        self.stop_flag = False

    def update_model(self, new_weights, num_examples):
        self.partition_sizes.append(num_examples)
        self.weights.append(num_examples * new_weights)

        logging.info(str(self.partition_sizes))
        logging.info(str(self.weights))

        if (len(self.weights) == self.NUM_CLIENTS) and (self.partition_sizes > 0):

            avg_weight = sum(self.weights) / sum(self.partition_sizes)

            self.weights = []
            self.partition_sizes = []

            self.model_weights = avg_weight

            self.training_cycles += 1

            for soc in self.sockets_list[1:]:
                self.send_model(soc)

            logging.info(
                "___________________________________________________ Training round %s done ______________________________________________________",
                self.training_cycles)

        else:

            logging.error("Invalid patition size")

    def send_model(self, client_socket):

        if self.ROUNDS == self.training_cycles:
            self.stop_flag = True

        weights = np.array(self.model_weights)
        weights_path = WEIGHT_FILE_PATH + "weights_round_" + str(self.training_cycles) + ".npy"
        np.save(weights_path, weights)

        data = {"STOP_FLAG": self.stop_flag, "WEIGHTS": weights}

        data = pickle.dumps(data)
        data = bytes(f"{len(data):<{HEADER_LENGTH}}", 'utf-8') + data

        client_socket.sendall(data)

        logging.info('Sent global model to client-%s at %s:%s', self.client_ids[client_socket],
                     *self.clients[client_socket])

    def receive(self, client_socket):
        try:

            message_header = client_socket.recv(HEADER_LENGTH)

            if not len(message_header):
                logging.error('Client-%s closed connection at %s:%s', self.client_ids[client_socket],
                              *self.clients[client_socket])
                return False

            message_length = int(message_header.decode('utf-8').strip())

            full_msg = b''
            while True:
                msg = client_socket.recv(message_length)

                full_msg += msg

                if len(full_msg) == message_length:
                    break

            return pickle.loads(full_msg)

        except Exception as e:
            logging.error('Client-%s closed connection at %s:%s', self.client_ids[client_socket],
                          *self.clients[client_socket])
            return False

    def run(self):
        batch_number = 0
        while NUM_TIMESTAMPS != batch_number:
        # while True:
            flags = []
            weights = []
            num_examples = []
            while len(flags) != NUM_ORGS:
                for i in range(NUM_ORGS):
                    while True:
                        try:
                            with open('aggregator_weights/' + str(i) + '_flag.txt', 'r') as f:
                                x = f.read().rstrip()
                                if int(x) == 1 and (i not in flags):
                                    weights.append(np.load('aggregator_weights/' + str(i) + ".npy",  allow_pickle=True))
                                    with open('aggregator_weights/' + str(i) + '_num_examples.txt', 'r') as f:
                                        y = f.read().rstrip()
                                        num_examples.append(int(y))
                                    flags.append(i)
                                    break
                        except:
                            print('Error occurred while reading')

            avg_weight = sum(weights) / sum(num_examples)
            np.save('aggregator_weights/' + str(i) + ".npy", avg_weight)
            batch_number += 1
            for i in range(NUM_ORGS):
                with open('aggregator_weights/' + str(i) + '_flag.txt', 'w') as f:
                    f.write('0')


if __name__ == "__main__":

    # from models.supervised import Model

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

    aggregator = GlobalAggregator(graph_id=GRAPH_ID,num_orgs=int(NUM_ORGS))

    logging.info('Federated training started!')

    start = timer()
    aggregator.run()
    end = timer()

    elapsed_time = end - start
    # logging.info('Federated training done!')
    # logging.info('Training report : Elapsed time %s seconds, graph ID %s, number of clients %s, number of rounds %s',
    #              elapsed_time, args['graph_id'], args['num_orgs'], args['num_rounds'])
