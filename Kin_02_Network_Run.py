#
import pandas as pd
from kinship_py.utils import print2
from example_CBDB.Kin_00_Config import *
import networkx as nx


# input_nodes = pd.read_csv(os.path.join(data_dir, 'input_node.txt'), sep='\t', keep_default_na=False, dtype={"c_birthyear": object, "c_deathyear": object})
# input_edges = pd.read_csv(os.path.join(data_dir, 'input_edge.txt'), sep='\t', keep_default_na=False)

def network_building(input_nodes, input_edges, output_dir):
    # initialization
    mkdir_chdir(output_dir)
    kinship_network = KinNetwork()

    # adding nodes to kinship network
    kinship_network.add_nodes_from(input_nodes['c_personid'])
    # adding node's attributes
    for col in input_nodes.columns[1:]:
        for PID, value in zip(tqdm(input_nodes['c_personid'], desc=col), input_nodes[col]):
            kinship_network.nodes[PID][col] = value
            kinship_network.nodes[PID]['status'] = 1

    # adding edge and its attributes
    # set(input_edges['c_kin_code'])
    vague_kin_code = {"G", "n", "K", "~", "P", "A", "°", "½", "©", "#", "O", "L", "%", "U", "y", "*", "^", "!"}
    for PID1, PID2, KinCode, KinName, generation_diff in zip(tqdm(input_edges['PID1']), input_edges['PID2'],
                                                             input_edges['c_kin_code'], input_edges['c_kin_name'],
                                                             input_edges['generation_diff']):
        code: ComplexKinshipCode = ComplexKinshipCode(KinCode)
        status: int = 1
        error_type = []
        for char in vague_kin_code:
            if char in code.to_str():
                status: int = 0
        kinship_network.add_edge(PID1, PID2, kinship_code=code, kin_name=KinName, status=status, round=0,
                                 error_type=error_type, gen_diff=generation_diff)

    # hyper-parameter setting
    kinship_network.characteristic_kin_relation = ['F', 'FF', 'FFF']
    kinship_network.kin_neighbor_distance_threshold = 4
    kinship_network.score_weight = (3, 2, 2, 1)
    kinship_network.score_threshold = 1

    # Data process 1a: error-in-gender,
    # Data process 1b: error-in-one-way-relation,
    # Data process 1c: error-in-inverse-kin-code
    # Data process 1d: remove the edges with (1) only one-way relationship; (2) code conflicts with its inverse form
    kinship_network.process_step1_gender_bidirection_inverse()

    # Data process 2:  multi-fathers_processing
    PID_list = set(input_nodes['c_personid'])
    kinship_network.process_step2_multiple_fathers(PID_list)

    # Data process 3:  multi-relations_processing
    kinship_network.process_step3_multiple_relations()

    # Data process 4:  multi-seniority_processing
    PIDs2seniority_dict = kinship_network.process_step4_multiple_seniority()

    # stop criteria (node_num_diff < 3)
    iteration = 0
    node_num_diff = 1000

    # summary log
    kinship_network.get_summary_log(iteration=0, node_adding_num=0, node_removing_num=0,
                                    edge_adding_num=0, node_close_num=0,
                                    edge_o_active_num=0, edge_i_active_num=0)

    file_name_output_node = os.path.join(data_dir, f'output_node_Istep_round{iteration}.txt')
    kinship_network.output_node(file_name=file_name_output_node)
    file_name_output_edge = os.path.join(data_dir, f'output_edge_Istep_round{iteration}.txt')
    kinship_network.output_edge(file_name=file_name_output_edge)

    # Optimization-Integration algorithm
    while node_num_diff > 0:
        iteration += 1
        node_total_num = kinship_network.number_of_nodes()

        # O-step
        edge_adding_num, node_close_num = kinship_network.o_step(iteration)
        edge_o_active_num = kinship_network.get_num_edge_active()

        # file_name_output_node = os.path.join(data_dir, f'output_node_Ostep_round{iteration}.txt')
        # kinship_network.output_node(file_name=file_name_output_node)
        file_name_output_edge = os.path.join(data_dir, f'output_edge_Ostep_round{iteration}.txt')
        kinship_network.output_edge(file_name=file_name_output_edge)

        # I-step
        node_adding_num, node_removing_num = kinship_network.i_step(iteration)
        edge_i_active_num = kinship_network.get_num_edge_active()

        # stop criteria
        node_num_diff = node_total_num - kinship_network.number_of_nodes()

        # summary log
        kinship_network.get_summary_log(iteration, node_adding_num, node_removing_num,
                                        edge_adding_num, node_close_num,
                                        edge_o_active_num, edge_i_active_num)

        file_name_output_node = os.path.join(data_dir, f'output_node_Istep_round{iteration}.txt')
        kinship_network.output_node(file_name=file_name_output_node)
        file_name_output_edge = os.path.join(data_dir, f'output_edge_Istep_round{iteration}.txt')
        kinship_network.output_edge(file_name=file_name_output_edge)

# save optimized network
# nx.write_gpickle(kinship_network, "opt_kinship_network.gpickle")
# with open("opt_kinship_network.pkl", 'wb') as out_put:
#     pickle.dump(kinship_network, out_put)


# load optimized network
# opt_kinship_network = nx.read_gpickle("opt_kinship_network.gpickle")
# with open("opt_kinship_network.pkl", "rb") as get_opt_network:
#     opt_kinship_network = pickle.load(get_opt_network)
