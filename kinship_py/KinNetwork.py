#
from kinship_py.KinshipCode import ComplexKinshipCode, is_compatible
from tqdm import tqdm
from typing import List, Set, Dict, Tuple
import networkx as nx
# from networkx import all_simple_edge_paths
# from networkx import MultiDiGraph, weakly_connected_components, get_node_attributes, get_edge_attributes
from kinship_py.utils import print2
from collections import defaultdict
from itertools import combinations
import pandas as pd
import numpy as np
import re

"""
========================
Network analysis functions
========================

    ==================== ===============================================================================
    Utility functions
    ====================================================================================================
    [F1] get_num_node_active                get total number of node (with active status).
    [F2] get_num_edge_active                get total number of edge (with active status).
    [F3] get_total_kin_distance             get total kin distance of network (with active edge).
    [F4] get_num_community                  get total number of community and maximum size of community.
    ==================== ===============================================================================

    ==================== ===============================================================================
    Edge-related functions
    ====================================================================================================
    [F5] get_edge_feature(PID1, PID2, key=None, feature='status')
    [F6] set_edge_status(PID1, PID2, from_status=0, to_status=0)                       
    ==================== ===============================================================================
    
========================
O-step functions
========================

    ==================== ===============================================================================
    Utility functions
    ====================================================================================================
    [F1] local_network                      Ei→· ≜ {ei,j ∈ Ea: vj ∈ Va}.
    [F2] o_step_scanning                    scan all possible patterns in a given local network.
    [F3] o_step_reasoning_operation         discovery kinship and plug predicted info. into network.
    [F4] o_step_pruning                     close redundant edges.
    ==================== ===============================================================================
    
    ==================== ===============================================================================
    Compatibility functions
    ====================================================================================================
    [F5] o_step_is_edge_redundant(PID1, PID2)                       
    ==================== ===============================================================================

========================
I-step functions
========================

    ==================== ===============================================================================
    Utility functions
    ====================================================================================================
    [F1] name2PIDs                      a dict, each element relates to no less than two PIDs.
    [F2] kinship_neighbourhood          Ni,d ≜ {ei,j ∈ E: vj ∈ Va, DK(ei,j) ≤ d}.
    [F3] score_function                 quantify the matching strength between Ni,d and Nj,d.
    [F4] i_step_matching_generating     generate merging rules.
    [F5] i_step_merging                 merge two nodes proposed in merging rules list.
    [F6] i_step_freeze                  undo all done operations (remove new node).
    ==================== ===============================================================================
    
    ==================== ===============================================================================
    Compatibility functions
    ====================================================================================================
    [F9] is_node_feature_consistent(PID1, PID2, attr= 'c_birthyear')                 
    [F10] is_edge_consistent_in_local_network(PID1)       
    ==================== ===============================================================================

"""


class KinNetwork(nx.MultiDiGraph):
    node_attribute_list = ['c_personid', 'c_name_chn', 'c_female', 'c_birthyear', 'c_deathyear',
       'c_surname_chn', 'c_mingzi_chn']
    edge_attribute_list = ['kinship_code', 'status', 'error_type']

    characteristic_relatives_inverse = {'S', 'D',
                                        'DD', 'SS', 'SD', 'DS',
                                        'SSS', 'SSD', 'SDS', 'DSS', 'SDD', 'DSD', 'DDS', 'DDD'}

    @staticmethod
    def init_print():
        print2('Welcome to use KinshipNetwork developed by Francis; Email:zhoumq19@mails.tsinghua.edu.cn',
               mode='w', add_time=False)
        # ==================== ==================== ====================
        #                       'summary log.txt'
        # ==================== ==================== ====================
        print2('date', 'time', 'iteration',
               'node_total_num', 'node_adding_num', 'node_removing_num',
               'edge_total_num', 'edge_o_active_num', 'edge_i_active_num', 'edge_adding_num', 'node_close_num',
               'complexity_total_kin_distance', 'complexity_ave_kin_distance',
               'connect_area_total_num', 'connect_area_ave_size', 'connect_area_max_size',
               file='summary log.txt', add_time=False, mode='w', print2console=False)
        # ==================== ==================== ====================
        #                       'edge operation.txt'
        # Note: operation takes values in {add, close};
        # if operation = 'add', then (PID2,PID3,k23,PID1,k12,k13)
        # if operation = 'close', then (PID1,PID2,redundant_edge,infer_node_in_path,infer_kinship_code_in_path,stage)
        # 'stage' takes values in {reasoning, pruning}
        # ==================== ==================== ====================
        print2('date', 'time', 'iteration',
               'operation', 'from_node', 'to_node', 'kin_prediction', 'PID1', 'k12', 'k13',
               file='edge operation.txt', add_time=False, mode='w', print2console=False)
        # ==================== ==================== ====================
        #                       'node operation.txt'
        # Note: 'operation' takes value in {'merging', 'recommendation'}
        # if operation = 'recommendation', then set 'new_PID' = -99999,
        # 'error_type' takes value in {consistent, 'inconsistent-node', 'inconsistent-edge'}
        # ==================== ==================== ====================
        print2('date', 'time', 'iteration',
               'operation', 'error_type', 'name', 'new_PID', 'from_PID1', 'from_PID2', 'score',
               file='node operation.txt', add_time=False, mode='w', print2console=False)
        # ==================== ==================== ====================
        #                       'contradiction.txt'
        # ==================== ==================== ====================
        print2('date', 'time', 'round',
               'error_type', 'record_ID', 'PID1', 'PID2', 'kinship_code', 'details',
               file='contradiction.txt', add_time=False,
               mode='w', print2console=False)

    def __init__(self, incoming_graph_data=None, **attr):
        super(KinNetwork, self).__init__(incoming_graph_data, **attr)
        self.characteristic_kin_relation = attr.get('characteristic_kin_relation', ['F', 'FF', 'FFF'])
        self.kin_neighbor_distance_threshold = attr.get('kin_neighbor_distance_threshold', 4)
        self.score_weight = attr.get('score_weight', (3, 2, 2, 1))
        self.score_threshold = attr.get('score_threshold', 1)
        self.min_PID = 0
        self.name2PIDs_no_merging = []
        self.init_print()

    def get_summary_log(self, iteration=0, node_adding_num=0, node_removing_num=0,
                        edge_adding_num=0, node_close_num=0,
                        edge_o_active_num=0, edge_i_active_num=0):
        node_total_num = self.number_of_nodes()
        edge_total_num = self.number_of_edges()
        if edge_o_active_num==0 and edge_i_active_num==0:
            edge_o_active_num = edge_i_active_num = self.get_num_edge_active()

        complexity_total_kin_distance = self.get_total_kin_distance()
        complexity_ave_kin_distance = complexity_total_kin_distance / self.get_num_edge_active()
        complexity_ave_kin_distance = round(complexity_ave_kin_distance, 5)

        connect_area_total_num, connect_area_max_size = self.get_num_community()
        connect_area_ave_size = connect_area_total_num / node_total_num
        connect_area_ave_size = round(connect_area_ave_size, 5)

        # print summary log to 'summary log.txt'
        print2(iteration, node_total_num, node_adding_num, node_removing_num,
               edge_total_num, edge_o_active_num, edge_i_active_num, edge_adding_num, node_close_num,
               complexity_total_kin_distance, complexity_ave_kin_distance,
               connect_area_total_num, connect_area_ave_size, connect_area_max_size,
               file='summary log.txt', mode='a', print2console=False)

    def get_num_node_active(self):
        node_list = nx.get_node_attributes(self, 'status')
        total = sum(node_list.values())
        return total

    def get_num_edge_active(self):
        edge_list = nx.get_edge_attributes(self, 'status')
        total = sum(i for i in edge_list.values())
        return total

    def get_total_kin_distance(self):
        total = 0
        edge_list = nx.get_edge_attributes(self, 'kinship_code')
        for edges, codes in edge_list.items():
            if self.edges[edges]['status'] == 1:
                total += codes.get_kin_distance()
        return total

    def get_num_community(self):
        connect_num = nx.weakly_connected_components(self)
        c_size = []
        total_num = 0
        for community in connect_num:
            total_num += 1
            c_size.append(len(community))
        max_size = max(c_size)
        return total_num, max_size

    def get_edge_feature(self, PID1, PID2, key=None, feature='status'):
        """

        :param PID1: node index
        :param PID2: node index
        :param key: if None, iterative assign @value of @feature to all parallel edges; else focus on specific edge
        :param feature: {'kinship_code', 'status'}
        :return:
        """
        assert self.get_edge_data(PID1, PID2) is not None, 'No edges between node' + str(PID1) + ' and node ' + str(
            PID2)
        out = []
        if key:
            edge_ix_feature = (key, self.get_edge_data(PID1, PID2, key)[feature])
            out.append(edge_ix_feature)
        else:
            for key, value in self.get_edge_data(PID1, PID2).items():
                edge_ix_feature = (key, value[feature])
                out.append(edge_ix_feature)
        # print(str(len(edge_ix_feature)) + 'edge(s) between' + str(PID1) + ',' + str(PID2))  # for debug test
        return out

    def set_edge_status(self, PID1, PID2, from_status=0, to_status=0):
        assert self.get_edge_data(PID1, PID2) is not None, 'No edges between node' + str(PID1) + ' and node ' + str(
            PID2)
        num_of_operation = 0
        for key, value in self.get_edge_data(PID1, PID2).items():
            if value['status'] == from_status:
                value['status'] = to_status
                num_of_operation += 1
        return num_of_operation

    def o_step(self, iteration=1):
        print2(f'----------------------------------------------------', mode='a', add_time=False)
        print2(f' ', mode='a', add_time=True)
        print2(f'Start O-step, iteration {iteration}', mode='a', add_time=False)
        # for each local network E_i
        # (1) scanning
        # (2) reasoning
        # (3) security check
        # (4) operating
        # (5) pruning
        edge_adding_num = 0
        node_close_num = 0
        edge_adding_list = []
        PID_list = set(self.nodes)
        for PID1 in PID_list:
            # print(PID1)
            adding_list = self.o_step_reasoning_operation(PID1=PID1, iteration=iteration)
            edge_adding_list.extend(adding_list)
            if adding_list:
                # print(f'Adding edges in node {PID1}: {edge_adding_list}')
                edge_adding_num += len(adding_list)
                # considering the bi-direction fashion, only check one direction
                pruning_list = self.o_step_pruning(adding_list[::2], iteration=iteration)
                if pruning_list:
                    # print(f'Closing edges in node {PID1}: {pruning_list}')
                    node_close_num += len(pruning_list)
            # print('Adding %d edges in node %d' % (len(edge_adding_list), PID))
            # print('Closing %d edges in round %d' % (len(pruning_list), iteration))
        print2(f'Add {edge_adding_num} edges, close {node_close_num} edges', mode='a', add_time=False)
        print2(f' ', mode='a', add_time=True)
        print2(f'End O-step, iteration {iteration}', mode='a', add_time=False)
        print2(f'----------------------------------------------------\n', mode='a', add_time=False)
        return edge_adding_num, node_close_num

    def i_step(self, iteration=1):
        # for each kinship neighbourhood N_i
        # (1) matching
        # (2) generating
        # (3) merging
        # (4) security check
        # (5) freezing
        print2(f'----------------------------------------------------', mode='a', add_time=False)
        print2(f' ', mode='a', add_time=True)
        print2(f'Start I-step, iteration {iteration}', mode='a', add_time=False)
        node_adding_num = 0
        name2PIDs = self.name2PIDs
        for name in self.name2PIDs_no_merging:
            name2PIDs.pop(name)
        for name, PID_list in name2PIDs.items():
            # return two lists --> (PID1 ,PID2, score), (PID1 ,PID2, score)
            merging_rule, recommendation_list = self.i_step_matching_generating(name, PID_list)
            if merging_rule:
                successful_merging, fail_merging = self.i_step_merging(merging_rule, iteration=iteration)
                if not successful_merging:
                    self.name2PIDs_no_merging.append(name)
                else:
                    node_adding_num += len(successful_merging)
            if recommendation_list:
                operation = 'recommendation'
                r_type = 'consistent'
                new_PID = -99999
                for _, PID1, PID2, score, _ in merging_rule:
                    print2(iteration, operation, r_type, name, new_PID, PID1, PID2, score,
                           file='node operation.txt', mode='a')
        node_removing_num = node_adding_num * 2
        print2(f'Add {node_adding_num} nodes, remove {node_removing_num} edges', mode='a', add_time=False)
        print2(f' ', mode='a', add_time=True)
        print2(f'End I-step, iteration {iteration}', mode='a', add_time=False)
        print2(f'----------------------------------------------------\n', mode='a', add_time=False)
        return node_adding_num, node_removing_num

    # O-step: Utility functions

    def local_network(self, PID1, active_edge=True):
        # {e_{i,j} \in E_a: v_j \in V_a}
        # for PID2, attr_dict in self[PID].items():
        #     kin_status = attr_dict['status']
        #     if kin_status == 1:  # judge if kin_status is active
        #         kinship_code = attr_dict['kinship_code']
        #         yield PID2, kinship_code
        out = []
        if self.nodes[PID1]['status'] == 0:
            print('Node %d is redundant' % PID1)
            return out
        if active_edge:
            for PID2, edge_key2attr_dict in self[PID1].items():
                for edge_key, attr_dict in edge_key2attr_dict.items():  # 如果两个节点存在多重关系呢？？？？？？
                    kin_status = attr_dict['status']
                    if kin_status == 1:  # judge if kin_status is active
                        k12 = attr_dict['kinship_code']
                        neighbor = (PID2, k12.to_str())  # type --> (int, str)
                        out.append(neighbor)
        else:
            for PID2, edge_key2attr_dict in self[PID1].items():
                for edge_key, attr_dict in edge_key2attr_dict.items():  # 如果两个节点存在多重关系呢？？？？？？
                    k12 = attr_dict['kinship_code']
                    neighbor = (PID2, k12.to_str())  # type --> (int, str)
                    out.append(neighbor)
        return out

    @staticmethod
    def o_step_pattern(kin1: str, kin2: str):
        # take values in ['F', 'FF', 'FFF']
        if set(kin1) != {'F'}:
            return None
        if len(kin1) > 3:
            return None
        if len(kin1) > len(kin2):
            return None
        if 'B' in kin2:
            kin2 = kin2.replace('B+', 'FS').replace('B-', 'FS').replace('B', 'FS')
        if 'Z' in kin2:
            kin2 = kin2.replace('Z+', 'FD').replace('Z-', 'FD').replace('Z', 'FD')
        if 'M' in kin2:
            kin2 = kin2.replace('M', 'FW')
        if 'S+' in kin2 or 'S-' in kin2:
            kin2 = kin2.replace('S+', 'S').replace('S-', 'S')
        if 'D+' in kin2 or 'D-' in kin2:
            kin2 = kin2.replace('D+', 'D').replace('D-', 'D')
        if kin2.startswith(kin1):
            return kin2[len(kin1):]

    def o_step_scanning(self, PID1) -> List[Tuple[int, int, str]]:
        out = []  # (v_{i2}, v_{i3}, e_{i2,i3}), e.g., (PID2,PID3, K23)
        local_network = self.local_network(PID1)
        for PID2, k12 in local_network:
            for PID3, k13 in local_network:
                k23 = self.o_step_pattern(k12, k13)
                if k23:
                    if 'FS' in k23:
                        k23 = k23.replace('FS', 'B')
                    if 'FD' in k23:
                        k23 = k23.replace('FD', 'Z')
                    if 'FW' in k23:
                        k23 = k23.replace('FW', 'M')
                    prediction = (PID2, PID3, k23, k12, k13)  # type --> (int, int, str)
                    out.append(prediction)
        self.optimize_prediction_list(out)
        return out

    def o_step_reasoning_operation(self, PID1, iteration=1):
        """
        simultaneously perform reasoning, security check, and adding operation
        :param PID1:
        :param iteration:
        :return: new KinNetwork, operation list
        """
        add_list = []
        prediction = self.o_step_scanning(PID1)
        for PID2, PID3, k23, k12, k13 in prediction:
            # judge if e_{2,3) exists already
            if self.has_edge(PID2, PID3):
                # extract all edges between PID2, PID3, w.r.t 'kinship_code'
                k23s = self.get_edge_feature(PID2, PID3, feature='kinship_code')
                for _, code in k23s:
                    # judge if k.hat(e_{2,3}) = k (e_{2,3})
                    if not code.kinship_code == k23:
                        # if false, switch status to -1, i.e. error state
                        self.set_edge_status(PID2, PID3, from_status=1, to_status=-1)
                        self.set_edge_status(PID1, PID2, from_status=1, to_status=-1)
                        self.set_edge_status(PID1, PID3, from_status=1, to_status=-1)
                        # report to 'contradiction.txt'
                        details = 'reasoning:' + '(k12,k13,k23)=' + str((k12, k13, k23))
                        self.contradiction_report_3(PID1, PID2, PID3, error_type='error-in-o-step', details=details)
                    else:
                        pass
                        # if true, switch status to 0, i.e. redundant state
                        # print(f'reasoning stage: close edge ({PID1},{PID2})')
                        # print(f'reasoning stage: close edge ({PID1},{PID3})')
                        # self.set_edge_status(PID1, PID2, from_status=1, to_status=0)
                        # self.set_edge_status(PID1, PID3, from_status=1, to_status=0)
                        # operation_type = 'close'
                        # stage = 'reasoning'
                        # infer_node_in_path = str(PID1) + ',' + str(PID3) + ',' + str(PID2)
                        # print2(iteration, operation_type, PID1, PID2, k12, infer_node_in_path, k12, stage,
                        #        file='edge operation.txt', mode='a')
                        # infer_node_in_path = str(PID1) + ',' + str(PID2) + ',' + str(PID3)
                        # print2(iteration, operation_type, PID1, PID3, k13, infer_node_in_path, k13, stage,
                        #        file='edge operation.txt', mode='a')

            # if no edges between PID2 and PID3 in database, then add the predicted k23 to the network
            else:
                code: ComplexKinshipCode = ComplexKinshipCode(k23)
                self.add_edge(PID2, PID3, kinship_code=code, status=1, round=iteration,
                              error_type=[], gen_diff=code.generation)
                add_list.append((PID2, PID3, k23))
                # report to 'edge operation.txt'
                operation_type = 'add'
                print2(iteration, operation_type, PID2, PID3, k23, PID1, k12, k13, file='edge operation.txt', mode='a')

            gender2 = self.nodes[PID2]['c_female']
            k23_inverse_code_list = ComplexKinshipCode(k23).get_inverse_kinship(gender2)
            k32 = ''.join(code.to_str() for code in k23_inverse_code_list)
            # judge if e_{3,2) exists already
            if self.has_edge(PID3, PID2):
                # extract all edges between PID3, PID2, w.r.t 'kinship_code'
                k32s = self.get_edge_feature(PID3, PID2, feature='kinship_code')
                for _, code in k32s:
                    # judge if k.hat(e_{2,3}) = k (e_{2,3})
                    if not code.kinship_code == k32:
                        # if false, switch status to -1, i.e. error state
                        self.set_edge_status(PID3, PID2, from_status=1, to_status=-1)
                        self.set_edge_status(PID2, PID1, from_status=1, to_status=-1)
                        self.set_edge_status(PID3, PID1, from_status=1, to_status=-1)
                        # report to 'contradiction.txt'
                        details = 'reasoning:' + '(k12,k13,k32)=' + str((k12, k13, k32))
                        self.contradiction_report_3(PID3, PID2, PID1, error_type='error-in-o-step', details=details)
                    else:
                        pass
                        # if true, switch status to 0, i.e. redundant state
                        # print(f'reasoning stage: close edge ({PID2},{PID1})')
                        # print(f'reasoning stage: close edge ({PID3},{PID1})')
                        # self.set_edge_status(PID2, PID1, from_status=1, to_status=0)
                        # self.set_edge_status(PID3, PID1, from_status=1, to_status=0)
            else:
                k32_ComplexKinshipCode = ComplexKinshipCode(primary_code_list=k23_inverse_code_list)
                self.add_edge(PID3, PID2, kinship_code=k32_ComplexKinshipCode,
                              status=1, round=iteration, error_type=[], gen_diff=k32_ComplexKinshipCode.generation)
                k32 = k32_ComplexKinshipCode.to_str()
                add_list.append((PID3, PID2, k32))
                # report to 'edge operation.txt'
                operation_type = 'add'
                print2(iteration, operation_type, PID3, PID2, k32, PID1, k12, k13,
                       file='edge operation.txt', mode='a')
        return add_list

    def o_step_is_edge_redundant(self, PID1, PID2, k12):
        if self.has_edge(PID1, PID2):
            edge_redundant = ComplexKinshipCode(k12)
            edge_redundant_kinship_code = k12
            edge_redundant_kinship_distance = edge_redundant.kin_distance
            if edge_redundant_kinship_distance == 1:
                return 0, 'Not redundant', 'kin distance: 1'
            else:
                visited_nodes = [PID1]
                edge_in_path_kinship_code = ''
                edge_in_path_kinship_distance = 0
                local_network = self.local_network(PID1)
                if (PID2, k12) in local_network:
                    local_network.remove((PID2, k12))
                local_network = [(PID1, PID2, k12) for PID2, k12 in local_network]
                stack = [iter(local_network)]
                while stack:
                    children = stack[-1]
                    child = next(children, None)  # neighbours
                    # print(child)
                    if child is None:
                        stack.pop()  # 移除列表中的最后一个元素
                        visited_nodes.pop()
                        edge_in_path_kinship_code = ''
                        edge_in_path_kinship_distance = 0
                    else:
                        child_kin_code = child[2].replace('B', 'FS').replace('Z', 'FD')
                        child_kin_distance = ComplexKinshipCode(child[2]).get_kin_distance()

                        if child_kin_distance is None:
                            continue
                        if child[1] == PID2:
                            if edge_in_path_kinship_distance + child_kin_distance == edge_redundant_kinship_distance:
                                edge_in_path_final_code = edge_in_path_kinship_code + child_kin_code
                                visited_nodes.append(PID2)
                                if is_compatible(edge_redundant_kinship_code, edge_in_path_final_code):
                                    return 1, visited_nodes, edge_in_path_final_code  # compatible
                                else:
                                    # print(f'Edge of {PID1} & {PID2} is error w.r.t paths: {visited[1:] + [child]}')
                                    return -1, visited_nodes, edge_in_path_final_code  # error
                            else:
                                pass
                        elif child[1] not in visited_nodes:
                            if edge_redundant_kinship_code.replace('B', 'FS').replace('Z', 'FD').startswith(
                                    child_kin_code) and edge_in_path_kinship_distance + child_kin_distance < edge_redundant_kinship_distance:
                                visited_nodes.append(child[1])
                                search_path = self.local_network(child[1])
                                search_path = [(child[1], PID2, k12) for PID2, k12 in search_path]
                                stack.append(iter(search_path))
                                edge_in_path_kinship_code += child_kin_code
                                edge_in_path_kinship_distance += child_kin_distance
                return 0, 'Not redundant!', 'No match path'
        else:
            return -1, 'No edge!', 'error!'

    def o_step_pruning(self, edge_adding_list=None, iteration=1):
        pruning_list = []
        pruning_dict = {}
        for id1, id2, _ in edge_adding_list:
            pruning_dict[id1] = 1
            pruning_dict[id2] = 1
        node_list = list(pruning_dict.keys())
        for PID1 in node_list:
            local_network = self.local_network(PID1)
            for PID2, k12 in local_network:
                # print(f'pruning step to check {PID1} and {PID2}')
                k21_code_list = ComplexKinshipCode(k12).get_inverse_kinship(gender_0=self.nodes[PID1]['c_female'])
                k21 = ''.join(code.to_str() for code in k21_code_list)
                is_redundant_k12, node_in_path_k12, edge_in_path_k12 = self.o_step_is_edge_redundant(PID1, PID2, k12)
                is_redundant_k21, node_in_path_k21, edge_in_path_k21 = self.o_step_is_edge_redundant(PID2, PID1, k21)
                # print(f'Is ({PID1}, {PID2}) redundant: {is_redundant}')
                if is_redundant_k12 == 1 and is_redundant_k21 == 1:
                    # print(f'redundant stage: close edge ({PID1},{PID2})')
                    # print(f'redundant stage: close edge ({PID2},{PID1})')
                    self.set_edge_status(PID1, PID2, from_status=1, to_status=0)
                    self.set_edge_status(PID2, PID1, from_status=1, to_status=0)
                    pruning_list.append((PID1, PID2, k12))
                    pruning_list.append((PID2, PID1, k21))
                    # report to 'edge operation.txt'
                    operation_type = 'close'
                    node_in_path = ','.join(str(x) for x in node_in_path_k12)
                    inverse_in_path = ''.join(reversed(node_in_path))
                    print2(iteration, operation_type, PID1, PID2, k12, node_in_path, edge_in_path_k12, 'redundant',
                           file='edge operation.txt', mode='a')
                    print2(iteration, operation_type, PID2, PID1, k21, inverse_in_path, edge_in_path_k21, 'redundant',
                           file='edge operation.txt', mode='a')
                if is_redundant_k12 == -1 or is_redundant_k21 == -1:
                    # print(f'redundant edge: {k12} conflicts with edge {edge_in_path_k12}')
                    error_type = 'kin-code-incompatible'
                    record_id = f'{PID1, PID2}'
                    details = 'pruning:' + str(node_in_path_k12) + ' ' + str(edge_in_path_k12)
                    print2(iteration, error_type, record_id, PID1, PID2, k12, details,
                           file='contradiction.txt', mode='a')
                    details = 'pruning:' + str(node_in_path_k21) + ' ' + str(edge_in_path_k21)
                    print2(iteration, error_type, record_id, PID2, PID1, k21, details,
                           file='contradiction.txt', mode='a')
        return pruning_list

    def optimize_prediction_list(self, prediction_list):
        pass

    def optimize_operation_list(self):
        pass

    def contradiction_report_2(self, record, PID1, PID2, error_type: str, details: str = ''):
        """
        :description: function reports contradiction to "contradiction.txt"
        :param details:
        :param record: f'{PID1}-{PID2}1'
        :param error_type: 'multiple-father', 'multiple-relation',
        :param error_type: 'error-in-seniority', 'error-in-gender', 'error-in-kin-code',
        :param error_type: 'error-in-o-step'
        :return:
        """
        assert error_type in ['multiple-father', 'multiple-relation',
                              'error-in-seniority', 'error-in-gender', 'error-in-kin-code',
                              'error-in-o-step'], 'Wrong input error type!'
        if record is None:
            record = f'{PID1},{PID2}'
        if self.has_edge(PID1, PID2):
            for key, value in self.get_edge_data(PID1, PID2).items():
                out = [value['round'], error_type, record, PID1, PID2, value['kinship_code'].kinship_code, details]
                print2(*out, sep='\t', file='contradiction.txt', mode='a', print2console=False)

    def contradiction_report_3(self, PID1, PID2, PID3, error_type, details):
        record = f'{PID1},{PID2},{PID3}'
        self.contradiction_report_2(record, PID1, PID2, error_type, details)
        self.contradiction_report_2(record, PID1, PID3, error_type, details)
        self.contradiction_report_2(record, PID2, PID3, error_type, details)

    def is_multiple_father(self, PID1):
        out = []
        multiple_F = []
        multiple_FF = []
        multiple_FFF = []
        local_network = self.local_network(PID1, active_edge=False)
        for kin_code in local_network:
            if 'F' == kin_code[1]:
                code = (PID1, kin_code[0], kin_code[1])
                multiple_F.append(code)
            if 'FF' == kin_code[1]:
                code = (PID1, kin_code[0], kin_code[1])
                multiple_FF.append(code)
            if 'FFF' == kin_code[1]:
                code = (PID1, kin_code[0], kin_code[1])
                multiple_FFF.append(code)
        out_dict = {"multiple_father": multiple_F,
                    "multiple_grand_father": multiple_FF,
                    "multiple_great_grand_father": multiple_FFF}
        if len(multiple_F) > 1:
            out.append('multiple_father')
        if len(multiple_FF) > 1:
            out.append('multiple_grand_father')
        if len(multiple_FFF) > 1:
            out.append('multiple_great_grand_father')
        return out, out_dict

    # I-step: Utility functions

    def kinship_neighbourhood(self, PID1):
        out = []
        if self.nodes[PID1]['status'] == 0:
            print('Node %d is redundant' % PID1)
            return out
        for PID2, edge_key2attr_dict in self[PID1].items():
            for edge_key, attr_dict in edge_key2attr_dict.items():
                kin_code = attr_dict['kinship_code']
                d = self.kin_neighbor_distance_threshold
                if kin_code.kin_distance is None:
                    continue
                if kin_code.kin_distance <= d:  # judge if kin_status is active
                    name = self.nodes[PID2]['c_name_chn']
                    k12 = attr_dict['kinship_code']
                    neighbor = (PID2, k12.to_str(), name)  # type --> (int, str, str)
                    out.append(neighbor)
        return out

    @property
    def name2PIDs(self):
        out = defaultdict(set)
        name_list = nx.get_node_attributes(self, 'c_name_chn')
        for PID, name in name_list.items():
            if self.nodes[PID]['status'] == 1:
                out[name].add(PID)
        out = dict((name, PID) for name, PID in out.items() if len(PID) > 1)
        return out

    def score_function(self, PID1, PID2, w1=3, w2=2, w3=2, w4=1):
        sex1 = self.nodes[PID1]['c_female']
        sex2 = self.nodes[PID2]['c_female']
        PID1_neighbour = self.kinship_neighbourhood(PID1)
        PID2_neighbour = self.kinship_neighbourhood(PID2)
        if not PID1_neighbour:
            score = 0
            overlap_dict = {'cij_c': [], 'cij_g': [], 'nij_c': [], 'nij_g': []}
            return score, overlap_dict
        # characteristic_relatives_inverse = ['S', 'D',
        #                                     'DD', 'SS', 'SD', 'DS',
        #                                     'SSS', 'SSD', 'SDS', 'DSS', 'SDD', 'DSD', 'DDS', 'DDD']
        # (1+2) Common characteristic/general relatives
        common_characteristic_relatives = []
        common_general_relatives = []
        for PID, code, name in PID1_neighbour:
            PID2_neighbour2_PID_code = [(PID, code) for PID, code, _ in PID2_neighbour]
            if code in self.characteristic_relatives_inverse:
                if sex1 == 'm' and sex2 == 'm':
                    if (PID, code) in PID2_neighbour2_PID_code:
                        common_characteristic_relatives.append((code, PID, name))
            else:
                if (PID, code) in PID2_neighbour2_PID_code:
                    common_general_relatives.append((code, PID, name))

        # (3+4) Name-matched characteristic/general relatives
        name_matched_characteristic_relatives = []
        name_matched_general_relatives = []
        P1s = pd.DataFrame(PID1_neighbour)[0].to_list()
        for PID, code, name in PID1_neighbour:
            # 如何把PID1邻居中的节点从PID2的邻居中剔除掉？？？ ，以避免 cij_g 和 nig_g同时出现一个信息的情况。
            PID2_neighbour2_code_name = [(code, name) for PID, code, name in PID2_neighbour if PID not in P1s]
            if code in self.characteristic_relatives_inverse:
                if sex1 == 'm' and sex2 == 'm':
                    if (code, name) in PID2_neighbour2_code_name:
                        name_matched_characteristic_relatives.append((code, PID, name))
            else:
                if (code, name) in PID2_neighbour2_code_name:
                    name_matched_general_relatives.append((code, PID, name))
        score = w1 * len(common_characteristic_relatives) + w2 * len(common_general_relatives) + \
                w3 * len(name_matched_characteristic_relatives) + w4 * len(name_matched_general_relatives)
        overlap_dict = {'cij_c': common_characteristic_relatives, 'cij_g': common_general_relatives,
                        'nij_c': name_matched_characteristic_relatives, 'nij_g': name_matched_general_relatives}
        return score, overlap_dict

    def i_step_matching_generating(self, name, PID_list):
        merging_rule = []
        for PID1, PID2 in combinations(PID_list, 2):
            if PID1 > PID2:
                PID1, PID2 = PID2, PID1
            w1, w2, w3, w4 = self.score_weight
            score, overlap_dict = self.score_function(PID1, PID2, w1=w1, w2=w2, w3=w3, w4=w4)
            # overlap_dict log !!!!!!!!
            merging_rule.append((name, PID1, PID2, score, overlap_dict))
        if merging_rule:
            merging_rule_df = pd.DataFrame(merging_rule, columns=('name', 'PID1', 'PID2', 'score', 'overlap_dict'))
            tau = self.score_threshold
            merging_rule_df = merging_rule_df[merging_rule_df['score'] >= tau]
            recommendation_df = merging_rule_df[merging_rule_df['score'] < tau]
            if len(merging_rule_df) > 0:
                merging_rule_df.sort_values(by='score', ascending=False, inplace=True)
                merging_rule = np.array(merging_rule_df).tolist()
                if len(recommendation_df) > 0:
                    recommendation_df.sort_values(by='score', ascending=False, inplace=True)
                    recommendation_list = np.array(recommendation_df).tolist()
                    return merging_rule, recommendation_list
                else:
                    return merging_rule, []
            else:
                return [], []
        else:
            return [], []

    def i_step_merging(self, merging_rule, iteration=1):
        successful_merging_list = []
        fail_merging_list = []
        if merging_rule is not None:
            for name, PID1, PID2, score, _ in merging_rule:
                if (self.has_node(PID1) and self.has_node(PID2)
                        and self.nodes[PID1]['status'] == 1 and self.nodes[PID2]['status'] == 1):
                    new_PID = self.min_PID - 1
                    self.min_PID -= 1
                    self.add_node(new_PID)
                    self.nodes[new_PID]['c_name_chn'] = name
                    # judge if node's attrs are consistent after merging
                    is_node_attr_consistent, attrs = self.i_step_merging_node_attr(new_PID, PID1, PID2)
                    if is_node_attr_consistent:
                        # self.nodes[PID1]['status'] = 0
                        # self.nodes[PID2]['status'] = 0
                        # self.min_PID -= 1
                        pass
                    else:
                        self.min_PID += 1
                        # report to 'node operation.txt'
                        operation = 'recommendation'
                        # o_type = 'inconsistent-node'
                        o_type = attrs
                        print2(iteration, operation, o_type, name, -99999, PID1, PID2, score,
                               file='node operation.txt', mode='a')
                        fail_case = (name, o_type, PID1, PID2, score)
                        fail_merging_list.append(fail_case)
                        # perform freeze operation
                        self.i_step_freeze(to_node=new_PID)
                        continue
                    # judge if kinship are consistent after merging
                    self.i_step_merging_edge_attr(from_node=PID1, to_node=new_PID)
                    self.i_step_merging_edge_attr(from_node=PID2, to_node=new_PID)
                    is_edge_consistent = self.is_edge_consistent_in_local_network(new_PID)
                    if len(is_edge_consistent):  # if edges with contradiction, fail to merge, then freeze
                        self.min_PID += 1
                        # report to 'node operation.txt'
                        operation = 'recommendation'
                        # o_type = 'inconsistent-edge'
                        o_type = str(is_edge_consistent)
                        print2(iteration, operation, o_type, name, -99999, PID1, PID2, score,
                               file='node operation.txt', mode='a')
                        fail_case = (name, o_type, PID1, PID2, score)
                        fail_merging_list.append(fail_case)
                        # perform freeze operation
                        self.i_step_freeze(to_node=new_PID)
                    else:  # else, successfully merge, remove PID1, PID2 from network
                        successful_case = (name, new_PID, PID1, PID2, score)
                        successful_merging_list.append(successful_case)
                        self.remove_node(PID1)
                        self.remove_node(PID2)
                        # report to 'node operation.txt'
                        operation = 'merging'
                        o_type = 'consistent'
                        print2(iteration, operation, o_type, name, new_PID, PID1, PID2, score,
                               file='node operation.txt', mode='a')
                else:
                    pass
                    # fail_case = (name, 'redundant', PID1, PID2, score)
                    # fail_merging_list.append(fail_case)
        return successful_merging_list, fail_merging_list

    def i_step_freeze(self, to_node):
        self.remove_node(to_node)

    def is_node_feature_consistent(self, PID1, PID2, attr: str = 'c_birthyear'):
        attr1 = nx.get_node_attributes(self, attr)[PID1]
        attr2 = nx.get_node_attributes(self, attr)[PID2]
        if not attr1:
            return True, attr2
        if not attr2:
            return True, attr1
        if attr == 'c_dy':
            c_dy1 = int(attr1)
            c_dy2 = int(attr2)
            if min(c_dy1, c_dy2) == 0:
                return True, max(c_dy1, c_dy2)
        if attr == 'c_birthyear' or attr == 'c_deathyear':
            attr1_list = []
            attr2_list = []
            for x in attr1.split(sep='-'):
                x = int(x)
                attr1_list.append(x)
            for x in attr2.split(sep='-'):
                x = int(x)
                attr2_list.append(x)
            attr1_min = min(attr1_list)
            attr1_max = max(attr1_list)
            attr2_min = min(attr2_list)
            attr2_max = max(attr2_list)
            age_min = abs(attr1_min - attr2_min)
            age_max = abs(attr1_max - attr2_max)
            if age_min < 30:
                if age_max < 30:
                    return True, '%d-%d' % (min(attr1_min, attr2_min), max(attr1_max, attr2_max))
            return False, ''
        elif attr1 == attr2:
            return True, attr1
        else:
            return False, ''

    def i_step_merging_node_attr(self, new_PID, PID1, PID2):
        """

        :param new_PID:
        :param PID1:
        :param PID2:
        :return: 0 if inconsistent; 1 if consistent
        """
        check, new_status = self.is_node_feature_consistent(PID1, PID2, attr='status')
        if check:
            self.nodes[new_PID]['status'] = new_status
        else:
            return 0, 'different status'

        check, new_gender = self.is_node_feature_consistent(PID1, PID2, attr='c_female')
        if check:
            self.nodes[new_PID]['c_female'] = new_gender
        else:
            return 0, 'different gender'

        # check, new_c_dy = self.is_node_feature_consistent(PID1, PID2, attr='c_dy')
        # if check:
        #     self.nodes[new_PID]['c_dy'] = new_c_dy
        # else:
        #     return 0, 'different dynasty'

        check, new_surname = self.is_node_feature_consistent(PID1, PID2, attr='c_surname_chn')
        if check:
            self.nodes[new_PID]['c_surname_chn'] = new_surname
        else:
            return 0, 'different surname'

        check, new_mingzi = self.is_node_feature_consistent(PID1, PID2, attr='c_mingzi_chn')
        if check:
            self.nodes[new_PID]['c_mingzi_chn'] = new_mingzi
        else:
            return 0, 'different mingzi'

        check, new_birth_year = self.is_node_feature_consistent(PID1, PID2, attr='c_birthyear')
        if check:
            self.nodes[new_PID]['c_birthyear'] = new_birth_year
        else:
            return 0, 'different birth'

        check, new_death_year = self.is_node_feature_consistent(PID1, PID2, attr='c_deathyear')
        if check:
            self.nodes[new_PID]['c_deathyear'] = new_death_year
        else:
            return 0, 'different death'

        return 1, 'No difference'

    def i_step_merging_edge_attr(self, from_node, to_node):
        from_node_neighbours = self[from_node].copy()
        # from_node_neighbours = {PID: {edge_key for edge_key in edge_key2attr_dict.keys()} for PID,
        # edge_key2attr_dict in self[from_node].items()}
        # neighbor_set = {PID for PID in self.nodes[from_node]} for
        # PID in neighbor_set: edge_key2attr_dict = self[from_node][PID] edge_key_set = {edge_key for edge_key in
        # edge_key2attr_dict} for edge_key in edge_key_set: attr_dict = edge_key2attr_dict[edge_key]
        for PID, edge_key2attr_dict in from_node_neighbours.items():
            for _, attr_dict in edge_key2attr_dict.items():
                if self.has_edge(to_node, PID, key=None):
                    pass
                else:
                    self.add_edge(u_for_edge=to_node, v_for_edge=PID, key=None, **attr_dict)
            assert self.has_edge(PID, from_node), f'Inverse edge between {PID} and {from_node} does not exist!'
            inverse_edge_key2attr_dict = self[PID][from_node]
            for _, attr_dict in inverse_edge_key2attr_dict.items():
                if self.has_edge(PID, to_node, key=None):
                    pass
                else:
                    self.add_edge(u_for_edge=PID, v_for_edge=to_node, key=None, **attr_dict)

    def is_edge_consistent_in_local_network(self, PID1):
        out = []
        multiple_F = []
        multiple_FF = []
        multiple_FFF = []
        neighbours = list(self.neighbors(PID1))
        for PID2 in neighbours:
            edges = self.get_edge_data(PID1, PID2)
            name = self.nodes[PID2]['c_name_chn']
            # case 1: 多重边，is_compatible()
            edges_number = len(edges)
            code_list = []
            for _, attr_dict in edges.items():
                kin_code = attr_dict['kinship_code']
                code_list.append(kin_code)
                if kin_code.to_str() == 'F':
                    multiple_father = (PID1, kin_code, PID2, name)
                    multiple_F.append(multiple_father)
                if kin_code.to_str() == 'FF':
                    multiple_grand_father = (PID1, kin_code, PID2, name)
                    multiple_FF.append(multiple_grand_father)
                if kin_code.to_str() == 'FFF':
                    multiple_great_grand_father = (PID1, kin_code, PID2, name)
                    multiple_FFF.append(multiple_great_grand_father)
            if edges_number > 1:
                for pair in list(combinations(code_list, 2)):
                    judgement = is_compatible(pair[0], pair[1])
                    if judgement:
                        pass
                    else:
                        # 补充记录"多重关系矛盾" ！！！！！
                        out.append('multiple-relation-error')
                        break
        # case 2: 多个父亲（不同 PID），raise error !
        # 补充记录"多个父亲关系矛盾" ！！！！！
        if len(multiple_F) > 1:
            names = [name for _, _, _, name in multiple_F]
            # 找到所有的F的名称，如果人民都相同，则不报错，否则，记录！
            for pair in list(combinations(names, 2)):
                if pair[0] == pair[1]:
                    pass
                else:
                    out.append('multiple-father-error')
        if len(multiple_FF) > 1:
            names = [name for _, _, _, name in multiple_FF]
            for pair in list(combinations(names, 2)):
                if pair[0] == pair[1]:
                    pass
                else:
                    out.append('multiple-grand-father-error')
        if len(multiple_FFF) > 1:
            names = [name for _, _, _, name in multiple_FFF]
            for pair in list(combinations(names, 2)):
                if pair[0] == pair[1]:
                    pass
                else:
                    out.append('multiple-great-grand-father-error')
        return out

    # Data-processing-step: Utility functions

    def process_step1_gender_bidirection_inverse(self):
        # check if bi-directional; if not, add the inverse direction
        network_edge2kinship = nx.get_edge_attributes(self, 'kinship_code')
        edge2remove_gender = []
        edge2remove_not_bi_direction = []
        edge_with_code_error = []
        for edge_key, edge_kin_code in network_edge2kinship.items():
            PID1 = edge_key[0]
            PID2 = edge_key[1]
            gender_1 = self.nodes[PID1]['c_female']
            gender_2 = self.nodes[PID2]['c_female']
            # if the kinship code can be parsed
            if edge_kin_code.generation is not None:
                k12_inverse_code_list = edge_kin_code.get_inverse_kinship(gender_0=gender_1)
                k21 = ''.join(code.to_str() for code in k12_inverse_code_list)
                k21_code = ComplexKinshipCode(k21)
                if gender_2 != edge_kin_code.gender_pair[1] or gender_1 != k21_code.gender_pair[1]:
                    # report to 'contradiction.txt'
                    details = 'cleaning:'
                    self.contradiction_report_2(None, PID1, PID2, error_type='error-in-gender', details=details)
                    self.contradiction_report_2(None, PID2, PID1, error_type='error-in-gender', details=details)
                    # 记录下来，将来一起remove，
                    # print('error in gender' + str((PID1, PID2)))
                    edge2remove_gender.append((PID1, PID2))
                    edge2remove_gender.append((PID2, PID1))
                    continue
                if self.has_edge(PID2, PID1):
                    # check kinship code and inverse code
                    for key, edge2attr in self.get_edge_data(PID2, PID1).items():
                        if is_compatible(k21, edge2attr['kinship_code']):
                            pass
                        else:
                            redundant_key = (PID2, PID1, key)
                            # del network_edge2kinship[redundant_key]
                            self.edges[edge_key]['error_type'].append('error-in-kin-code')
                            self.edges[edge_key]['status'] = 0
                            # report to 'contradiction.txt'
                            details = 'cleaning:' + str(
                                (edge_key[0], edge_key[1], edge_key[2], edge_kin_code.to_str())) + \
                                      ' conflicts with its inverse code ' + \
                                      str((redundant_key[0], redundant_key[1], redundant_key[2], k21))
                            self.contradiction_report_2(None, PID1, PID2, 'error-in-kin-code', details=details)
                            edge_with_code_error.append((PID1, PID2))
                            # self.contradiction_report_2(None, PID2, PID1, 'error-in-kin-code', details=details)
                else:
                    # report to 'contradiction.txt'
                    # self.edges[edge_key]['error_type'].append('error-in-kin-code')
                    # self.edges[edge_key]['status'] = 0
                    details = 'cleaning: only one-way relationship'
                    self.contradiction_report_2(None, PID1, PID2, error_type='error-in-kin-code',
                                                details=details)
                    # 记录下来，将来一起remove，
                    # print('only one:' + str((PID1, PID2)))
                    edge2remove_not_bi_direction.append((PID1, PID2))
                    # add the inverse one
                    # code: ComplexKinshipCode = ComplexKinshipCode(k21)
                    # status = 0
                    # error_type = ['error-in-kin-code']
                    # self.add_edge(PID2, PID1, kinship_code=code, status=status, round=0, error_type=error_type)
            else:
                if not self.has_edge(PID2, PID1):
                    details = 'cleaning: only one-way relationship'
                    self.contradiction_report_2(None, PID1, PID2, error_type='error-in-kin-code',
                                                details=details)
                    # print('only one:' + str((PID1, PID2)))
                    edge2remove_not_bi_direction.append((PID1, PID2))
        edge2remove = edge2remove_gender + edge2remove_not_bi_direction
        self.remove_edges_from(edge2remove)
        print2(f'----------------------------------------------------', mode='a', add_time=False)
        print2(f' ', mode='a', add_time=True)
        print2(f'Start data process ...', mode='a', add_time=False)
        print2(f'{len(edge2remove)} edges are removed due to contradiction in gender or one-way', mode='a',
               add_time=False)
        print2(f'{len(edge2remove_gender)} edges with contradiction in gender', mode='a', add_time=False)
        print2(f'{len(edge2remove_not_bi_direction)} edges only have one-way relation', mode='a', add_time=False)
        print2(f'{len(edge_with_code_error)} edges conflict with their inverse kin codes', mode='a', add_time=False)
        # return edge2remove

    def process_step2_multiple_fathers(self, PID_list):
        edge_with_multiple_father = []
        for PID in PID_list:
            is_multiple, multiple_dict = self.is_multiple_father(PID)
            if is_multiple:
                for multiple_type in is_multiple:
                    details = 'cleaning:' + str(multiple_dict[multiple_type])
                    for PID1, PID2, k12 in multiple_dict[multiple_type]:
                        # print(f'{PID1},{PID2},{k12}')
                        for key, value in self.get_edge_data(PID1, PID2).items():
                            value['error_type'].append(multiple_type)
                            value['status'] = 0
                        # report to 'contradiction.txt'
                        self.contradiction_report_2(None, PID1, PID2, error_type='multiple-father', details=details)
                        edge_with_multiple_father.append((PID1, PID2, k12))
        print2(f'{len(edge_with_multiple_father)} edges with contradiction in multiple fathers', mode='a',
               add_time=False)

    def process_step3_multiple_relations(self):
        edge_with_multiple_relations = []
        network_edge2kinship = nx.get_edge_attributes(self, 'kinship_code')
        network_edge_key_df = pd.DataFrame(network_edge2kinship.keys(), columns=("PID1", "PID2", "key"))
        multiple_relation_df = network_edge_key_df.loc[(network_edge_key_df["key"] > 0) & (network_edge_key_df["key"] < 2)]
        multiple_relation_list = np.array(multiple_relation_df).tolist()
        multiple_relation_list = set((PID1, PID2, key) for PID1, PID2, key in multiple_relation_list)
        for PID1, PID2, _ in multiple_relation_list:
            all_paths = self.get_edge_data(PID1, PID2)
            details = 'cleaning:'
            for key, value in all_paths.items():
                value['error_type'].append('multiple-relation')
                value['status'] = 0
                kinship_name = value['kin_name']
                details += f'{kinship_name}<-?->'
                # report to 'contradiction.txt'
            self.contradiction_report_2(None, PID1, PID2, error_type='multiple-relation', details=details)
            edge_with_multiple_relations.append((PID1, PID2))
        print2(f'{len(edge_with_multiple_relations)} edges with contradiction in multiple relations', mode='a',
               add_time=False)

    def process_seniority_dict_generator(self, PID1, local_network_sub, PIDs2seniority):
        # seniroirty_list = []
        local_network_sub_dim = len(local_network_sub)
        if local_network_sub_dim > 1:
            for i in range(0, local_network_sub_dim - 1):
                for j in range(i + 1, local_network_sub_dim):
                    code_i = local_network_sub[i][1]
                    code_j = local_network_sub[j][1]
                    PID_i = local_network_sub[i][0]
                    PID_j = local_network_sub[j][0]
                    # if PID_i == 305 or PID_i == 3167:
                    #     print(PID_i,code_i, PID_j,code_j)
                    # print(PID_i, code_i, PID_j, code_j)
                    if int(code_i[1:]) < int(code_j[1:]):
                        has_report = False
                        # judgeable_seniority1 = (PID_i, PID_j, '-')
                        # judgeable_seniority2 = (PID_j, PID_i, '+')
                        if PIDs2seniority.__contains__((PID_i, PID_j)):
                            if PIDs2seniority[(PID_i, PID_j)] != '-':
                                # report to 'contradiction.txt' ,
                                error_type = 'error-in-seniority'
                                if self.has_edge(PID_i, PID_j):
                                    code_i_j = self.get_edge_data(PID_i, PID_j, 0)['kinship_code'].to_str()
                                    details = f'cleaning:{(PID1, PID_i, code_i)},{(PID1, PID_j, code_j)}, {(PID_i, PID_j, code_i_j)}'
                                else:
                                    sign = '-'
                                    details = f'cleaning:{(PID1, PID_i, code_i)},{(PID1, PID_j, code_j)}, {(PID_i, PID_j, sign)}'

                                has_not_report1 = has_not_report2 = False
                                all_paths = self.get_edge_data(PID1, PID_i)
                                for key, value in all_paths.items():
                                    if 'multiple-seniority' not in value['error_type']:
                                        value['error_type'].append('multiple-seniority')
                                        value['status'] = 0
                                        has_not_report1 = True
                                all_paths = self.get_edge_data(PID1, PID_j)
                                for key, value in all_paths.items():
                                    if 'multiple-seniority' not in value['error_type']:
                                        value['error_type'].append('multiple-seniority')
                                        value['status'] = 0
                                        has_not_report2 = True
                                if has_not_report1 and has_not_report2:
                                    self.contradiction_report_3(PID1, PID_i, PID_j, error_type, details)
                                    has_report = True
                        else:
                            PIDs2seniority[(PID_i, PID_j)] = f'-{(PID_i, code_i, PID_j, code_j)}'

                        if PIDs2seniority.__contains__((PID_j, PID_i)):
                            if PIDs2seniority[(PID_j, PID_i)] != '+':
                                # report to 'contradiction.txt' ,
                                error_type = 'error-in-seniority'
                                if self.has_edge(PID_j, PID_i):
                                    code_j_i = self.get_edge_data(PID_j, PID_i, 0)['kinship_code'].to_str()
                                    details = f'cleaning:{(PID1, PID_i, code_i)},{(PID1, PID_j, code_j)}, {(PID_i, PID_j, code_j_i)}'
                                else:
                                    sign = '+'
                                    details = f'cleaning:{(PID1, PID_i, code_i)},{(PID1, PID_j, code_j)}, {(PID_i, PID_j, sign)}'

                                has_not_report3 = has_not_report4 = False
                                all_paths = self.get_edge_data(PID1, PID_i)
                                for key, value in all_paths.items():
                                    if 'multiple-seniority' not in value['error_type']:
                                        value['error_type'].append('multiple-seniority')
                                        value['status'] = 0
                                        has_not_report3 = True
                                all_paths = self.get_edge_data(PID1, PID_j)
                                for key, value in all_paths.items():
                                    if 'multiple-seniority' not in value['error_type']:
                                        value['error_type'].append('multiple-seniority')
                                        value['status'] = 0
                                        has_not_report4 = True
                                if has_not_report4 and has_not_report3 and (not has_report):
                                    self.contradiction_report_3(PID1, PID_i, PID_j, error_type, details)
                        else:
                            PIDs2seniority[(PID_j, PID_i)] = f'+{(PID_i, code_i, PID_j, code_j)}'
                    elif int(code_i[1:]) > int(code_j[1:]):
                        has_report = False
                        # judgeable_seniority1 = (PID_i, PID_j, '+')
                        # judgeable_seniority2 = (PID_j, PID_i, '-')
                        if PIDs2seniority.__contains__((PID_i, PID_j)):
                            if PIDs2seniority[(PID_i, PID_j)] != '+':
                                # report to 'contradiction.txt' ,
                                error_type = 'error-in-seniority'
                                if self.has_edge(PID_i, PID_j):
                                    code_i_j = self.get_edge_data(PID_i, PID_j, 0)['kinship_code'].to_str()
                                    details = f'cleaning:{(PID1, PID_i, code_i)},{(PID1, PID_j, code_j)}, {(PID_i, PID_j, code_i_j)}'
                                else:
                                    sign = '+'
                                    details = f'cleaning:{(PID1, PID_i, code_i)},{(PID1, PID_j, code_j)}, {(PID_i, PID_j, sign)}'

                                has_not_report1 = has_not_report2 = False
                                all_paths = self.get_edge_data(PID1, PID_i)
                                for key, value in all_paths.items():
                                    if 'multiple-seniority' not in value['error_type']:
                                        value['error_type'].append('multiple-seniority')
                                        value['status'] = 0
                                        has_not_report1 = True
                                all_paths = self.get_edge_data(PID1, PID_j)
                                for key, value in all_paths.items():
                                    if 'multiple-seniority' not in value['error_type']:
                                        value['error_type'].append('multiple-seniority')
                                        value['status'] = 0
                                        has_not_report2 = True
                                if has_not_report1 and has_not_report2:
                                    self.contradiction_report_3(PID1, PID_i, PID_j, error_type, details)
                                    has_report = True
                        else:
                            PIDs2seniority[(PID_i, PID_j)] = f'+{(PID_i, code_i, PID_j, code_j)}'

                        if PIDs2seniority.__contains__((PID_j, PID_i)):
                            if PIDs2seniority[(PID_j, PID_i)] != '-':
                                # report to 'contradiction.txt' ,
                                error_type = 'error-in-seniority'
                                if self.has_edge(PID_j, PID_i):
                                    code_j_i = self.get_edge_data(PID_j, PID_i, 0)['kinship_code'].to_str()
                                    details = f'cleaning:{(PID1, PID_i, code_i)},{(PID1, PID_j, code_j)}, {(PID_i, PID_j, code_j_i)}'
                                else:
                                    sign = '-'
                                    details = f'cleaning:{(PID1, PID_i, code_i)},{(PID1, PID_j, code_j)}, {(PID_i, PID_j, sign)}'

                                has_not_report3 = has_not_report4 = False
                                all_paths = self.get_edge_data(PID1, PID_i)
                                for key, value in all_paths.items():
                                    if 'multiple-seniority' not in value['error_type']:
                                        value['error_type'].append('multiple-seniority')
                                        value['status'] = 0
                                        has_not_report3 = True
                                all_paths = self.get_edge_data(PID1, PID_j)
                                for key, value in all_paths.items():
                                    if 'multiple-seniority' not in value['error_type']:
                                        value['error_type'].append('multiple-seniority')
                                        value['status'] = 0
                                        has_not_report4 = True
                                if has_not_report4 and has_not_report3 and (not has_report):
                                    self.contradiction_report_3(PID1, PID_i, PID_j, error_type, details)
                        else:
                            PIDs2seniority[(PID_j, PID_i)] = f'-{(PID_i, code_i, PID_j, code_j)}'
                    else:  # int(code_i[1:]) = int(code_j[1:])
                        if self.nodes[PID_i]['c_name_chn'] != self.nodes[PID_j]['c_name_chn']:
                            # report to 'contradiction.txt' ,
                            error_type = 'error-in-seniority'
                            details = f'{(PID1, PID_i, code_i)},{(PID1, PID_j, code_j)}'
                            self.contradiction_report_3(PID1, PID_i, PID_j, error_type, details)
        else:
            pass
        return PIDs2seniority

    def process_step4_multiple_seniority(self):
        PIDs2seniority_dict = {}
        unjudgeable_seniority = ["WB+", "Z+H", "FB+", "WZ+", "M+", "FFB+", "W2B+", "Z-H", "WB-", "WZ-", "FB–", "FFB–"]
        PID_list = set(self.nodes)
        for PID1 in PID_list:
            local_network = self.local_network(PID1, active_edge=False)

            # deal with pattern 'S+digit',  re.match('D\d{1,2}', k12)
            local_network_son = [(PID2, k12) for PID2, k12 in local_network if k12[0] == 'S' and k12[1:].isdigit()]
            PIDs2seniority_dict = self.process_seniority_dict_generator(PID1, local_network_son, PIDs2seniority_dict)

            # deal with pattern 'D+digit'
            local_network_daughter = [(PID2, k12) for PID2, k12 in local_network if k12[0] == 'D' and k12[1:].isdigit()]
            PIDs2seniority_dict = self.process_seniority_dict_generator(PID1, local_network_daughter, PIDs2seniority_dict)

            # deal with pattern '+', '-'
            for PID2, k12 in local_network:
                code = ComplexKinshipCode(k12)
                if code.generation == 0 and (k12 not in unjudgeable_seniority):
                    if '+' in k12:
                        # "+" means : PID1 has a older relative PID2
                        if PIDs2seniority_dict.__contains__((PID1, PID2)):
                            if '+' not in PIDs2seniority_dict[(PID1, PID2)]:
                                all_paths = self.get_edge_data(PID1, PID2)
                                for key, value in all_paths.items():
                                    value['error_type'].append('multiple-seniority')
                                    value['status'] = 0
                                # report to 'contradiction.txt' ,
                                error_type = 'error-in-seniority'
                                details = f'cleaning:{(PID1, PID2, k12)}, {(PID1, PID2, PIDs2seniority_dict[(PID1, PID2)])}'
                                self.contradiction_report_2(None, PID1, PID2, error_type, details)
                        else:
                            PIDs2seniority_dict[(PID1, PID2)] = '+'
                    if '-' in k12:
                        # "-" means : PID1 has a yonger relative PID2
                        if PIDs2seniority_dict.__contains__((PID1, PID2)):
                            if '-' not in PIDs2seniority_dict[(PID1, PID2)]:
                                all_paths = self.get_edge_data(PID1, PID2)
                                for key, value in all_paths.items():
                                    value['error_type'].append('multiple-seniority')
                                    value['status'] = 0
                                # report to 'contradiction.txt' ,
                                error_type = 'error-in-seniority'
                                details = f'cleaning:{(PID1, PID2, k12)}, {(PID1, PID2, PIDs2seniority_dict[(PID1, PID2)])}'
                                self.contradiction_report_2(None, PID1, PID2, error_type, details)
                        else:
                            PIDs2seniority_dict[(PID1, PID2)] = '-'
        print2(f' ', mode='a', add_time=True)
        print2(f'End data process ...', mode='a', add_time=False)
        print2(f'----------------------------------------------------', mode='a', add_time=False)
        return PIDs2seniority_dict

    def output_node(self, file_name):
        print2('c_personid', 'c_name_chn', 'c_female', 'c_surname_chn', 'c_mingzi_chn', 'c_birthyear', 'c_deathyear',
               file=file_name, add_time=False, mode='w', print2console=False)
        nodes = self.nodes
        for PID in nodes:
            node_attr = self.nodes[PID]
            out = [PID, node_attr['c_name_chn'], node_attr['c_female'], node_attr['c_surname_chn'],
                   node_attr['c_mingzi_chn'], node_attr['c_birthyear'], node_attr['c_deathyear']]
            print2(*out, file=file_name, add_time=False, mode='a', print2console=False)

    def output_edge(self, file_name):
        print2('PID1', 'PID2', 'c_kin_code', 'c_kin_name', 'status', 'round', 'error_type', 'gen_diff',
               file=file_name, add_time=False, mode='w', print2console=False)
        nodes = self.nodes
        for PID1 in nodes:
            for PID2, edge_key2attr_dict in self[PID1].items():
                for edge_key, attr_dict in edge_key2attr_dict.items():
                    kinship_code = attr_dict['kinship_code']
                    if attr_dict.__contains__('kin_name'):
                        kin_name = attr_dict['kin_name']
                    else:
                        kin_name = ''
                    if attr_dict.__contains__('gen_diff'):
                        gen_diff = attr_dict['gen_diff']
                    else:
                        gen_diff = None
                    status = attr_dict['status']
                    round = attr_dict['round']
                    error_type = attr_dict['error_type']
                    out = [PID1, PID2, kinship_code.to_str(), kin_name, status, round, error_type, gen_diff]
                    print2(*out, file=file_name, add_time=False, mode='a', print2console=False)

    def input_network(self, input_nodes, input_edges):
        # adding nodes to kinship network
        self.add_nodes_from(input_nodes['c_personid'])
        for col in input_nodes.columns[1:]:
            for PID, value in zip(input_nodes['c_personid'], input_nodes[col]):
                self.nodes[PID][col] = value
                # cls.nodes[PID]['status'] = 1

        for PID1, PID2, KinCode, KinName, status, round, error_type in zip(
                input_edges['PID1'], input_edges['PID2'],
                input_edges['c_kin_code'], input_edges['c_kin_name'],
                input_edges['status'], input_edges['round'], input_edges['error_type']):
            code: ComplexKinshipCode = ComplexKinshipCode(KinCode)
            self.add_edge(PID1, PID2, kinship_code=code, kin_name=KinName, status=status, round=round, error_type=error_type, gen_diff=code.generation)

    # def o_step(self, iteration=1):
    #     print2(f'Start O-step, iteration {iteration}', mode='a')
    #     # for each local network E_i
    #     # (1) scanning
    #     # (2) reasoning
    #     # (3) security check
    #     # (4) operating
    #     # (5) pruning
    #     edge_adding_num = 0
    #     node_close_num = 0
    #     connected_area_set = list(nx.weakly_connected_components(self))  # block each connected area
    #     for connection in tqdm(connected_area_set):
    #         sub_network = self.subgraph(connection)  # create sub-graph w.r.t given block
    #         sub_PID_list = set(sub_network.nodes)
    #         for PID1 in sub_PID_list:
    #             edge_adding_list = self.o_step_reasoning_operation(PID1=PID1, iteration=iteration)
    #             if edge_adding_list:
    #                 # print(f'Adding edges in node {PID1}: {edge_adding_list}')
    #                 edge_adding_num += len(edge_adding_list)
    #                 pruning_list = self.o_step_pruning(edge_adding_list, iteration=iteration)
    #                 # considering the bi-direction fashion, only check one direction
    #                 # pruning_list = self.o_step_pruning(edge_adding_list[::2], iteration=iteration)
    #                 if pruning_list:
    #                     # print(f'Closing edges in node {PID1}: {pruning_list}')
    #                     node_close_num += len(pruning_list)
    #             # print('Adding %d edges in node %d' % (len(edge_adding_list), PID))
    #             # print('Closing %d edges in round %d' % (len(pruning_list), iteration))
    #     # print2(self.number_of_nodes(), self.number_of_edges(), file='summary log.txt', mode='a')
    #     print2(f'End O-step, iteration {iteration}', mode='a')
    #     return edge_adding_num, node_close_num

    # def o_step_is_edge_redundant(self, PID1, PID2):
    #     paths = nx.all_simple_edge_paths(G=self, source=PID1, target=PID2)  # a generator,
    #     paths = list(paths)
    #     if paths is None:
    #         return -1, 'No path', ''
    #     if len(paths) == 1:
    #         return 0, 'Only one path', ''
    #     else:
    #         edge_redundant = nx.all_simple_edge_paths(G=self, source=PID1, target=PID2, cutoff=1)  # a generator,
    #         edge_redundant = list(edge_redundant)  # returned list is sorted according to path length
    #         edge_redundant = edge_redundant[0][0]  # choose the shortest path length, e.g. len()=2
    #         edge_redundant_kinship = self.edges[edge_redundant]['kinship_code']  # extract code from specific path
    #         edge_redundant_kinship_code = edge_redundant_kinship.kinship_code
    #         edge_redundant_kinship_distance = edge_redundant_kinship.kin_distance
    #         if edge_redundant_kinship_distance == 1:
    #             return 0, 'Not redundant', ''
    #         else:
    #         # 计算每个path的 kin distance之和，kin distance最大为可能的冗余边，然后判断该冗余边的是否可以由其他边拼接得到！
    #         # paths_kinship = []
    #             for p in paths:
    #                 if len(p) > 1:
    #                     edge_in_path_kinship_code = ''
    #                     edge_in_path_kinship_distance = 0
    #                     for e_k1_k2 in p:
    #                         kin_e_k1_k2 = self.edges[e_k1_k2]['kinship_code']
    #                         edge_in_path_kinship_code += kin_e_k1_k2.kinship_code
    #                         edge_in_path_kinship_distance += kin_e_k1_k2.kin_distance
    #                     # paths_kinship.append(edge_in_path_kinship_code)  # possible kinship paths between two nodes
    #                     if edge_in_path_kinship_distance == edge_redundant_kinship_distance:
    #                         compatible_check = is_compatible(edge_redundant_kinship_code, edge_in_path_kinship_code)
    #                         if compatible_check:
    #                             # 记录是哪条路径的存在导致该边是冗余的
    #                             return 1, edge_redundant_kinship_code, edge_in_path_kinship_code
    #                         else:
    #                             return -1, edge_redundant_kinship_code, edge_in_path_kinship_code  # error

    # # summary log
    # node_total_num = kinship_network.number_of_nodes()
    # edge_total_num = kinship_network.number_of_edges()
    #
    # edge_i_active_num = kinship_network.get_num_edge_active()
    #
    # complexity_total_kin_distance = kinship_network.get_total_kin_distance()
    # complexity_ave_kin_distance = complexity_total_kin_distance / kinship_network.get_num_edge_active()
    # complexity_ave_kin_distance = round(complexity_ave_kin_distance, 5)
    #
    # connect_area_total_num, connect_area_max_size = kinship_network.get_num_community()
    # connect_area_ave_size = connect_area_total_num / node_total_num
    # connect_area_ave_size = round(connect_area_ave_size, 5)
    #
    # # print2file
    # print2(iteration, node_total_num, node_adding_num, node_removing_num,
    #        edge_total_num, edge_o_active_num, edge_i_active_num, edge_adding_num, node_close_num,
    #        complexity_total_kin_distance, complexity_ave_kin_distance,
    #        connect_area_total_num, connect_area_ave_size, connect_area_max_size,
    #        file='summary log.txt', mode='a', print2console=False)