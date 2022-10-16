#
from example_CBDB.Kin_00_Config import *
from example_CBDB.Kin_01_Data_Cleaning import *
from example_CBDB.Kin_02_Network_Run import *

project_dir = '/Users/francis/PycharmProjects/Kinship_py'
work_dir = os.path.join(project_dir, 'example_CBDB')
data_dir = os.path.join(work_dir, 'data')
output_dir = os.path.join(work_dir, 'output')

if __name__ == '__main__':
    input_nodes, input_edges, kin_code = input_data_preprocessing(data_dir=data_dir, data='CBDB_20201110.db')
    network_building(data_dir, input_nodes, input_edges, output_dir)
