#
import re

from example_CBDB.Kin_00_Config import *
import sqlite3


# (1) 删除 空格等不常用字符
# (2) 检测单向边，如果是，则补齐反向！！！！
#
# ####################################################################################################
# # Data loading and cleaning
#
# # input: data (.db file)
# # output: tables (BIOG_MAIN, KIN_DATA, KIN_CODE, ALTNAME_DATA)
# ####################################################################################################
def input_data_preprocessing(data_dir, data='CBDB_20201110.db'):
    CBDB_file = os.path.join(data_dir, data)

    with closing(sqlite3.connect(CBDB_file)) as conn:
        with conn:  # auto-commits
            with closing(conn.cursor()) as cursor:  # auto-closes
                table_names = [
                    row[0]
                    for row in cursor.execute(
                        "select name from sqlite_master where type='table' order by name"
                    ).fetchall()
                ]
                print(table_names)
            BIOG_MAIN = pd.read_sql("SELECT * FROM BIOG_MAIN", conn, coerce_float=False)

            KIN_DATA = pd.read_sql("SELECT * FROM KIN_DATA", conn, coerce_float=False)
            KIN_CODE = pd.read_sql("SELECT * FROM KINSHIP_CODES", conn, coerce_float=False)

    # ####################################################################################################
    # invalid characters setting
    # ####################################################################################################
    invalid_char = set()
    # from name
    invalid_char |= {' ', '(', '+', '-', '<', '>', '?', '[', 'Í', 'ú', 'Γ', 'α', 'π', '■', '□', '○', '\u3000', 'テ', 'ㄔ',
                     'ㄘ', 'ㄧ', '︱', '︵', '﹀', '＊', '＋', '？'}

    # from mingzi
    invalid_char |= {' ', '(', '+', '-', '?', '[', 'Í', 'ú', 'Γ', 'α', 'π', '■', '□', '\u3000', 'テ', 'ㄔ', 'ㄘ', 'ㄧ', '︱',
                     '︵', '﹀', '＊', '＋', '？'}

    # from altname
    invalid_char |= {' ', '#', "'", '.', '/', '=', '?', '\\', '\x7f', '·', 'š', '‧', '□', '▫', '，', '；', '？', '�'}

    def modify_word(word):
        out = word
        if out is not None:
            out = out.replace('（', '(').replace('）', ')')

            out = re.sub(r'\(.*\)', '', out)
            out = re.sub(r'{.*}', '', out)
            out = re.sub(r'\[.*\]', '', out)
            out = re.sub(r'【.*】', '', out)
            out = re.sub(r'[a-zA-Z0-9]', '', out)

            for char in invalid_char:
                out = out.replace(char, '')
        return out

    # from kincode
    invalid_kin_code = {' \xa0', '\xa0', ' (only surviving son)', ' (only son)', ' (eldest surviving son)', '(male)',
                        '(female)', '(C)', ' (claimed)', ' (apical)', ' (only daughter)'}

    def modify_kin_code(word):
        out = word
        for char in invalid_kin_code:
            out = out.replace(char, '')
        return out

    # ####################################################################################################
    # input: BIOG_MAIN， DYNASTIES（朝代與日期的代碼）
    # output: input_nodes
    # ####################################################################################################
    node_vars = ['c_personid', 'c_name_chn', 'c_female', 'c_birthyear', 'c_deathyear', 'c_dy', 'c_surname_chn',
                 'c_mingzi_chn']
    input_nodes = BIOG_MAIN[node_vars].copy()

    # modify name with invalid chars
    input_nodes['c_name_chn'] = [modify_word(name) for name in input_nodes['c_name_chn']]
    input_nodes['c_surname_chn'] = [modify_word(name) for name in input_nodes['c_surname_chn']]
    input_nodes['c_mingzi_chn'] = [modify_word(name) for name in input_nodes['c_mingzi_chn']]

    # drop rows with invalid name
    row_with_invalid_name = []
    for name in input_nodes['c_name_chn']:
        if (len(name) == 1 or
                name.startswith('某') or
                name.endswith('某') or
                name.endswith('氏') or
                '?' in name or
                '未詳' in name or
                '<待删除>' in name):
            row_with_invalid_name.append(False)
        else:
            row_with_invalid_name.append(True)
    input_nodes = input_nodes[row_with_invalid_name]

    # convert float to str
    input_nodes['c_birthyear'] = ['' if pd.isnull(i) else str(int(i)) for i in input_nodes['c_birthyear']]
    input_nodes['c_birthyear'] = ['' if i == '0' else i for i in input_nodes['c_birthyear']]

    input_nodes['c_deathyear'] = ['' if pd.isnull(i) else str(int(i)) for i in input_nodes['c_deathyear']]
    input_nodes['c_deathyear'] = ['' if i == '0' else i for i in input_nodes['c_deathyear']]

    input_nodes['c_dy'] = ['' if pd.isnull(i) else str(int(i)) for i in input_nodes['c_dy']]

    # convert c_female = {0, 1} to {'m', 'f'}
    input_nodes['c_female'] = ['m' if i == 0 else 'f' for i in input_nodes['c_female']]

    # ####################################################################################################
    # input: KIN_DATA, KIN_CODE
    # output: input_edges
    # ####################################################################################################
    KIN_CODE = KIN_CODE.rename(columns={"c_kincode": "c_kin_code", "c_kinrel_chn": "c_kin_name"})
    KIN_CODE['generation_diff'] = KIN_CODE['c_dwnstep'] - KIN_CODE['c_upstep']
    edge_vars = ['c_personid', 'c_kin_id', 'c_kin_code']
    input_edges = KIN_DATA[edge_vars].copy()
    input_edges = pd.merge(input_edges, KIN_CODE[['c_kinrel', 'c_kin_code', 'c_kin_name', 'generation_diff']],
                           on=['c_kin_code'], how='left')
    input_edges = input_edges.drop('c_kin_code', 1)
    input_edges = input_edges.rename(columns={"c_personid": "PID1", "c_kin_id": "PID2", "c_kinrel": "c_kin_code"})
    # set(input_edges['c_kinrel'])

    input_edges['c_kin_code'] = [modify_kin_code(name) for name in input_edges['c_kin_code']]

    # ####################################################################################################
    # input: input_nodes, input_edges
    # output: 'input_node.txt', 'input_edges.txt'
    # ####################################################################################################
    edge_with_nodes = []
    input_nodes_set = set(input_nodes['c_personid'])
    for PID1, PID2 in zip(input_edges['PID1'], input_edges['PID2']):
        if PID1 in input_nodes_set and PID2 in input_nodes_set:
            edge_with_nodes.append(True)
        else:
            edge_with_nodes.append(False)
    input_edges_out = input_edges[edge_with_nodes]
    input_edges_out.to_csv(join(data_dir, 'input_edge.txt'), sep='\t', index=False)

    node_with_edges = []
    input_nodes_set2 = set(input_edges['PID1']) | set(input_edges['PID2'])
    for PIDs in input_nodes['c_personid']:
        if PIDs in input_nodes_set2:
            node_with_edges.append(True)
        else:
            node_with_edges.append(False)
    input_nodes_out = input_nodes[node_with_edges]
    input_nodes_out.to_csv(join(data_dir, 'input_node.txt'), sep='\t', index=False)

    kin_code_out = KIN_CODE[['c_kinrel', 'c_kin_code', 'c_kin_name', 'c_dwnstep', 'c_upstep']]
    kin_code_out.to_csv(join(data_dir, 'kin_code.txt'), sep='\t', index=False)

    return input_nodes_out, input_edges_out, kin_code_out
