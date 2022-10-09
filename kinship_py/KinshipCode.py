from collections import deque
import re
from typing import List


class PrimaryKinshipCode:
    code_set = {'B', 'C', 'D', 'F', 'H', 'M', 'S', 'W', 'Z'}

    code2generation = {'F': -1, 'M': -1,
                       'S': 1, 'D': 1,
                       'H': 0, 'W': 0, 'C': 0,
                       'B': 0, 'Z': 0}

    code2distance = {'F': 1, 'M': 1,
                     'S': 1, 'D': 1,
                     'H': 1, 'W': 1, 'C': 1,
                     'B': 2, 'Z': 2}

    def __init__(self, kinship_code: str):
        self.kinship_code = kinship_code
        self.kinshipType = kinship_code[0]
        self.seniority = self.get_seniority()
        self.generation = self.get_generation()
        self.kin_distance = self.get_kin_distance()
        self.gender_pair = self.get_gender_pair()
        # self.inverse_kinship = self.get_inverse_kinship()

    def to_str(self):
        return self.kinship_code

    def __str__(self):
        return '<PrimaryKinshipCode:' + self.kinship_code + '>'

    def __repr__(self):
        return '<PrimaryKinshipCode:' + self.kinship_code + '>'

    def get_seniority(self):
        if len(self.kinship_code) > 1:
            seniority = self.kinship_code[1:]
            if (seniority.isdigit()) or (seniority in {'+', '-'}):
                return seniority
            else:
                return ''
        else:
            return ''

    def get_generation(self):
        gen_diff: int = self.code2generation.get(self.kinshipType, None)
        return gen_diff

    def get_kin_distance(self):
        kin_dist: int = self.code2distance.get(self.kinshipType, None)
        return kin_dist

    def get_gender_pair(self):
        gender_pair = ['?', '?']
        if self.kinshipType == 'F':
            gender_pair[1] = 'm'
        elif self.kinshipType == 'M':
            gender_pair[1] = 'f'
        elif self.kinshipType == 'S':
            gender_pair[1] = 'm'
        elif self.kinshipType == 'D':
            gender_pair[1] = 'f'
        elif self.kinshipType == 'H':
            gender_pair[0] = 'f'
            gender_pair[1] = 'm'
        elif self.kinshipType == 'W':
            gender_pair[0] = 'm'
            gender_pair[1] = 'f'
        elif self.kinshipType == 'C':
            gender_pair[0] = 'm'
            gender_pair[1] = 'f'
        elif self.kinshipType == 'B':
            gender_pair[1] = 'm'
        elif self.kinshipType == 'Z':
            gender_pair[1] = 'f'
        else:
            pass
        return gender_pair

    def get_inverse_kinship(self, gender1: str):
        inverse_kinship = ''
        inverse_seniority = ''
        if self.kinshipType == 'H':
            inverse_kinship = 'W'
        if self.kinshipType == 'W':
            inverse_kinship = 'H'
        if self.kinshipType == 'C':
            inverse_kinship = 'H'

        if self.kinshipType == 'F' or self.kinshipType == 'M':
            if gender1 == 'm':
                inverse_kinship = 'S'
            elif gender1 == 'f':
                inverse_kinship = 'D'
            else:
                raise TypeError('Wrong inverse kinship!')

        if self.kinshipType == 'S' or self.kinshipType == 'D':
            if gender1 == 'm':
                inverse_kinship = 'F'
            elif gender1 == 'f':
                inverse_kinship = 'M'
            else:
                raise TypeError('Wrong inverse kinship!')

        if self.kinshipType == 'B' or self.kinshipType == 'Z':
            if gender1 == 'm':
                inverse_kinship = 'B'
            elif gender1 == 'f':
                inverse_kinship = 'Z'
            else:
                pass
                # raise TypeError('Wrong inverse kinship!')
        if self.seniority == '+':
            inverse_seniority = '-'
        if self.seniority == '-':
            inverse_seniority = '+'
        if self.seniority.isdigit():
            inverse_seniority = ''

        inverse_kin = PrimaryKinshipCode(inverse_kinship + inverse_seniority)
        return inverse_kin


###################################################################
#					class ComplexKinship
###################################################################


class ComplexKinshipCode:

    def __init__(self, kinship_code=None, primary_code_list=None):
        if primary_code_list is None:
            self.kinship_code = kinship_code
            self.primary_code_list: List[PrimaryKinshipCode] = self.decode_kinship_code()
            self.generation = self.get_generation()
            self.kin_distance = self.get_kin_distance()
            self.gender_pair = self.get_gender_pair()
        else:
            self.primary_code_list: List[PrimaryKinshipCode] = primary_code_list
            self.kinship_code = self.to_str()
            self.generation = self.get_generation()
            self.kin_distance = self.get_kin_distance()
            self.gender_pair = self.get_gender_pair()

    def __repr__(self):
        return '<ComplexKinshipCode:' + self.to_str() + '>'

    def to_str(self):
        return ''.join(code.to_str() for code in self.primary_code_list)

    #
    # @staticmethod
    # def primary_code_list_to_str(primary_code_list):
    #

    # complex_kinship_code_parsing
    def decode_kinship_code(self):
        # if string only contains alpha
        if self.kinship_code.isalpha() and self.kinship_code.isupper():
            element = list(self.kinship_code)
        else:  # string contains both alpha and number
            element = []
            head_loc = 0
            code_len = 1
            for i in range(1, len(self.kinship_code)):
                if self.kinship_code[i].isalpha():
                    element.append(self.kinship_code[head_loc:(head_loc + code_len)])
                    head_loc = i
                    code_len = 1
                else:
                    code_len = code_len + 1
            element.append(self.kinship_code[head_loc:(head_loc + code_len)])
        out = []  # in which each element is PrimaryKinshipCode object
        for code in element:
            out.append(PrimaryKinshipCode(code))
        return out

    def get_generation(self):
        out: int = 0
        for code in self.primary_code_list:
            gen_diff = code.generation
            if gen_diff is None:
                return None
            out += gen_diff
        return out

    def get_kin_distance(self):
        out: int = 0
        for code in self.primary_code_list:
            kin_dist = code.kin_distance
            if kin_dist is None:
                return None
            out += kin_dist
        return out

    def get_gender_pair(self):
        gender_pair = ['?', '?']
        gender_pair[0] = self.primary_code_list[0].gender_pair[0]
        gender_pair[1] = self.primary_code_list[-1].gender_pair[1]
        return gender_pair

    def get_gender_list(self, gender_0: str):
        assert gender_0 in {'m', 'f', '?'}
        size = len(self.primary_code_list)
        gender_list = ['?' for _ in range(size + 1)]
        gender_list[0] = gender_0
        for pos, code in enumerate(self.primary_code_list):
            gender_first, gender_second = code.gender_pair
            if gender_first == '?':
                pass
            elif gender_list[pos] == '?':
                gender_list[pos] = gender_first
            else:
                pass
                # assert gender_list[pos] == gender_first
                # return None
            if gender_second == '?':
                pass
            elif gender_list[pos + 1] == '?':
                gender_list[pos + 1] = gender_second
            else:
                pass
                # assert gender_list[pos + 1] == gender_second
                # return None
        return gender_list

    def get_inverse_kinship(self, gender_0: str) -> List[PrimaryKinshipCode]:
        inverse_kinship = []
        gender_list = self.get_gender_list(gender_0)
        rev_gender_list = list(reversed(gender_list))[1:]  # delete the last person's gender
        rev_code_list: List[PrimaryKinshipCode] = list(reversed(self.primary_code_list))
        for pos, gender in enumerate(rev_gender_list):
            # print(pos, gender)
            inverse_kinship.append(rev_code_list[pos].get_inverse_kinship(gender))
        return inverse_kinship


def joint(code1: PrimaryKinshipCode, code2: PrimaryKinshipCode):
    pass


def is_compatible(code1, code2):
    if type(code1) != ComplexKinshipCode:
        code1 = ComplexKinshipCode(code1)
    if type(code2) != ComplexKinshipCode:
        code2 = ComplexKinshipCode(code2)

    if code1.generation != code2.generation:
        return False
    # if code1.kin_distance != code2.kin_distance:
    #     return False
    if code1.gender_pair != code2.gender_pair:
        return False

    code1_path = code1.to_str()
    code2_path = code2.to_str()

    s = [i for i in code1_path if i.isalpha()]
    code1_path = "".join(s)

    s = [i for i in code2_path if i.isalpha()]
    code2_path = "".join(s)

    code1_path = indirect2direct(code1_path)
    code2_path = indirect2direct(code2_path)

    if code1_path == code2_path:
        return True
    else:
        return False


def indirect2direct(code: str):
    if 'B' in code:
        code = code.replace('B+', 'FS').replace('B-', 'FS').replace('B', 'FS')
    if 'Z' in code:
        code = code.replace('Z+', 'FD').replace('Z-', 'FD').replace('Z', 'FD')
    if 'M' in code:
        code = code.replace('M', 'FW').replace('M*', 'FW')
    return code

# a = PrimaryKinshipCode('B+')
# a.get_inverse_kinship('m')
#
# a = ComplexKinship('FB+D13S16')
# a.decode_kinship_code()
# a.get_gender_list('m')
# a.get_inverse_kinship('m')
# cc1 = 'B'
# cc2 = 'FS'
# is_compatible(cc1, cc2)
# cc1 = 'FBS'
# cc2 = 'FFS'
# is_compatible(cc1, cc2)

# def is_PrimaryKinshipCode_compatible(code1: PrimaryKinshipCode, code2: PrimaryKinshipCode):
#     if code1.generation != code2.generation:
#         return False
#     if code1.kin_distance != code2.kin_distance:
#         return False
#     if code1.gender_pair != code2.gender_pair:
#         return False
#
#     # if KinCode1 == KinCode2, it's simple
#     if code1.kinship_code == code2.kinship_code:
#         if code1.seniority == code1.seniority:
#             return True
#         elif code1.seniority == '' or code2.seniority == '':
#             return True
#         else:
#             return False
