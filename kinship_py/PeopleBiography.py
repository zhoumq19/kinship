#
# from typing import Set, List, Dict
# # ///////////////////////////////////////////////////////////////////
# #					 class PeopleBiography
# # ///////////////////////////////////////////////////////////////////
# """
# Description:
#     container for one personal profile
# Data elements:
#     [D1] PeopleID: a unique ID for the profile, non-negative IDs for HFs formally recorded by CBDB, negative IDs for HF candidates
#     [D2] FamilyName, GivenName, AltNames: names of the person
#     [D3] gender: 'm' for male, 'f' for female, '?' for unknown
#     [D4] DateOfBirth,DateOfDeath:
#     [D5] ErrorCode: map<ErrorType,ErrorCode>, error report for the person generated in the analysis procedure
#     [D6] status: internal status for the profile, takes values in {0,1}; take 1 if the person profile is active, 0 if it is redundant
# Functions:
#     [F1]
# """
#
#
# class PeopleBiography:
#     __slots__ = ['PID']
#     def __init__(self, PID: int, status: int = 1):
#         self.PID = PID  # person ID
#         self.attr2value: Dict = dict()  # dict of features:  family_name, given_name, birth/death, gender
#         self.status = status  # '1' fir active, '0' for redundant
#         self.error_code: Set[str] = set()  # error report for the person generated in the analysis procedure
#         self.alt_names = set()  #
#
#     def is_active(self):
#         return self.status == 1
#
#     def get_age(self):
#         if self.attr2value['birth'] is not None or self.attr2value['death'] is not None:
#             return self.attr2value['death'] - self.attr2value['birth']
#         else:
#             return None
#
#     def add_alt_name(self, names: str) -> Set[str]:
#         return self.alt_names.add(names)
