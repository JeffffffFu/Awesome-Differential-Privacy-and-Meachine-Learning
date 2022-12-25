import math
from datetime import time

import torch
from openpyxl import Workbook


def testtt():
    a1=torch.load("record/pathological_one_epoch/1.pth")
    a2=torch.load("record/pathological_one_epoch/5.pth")
    a3=torch.load("record/pathological_one_epoch/10.pth")
    a4=torch.load("record/pathological_one_epoch/20.pth")
    a5=torch.load("record/pathological_one_epoch/50.pth")
    print(a1)
    print(a2)
    print(a3)
    print(a4)
    print(a5)

testtt()