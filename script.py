import random
import subprocess
import os
import math
import csv


agExec = "python3 nn_ag.py"
plotExec = "python3 live_plot.py"
p = subprocess.Popen(agExec, shell=True,universal_newlines=True)
q = subprocess.Popen(plotExec, shell=True,universal_newlines=True)
p.wait()
q.wait()
                

   
