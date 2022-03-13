import json
import pandas as pd
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import textdistance
import time
from glob import glob
import os

from preprocess import Preprocess
from run import SEARCH_RECOMMEND

class EVALUATE(SEARCH_RECOMMEND):
    
    def __init__(self):
        super().__init__()
        
    def add_projectId(self, projectId):
        project_id = []
        project_id.append(projectId)
        return project_id
    
    