#!/usr/bin/env python
import pandas as pd
import numpy as np
import string
import logging
import nltk
import re


data = pd.read_csv('Data/cleaned_data.csv')
print(data.head)