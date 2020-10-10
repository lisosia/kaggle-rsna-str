#!/usr/bin/env python3

import sys
import pandas as pd
import numpy as np

def prob_to_logit(prob):
    return np.log(prob/(1-prob))
def logit_to_prob(logit):
    return 1 / (1 + np.exp(- logit))
def logit_average(prob1, prob2):
    return logit_to_prob(
        (prob_to_logit(prob1) + prob_to_logit(prob2) ) / 2.0
    )
assert 0.35 == logit_to_prob(prob_to_logit(0.35))


f1 = sys.argv[1]
f2 = sys.argv[2]

sub1 = pd.read_csv(f1)
sub2 = pd.read_csv(f2)

sub1.set_index('id', inplace=True)
sub1.loc[sub2.id, 'label'] = logit_average(sub1.loc[sub2.id, 'label'], sub2.label.values)
sub1.reset_index(inplace=True)

sub1.to_csv("out.csv", index=False)
