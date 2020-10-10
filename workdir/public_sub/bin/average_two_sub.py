#!/usr/bin/env python3

import sys
import pandas as pd

f1 = sys.argv[1]
f2 = sys.argv[2]

sub1 = pd.read_csv(f1)
sub2 = pd.read_csv(f2)

sub1.set_index('id', inplace=True)
sub1.loc[sub2.id, 'label'] = (sub1.loc[sub2.id, 'label'] + sub2.label.values) / 2.0
sub1.reset_index(inplace=True)

sub1.to_csv("out.csv", index=False)
