import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(".")
from src.factory import *
from src.utils import *
from sklearn.metrics import log_loss

FOLD = 0
pickle_path = sys.argv[1]
DATADIR = Path("../input/rsna-str-pulmonary-embolism-detection/")

train = pd.read_csv(DATADIR / "train.csv")
pre = pd.read_csv(DATADIR / "split.csv")
train = train.merge(pre, on="StudyInstanceUID")

portion = pd.read_csv(DATADIR / "study_pos_portion.csv")
train = train.merge(portion, on="StudyInstanceUID")
t = train[train.fold == FOLD]
studies = t.StudyInstanceUID.unique()

agg = t.groupby("StudyInstanceUID")["SOPInstanceUID"].apply(list)
agg_one = t.groupby("StudyInstanceUID").first()
t = t.set_index("SOPInstanceUID")

def get_pred(_path):
    res = load_pickle(_path)
    raw_pred = pd.DataFrame({
        "sop": res["ids"],
        **res["outputs"],
    })
    return raw_pred.set_index("sop")

raw_pred = get_pred(pickle_path)
print(raw_pred.head())

def calib_p(arr, factor):  # set factor>1 to enhance positive prob
    return arr * factor / (arr * factor + (1-arr))
def post1(p_arr, q=90):
    return np.percentile(p_arr, q=q)

def search_best(raw_pred):
    best = np.inf
    best_f = -1
    calibrated_probs = None
    for factor in np.logspace(start=0, stop=np.log10(64), num=32, endpoint=True):
        sops = raw_pred.index
        LABELS = t.loc[sops].pe_present_on_image
        PREDS = raw_pred.pe_present_on_image
        PREDS = calib_p(PREDS, factor)
        WEIGHT = t.loc[sops].pe_present_portion
        loss = log_loss(LABELS, PREDS, sample_weight=WEIGHT)
        if loss < best:
            best = loss
            best_f = factor
            calibrated_probs = PREDS.copy()
    return best_f, best, calibrated_probs

print( "best factor for pe_present", search_best(raw_pred)[:2] ) 
print( "=========================== \n\n")

def search_best_pos(raw_pred):
    MEANS_IND = 0.020484822355039723

    for q in [97,97.5,98, 98.5,99,99.5]:
    #for q in np.arange(98.5, 99.3, 0.1):
    #for q in np.arange(90.0, 99.9, 1):
    # for q in [98.9]:

        LABELS = []
        PREDS = []
        LABELS_RIGHT = []
        PREDS_RIGHT = []
        PREDS_RIGHT2 = []
        LABELS_LEFT= []
        PREDS_LEFT = []
        PREDS_LEFT2 = []
        LABELS_CENT = []
        PREDS_CENT = []
        PREDS_CENT2 = []

        for study in t.StudyInstanceUID.unique():
            sops = agg.loc[study]
            label = agg_one.loc[study]
            label_is_pe = int((not label.indeterminate) and (not label.negative_exam_for_pe))
            LABELS.append(label_is_pe)

            LABELS_RIGHT.append(label.rightsided_pe)
            LABELS_LEFT.append(label.leftsided_pe)
            LABELS_CENT.append(label.central_pe)

            prediction =  raw_pred.loc[sops]  # preds for current study
            # pe pre
            probs_pe_present = prediction.pe_present_on_image
            pe_prob = post1(probs_pe_present, q=q)
            PREDS.append(pe_prob)

            ### rightsided
            ave_right = np.clip( np.sum( prediction.rightsided_pe ) / np.sum( prediction.pe_present_on_image ), 0, 1)
            ave_left  = np.clip( np.sum( prediction.leftsided_pe )  / np.sum( prediction.pe_present_on_image ), 0, 1)
            ave_cent  = np.clip( np.sum( prediction.central_pe )    / np.sum( prediction.pe_present_on_image ), 0, 1)
            # print(ave_right, label_is_right)
            PREDS_RIGHT.append( (1-MEANS_IND) * pe_prob *  ave_right)
            PREDS_LEFT.append( (1-MEANS_IND) * pe_prob *  ave_left)
            PREDS_CENT.append( (1-MEANS_IND) * pe_prob *  ave_cent)

            PREDS_RIGHT2.append( pe_prob * 0.849707702540123 + (1-pe_prob) * 0 )
            PREDS_LEFT2.append( pe_prob * 0.6957545475146886 + (1-pe_prob) * 0 )
            PREDS_CENT2.append( pe_prob * 0.1796915446534345 + (1-pe_prob) * 0 )

            # print(study, label_is_pe, pe_prob)

        print(f"q={q:.2f} logloss:{log_loss(LABELS, PREDS)}")

        #print("RIGHT")
        print(f"   R logloss:{log_loss(LABELS_RIGHT, PREDS_RIGHT)}")
        # print(f"    past logloss:{log_loss(LABELS_RIGHT, PREDS_RIGHT2)}")
        # print(f"    base logloss:{log_loss(LABELS_RIGHT, len(LABELS_RIGHT)*[np.mean(LABELS_RIGHT)] )}")

        #print("LEFT")
        print(f"   L logloss:{log_loss(LABELS_LEFT, PREDS_LEFT)}")
        # print(f"   logloss:{log_loss(LABELS_LEFT, PREDS_LEFT2)}")
        # print(f"   logloss:{log_loss(LABELS_LEFT, len(LABELS_LEFT)*[np.mean(LABELS_LEFT)] )}")

        #print("CENTRAL")
        print(f"   C logloss:{log_loss(LABELS_CENT, PREDS_CENT)}")
        # print(f"   logloss:{log_loss(LABELS_CENT, PREDS_CENT2)}")
        # print(f"   logloss:{log_loss(LABELS_CENT, len(LABELS_CENT)*[np.mean(LABELS_CENT)] )}")

print( "best factor for right/left/central", search_best_pos(raw_pred) ) 