"""读取已有文本指标或风险预测CSV出图，不重新训练。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_text_metrics(metrics,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    matrix=np.asarray(metrics["confusion_matrix"])
    axes[0].imshow(matrix,cmap="Blues")
    for (i,j),value in np.ndenumerate(matrix):axes[0].text(j,i,str(value),ha="center",va="center")
    axes[0].set(title="Confusion matrix",xlabel="Predicted class",ylabel="Actual class")
    history=metrics.get("history",[])
    if history:
        axes[1].plot([r["epoch"] for r in history],[r["train_loss"] for r in history],label="Training loss")
        axes[1].plot([r["epoch"] for r in history],[r["val_macro_f1"] for r in history],label="Validation macro F1")
        axes[1].legend()
    axes[1].set(xlabel="Epoch",title="Text model training")
    fig.savefig(output/"text_metrics.png",dpi=160);plt.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment",type=Path)
    args=parser.parse_args()
    metricpath=args.experiment/"metrics.json"
    metrics=json.loads(metricpath.read_text(encoding="utf-8"))
    if "confusion_matrix" in metrics:
        plot_text_metrics(metrics,args.experiment/"figures");return
    with (args.experiment/"predictions.csv").open(encoding="utf-8-sig",newline="") as handle:
        rows=[r for r in csv.DictReader(handle) if r["split"]=="test"]
    actual=np.array([float(r["actual_delta_covar"])*100 for r in rows])
    estimate=np.array([float(r["predicted_delta_covar"])*100 for r in rows])
    fig,ax=plt.subplots(figsize=(6,6),constrained_layout=True)
    ax.scatter(actual,estimate,s=5,alpha=.3)
    lo,hi=min(actual.min(),estimate.min()),max(actual.max(),estimate.max())
    ax.plot([lo,hi],[lo,hi],"--",color="grey")
    ax.set(xlabel="Actual (percentage points)",ylabel="Predicted (percentage points)",title="Test predictions")
    output=args.experiment/"figures";output.mkdir(exist_ok=True)
    fig.savefig(output/"test_predictions_replot.png",dpi=160);plt.close(fig)

if __name__=="__main__":main()
