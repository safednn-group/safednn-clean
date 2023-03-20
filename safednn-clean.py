"""
SafeDNN Clean: Find incorrect boxes in object detection datasets
Copyright (C) 2023  SafeDNN group

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.


Requirements:
    * numpy>=1.20.0 (interface)
    * sklearn>1.2.2 (clustering)
    * cleanlab>=2.2.0 (label quality scores)
    * tqdm>=4.40.0 (progress bars)

Usage:
    python3 safednn-clean.py --help
"""

import argparse
import collections
import itertools
import json
import sys

import cleanlab
import numpy as np


def bbox_iou(a, b):
    """
    Intersection over union of bboxes of two annotations
    """
    xa, ya, wa, ha = a["bbox"]
    xb, yb, wb, hb = b["bbox"]

    if (wa == 0 and wb == 0) or (ha == 0 and hb == 0):
        return 1

    w = min(xa + wa, xb + wb) - max(xa, xb)
    h = min(ya + ha, yb + hb) - max(ya, yb)
    if w < 0 or h < 0:
        return 0

    intersection = w * h
    union = wa * ha + wb * hb - intersection

    iou = intersection / union

    return iou


def cluster(annotations, iou):
    """
    Bounding box clustering with IoU threshold
    """
    # import here to emphasize insignificance of specific implementation
    from sklearn.cluster import AgglomerativeClustering

    annotations = list(annotations)
    if len(annotations) == 1:
        for annot in annotations:
            annot["cluster"] = 1
        return [annotations]

    dists = np.ndarray([len(annotations), len(annotations)])
    for ia, a in enumerate(annotations):
        dists[ia, ia] = 0
        for ib, b in enumerate(annotations[(ia + 1):]):
            dists[ia, ia + 1 + ib] = dists[ia + 1 + ib, ia] = 1 - bbox_iou(a, b)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="single",
        distance_threshold=(1 - iou),
    )
    clustering.fit(dists)

    for annot, cluster in zip(annotations, clustering.labels_, strict=True):
        annot["cluster"] = int(cluster)

    sorted_annots = sorted(annotations, key=lambda x: x["cluster"])

    return map(lambda x: x[1],
        itertools.groupby(sorted_annots, key=lambda x: x["cluster"]))


def reduce(annotations, iou):
    """
    Produce classification output from detection output
    """
    labels = []
    pred_probs = []
    clusters = []

    grouper = itertools.groupby(annotations, key=lambda x: x["image_id"])

    for img, annots in grouper:
        annots = list(annots)

        for clust in cluster(annots, iou):
            clust = list(clust)

            probs = {}
            for category, cat_annots in itertools.groupby(sorted(clust,
                    key=lambda x: x["category"]), key=lambda x: x["category"]):

                probs[category] = max(map(lambda x: x.get("score", 0),
                    cat_annots))

            if not sum(probs.values()):
                probs["background"] = 1

            cl_cats = list({x["category"] for x in clust if "score" not in x})
            labels.append(cl_cats if cl_cats else ["background"])
            pred_probs.append(probs)
            clusters.append(clust)

    cl_classes = list(set(itertools.chain.from_iterable(labels)))
    labels = [
        [cl_classes.index(xx) for xx in x]
        for x in labels
    ]
    pred_probs = [
        [x.get(category, 0) for category in cl_classes]
        for x in pred_probs
    ]

    return labels, pred_probs, clusters


def classify(clusters, scores, threshold, top_n):
    """
    Classify kinds of labeling errors
    """
    clust_counts = collections.Counter()
    annot_counts = collections.Counter()
    issues_list = []

    scores_clusters = list(zip(scores, clusters))
    scores_clusters.sort(key=lambda x: x[0])

    for score, clust in scores_clusters:
        # Please note that cluster score is the annotation quality score
        # but annotation["score"] is the detection softmax score.

        if score > threshold:
            continue

        annots = list(filter(lambda x: "score" not in x, clust))
        preds = list(filter(lambda x: "score" in x, clust))

        if annots and not preds:
            issue = "spurious"
            mark = annots
            annot_counts.update({issue: len(annots)})
            issues_list.append({"issue": "spurious", "annotations": annots})
        if preds and not annots:
            issue = "missing"
            mark = preds
            annot_counts.update({issue: 1})
            issues_list.append({"issue": "missing", "predictions": preds})
        if annots and preds:
            if len(set([x["category"] for x in annots + preds])) == 1:
                issue = "location"
                mark = annots
                annot_counts.update({issue: len(annots)})
                issues_list.append({
                    "issue": "location",
                    "annotations": annots,
                    "predictions": preds
                })
            else:  # different labels present
                issue = "label"
                mark = annots
                annot_counts.update({issue: len(annots)})
                issues_list.append({
                    "issue": "label",
                    "annotations": annots,
                    "predictions": preds
                })

        for annot in (mark[:top_n] if top_n is not None else mark):
            annot["issue"] = issue
            annot["quality"] = score
            if top_n is not None:
                top_n -= 1

        clust_counts.update({issue: 1})

    return clust_counts, annot_counts, issues_list


parser = argparse.ArgumentParser(
    description="Find incorrect boxes in object detection datasets")
parser.add_argument("annotations", type=argparse.FileType(),
        help="annotations in COCO format")
parser.add_argument("predictions", type=argparse.FileType(),
        help="predictions in COCO format with additional fields: "
        + "annotations[].score (softmax score), annotations[].category "
        + "(category name)")
parser.add_argument("--iou", default=0.5, type=float,
        help="IoU clustering threshold (min. overlap); default: 0.5")
parser.add_argument("--threshold", "-t", default=1.0, type=float,
        help="quality score cut-off, labels with greater quality will not be "
        + "outputted; default: 1.0 (no cut-off)")
parser.add_argument("--topn", "-n", type=int,
        help="output only top n worst quality labels")
parser.add_argument("--output", "-o", type=argparse.FileType("w"),
        default=sys.stdout, help="output in COCO format with additional fields: "
        + "annotations[].quality (annotation quality [0,1], lower number is "
        + "worse quality) and annotations[].issue (type of potential issue, can "
        + "be one of: spurious, missing, location, label); default: stdout")

if __name__ == "__main__":
    from tqdm import tqdm

    args = parser.parse_args()

    with tqdm.wrapattr(args.annotations, "read", miniters=1,
            desc="Loading annotations") as f:
        data = json.load(f)
        categories = {x["id"]: x["name"] for x in data["categories"]}
        category_ids = {x["name"]: x["id"] for x in data["categories"]}
        data["annotations"] = [
            {**x, "category": categories[x["category_id"]]}
            for x in data["annotations"]
        ]
    annot_cats = set(map(lambda x: x["category"], data["annotations"]))

    with tqdm.wrapattr(args.predictions, "read", miniters=1,
            desc="Loading predictions") as f:
        predictions = json.load(f)["annotations"]
    pred_cats = set(map(lambda x: x["category"], predictions))

    assert pred_cats.issubset(annot_cats), \
        "Cat. set of predictions is not a subset of cat. set of annotations"

    print("Sorting annotations...", file=sys.stderr)
    annotations = data["annotations"] + predictions
    annotations.sort(key=lambda x: x["image_id"])

    labels, pred_probs, clusters = reduce(
        tqdm(annotations, "Reducing annotations"), args.iou)

    assert labels and pred_probs

    print("Finding issues...", file=sys.stderr)
    scores = cleanlab.multilabel_classification.get_label_quality_scores(
        labels, np.array(pred_probs))

    clust_counts, annot_counts, issues_list = classify(clusters, scores,
        args.threshold, args.topn)

    # include "missing" annotations with negative ids
    pred_ids = itertools.count(-1, -1)
    data["annotations"] += [{**x, "id": next(pred_ids),
        "category_id": category_ids[x["category"]]}
        for x in predictions if "issue" in x]

    with tqdm.wrapattr(args.output, "write", miniters=1,
            desc="Outputting classified annotations...") as f:
        json.dump(data, f)

