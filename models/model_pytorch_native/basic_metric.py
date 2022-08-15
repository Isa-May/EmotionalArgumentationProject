from sklearn.metrics import f1_score, precision_score, recall_score


def basic_metric_for_classes(classes, predictions, golds):
    f1_micro = f1_score(golds, predictions, average='micro')
    f1_macro = f1_score(golds, predictions, average='macro')
    f1_weighted = f1_score(golds, predictions, average='weighted')

    precision_micro = precision_score(golds, predictions, average='micro')
    precision_macro = precision_score(golds, predictions, average='macro')
    precision_weighted = precision_score(golds, predictions, average='weighted')

    recall_micro = recall_score(golds, predictions, average='micro')
    recall_macro = recall_score(golds, predictions, average='macro')
    recall_weighted = recall_score(golds, predictions, average='weighted')

    metrics = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision_micro': precision_micro,
        'precision_macro': precision_macro,
        'precision_weighted': precision_weighted,
        'recall_micro': recall_micro,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted
    }

    f1_score_class = f1_score(golds, predictions, average=None)
    for name, score in zip(classes, f1_score_class):
        metrics["f1_" + name] = score

    precision_score_class = precision_score(golds, predictions, average=None)
    for name, score in zip(classes, precision_score_class):
        metrics["precision_" + name] = score

    recall_score_class = recall_score(golds, predictions, average=None)
    for name, score in zip(classes, recall_score_class):
        metrics["recall_" + name] = score

    return metrics

#The MIT License (MIT)
#Copyright (c) 2021, Team Orange

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


