from collections import defaultdict

import numpy as np
import pandas as pd
import torch

apply_softmax_dim_one = torch.nn.Softmax(dim=1)


def format_predictions(raw_predictions, raw_dataset, mapping=None):
    """
    Let's assume our model outputs the labels, 0 (neutral), 1 (positive), 2 (negative),
    but we want to use the labels 0 (neutral), 1 (emotional) instead.
    In this case we would want to say both label 1 (positive) and label 2 (negative)
    correspond to label 1 (emotional). The mapping parameters allows you to achieve this.

    Mapping example:
    {
        0: 0,
        1: 1,
        2: 1
    }
    """

    predictions_tensor = torch.tensor(raw_predictions[0])
    predictions_softmax = apply_softmax_dim_one(predictions_tensor).tolist()
    predictions = np.argmax(raw_predictions[0], axis=1)

    if mapping is not None:
        predictions = [mapping[x] for x in predictions]
        mapped_predictions_softmax = []
        for prediction in predictions_softmax:
            # this contains the added up probabilities/confidences for labels which should be merged
            mapped_softmax_dict = defaultdict(float)
            for label, probability in enumerate(prediction):
                mapped_label = mapping[label]
                mapped_softmax_dict[mapped_label] += probability

            # this sorts the entries in the dictionary by the key (the label/class),
            sorted_labels = sorted(mapped_softmax_dict.keys())
            # this outputs the value (the added probabilities/confidences),
            # which is mapped by the label into a list
            mapped_softmax = [mapped_softmax_dict[label] for label in sorted_labels]
            mapped_predictions_softmax.append(mapped_softmax)

        predictions_softmax = mapped_predictions_softmax

    predictions_confidence = np.max(predictions_softmax, axis=1)
    predictions = pd.DataFrame(predictions, columns=["label"])
    predictions_confidence = pd.DataFrame(predictions_confidence, columns=["confidence"])

    dataset = pd.read_csv(raw_dataset, header=0)

    final_dataset = pd.concat([dataset, predictions, predictions_confidence], axis=1)
    del final_dataset['Id']

    final_dataset = final_dataset.reset_index(drop=True)
    final_dataset.index.name = 'Id'

    return final_dataset

#The MIT License (MIT)
#Copyright (c) 2021, Team Orange

#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


