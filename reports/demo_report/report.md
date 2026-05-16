# Autonomous Driving Model Evaluation Report

_Generated 2026-05-16 01:17:02._

## Summary
- Precision: **0.897**
- Recall:    **0.788**
- F1:        **0.839**
- mAP@0.5:   **0.858**
- TP / FP / FN: 26 / 3 / 7
- Images: 10

## Per-class performance
| class         |   support |   tp |   fp |   fn |   precision |   recall |    f1 |
|:--------------|----------:|-----:|-----:|-----:|------------:|---------:|------:|
| bicycle       |         3 |    3 |    0 |    0 |       1     |    1     | 1     |
| car           |        14 |   13 |    1 |    1 |       0.929 |    0.929 | 0.929 |
| pedestrian    |        10 |    4 |    2 |    6 |       0.667 |    0.4   | 0.5   |
| traffic_light |         3 |    3 |    0 |    0 |       1     |    1     | 1     |
| truck         |         3 |    3 |    0 |    0 |       1     |    1     | 1     |

## Performance by condition
### lighting
| lighting   |   support |   tp |   fp |   fn |   precision |   recall |    f1 |
|:-----------|----------:|-----:|-----:|-----:|------------:|---------:|------:|
| day        |        24 |   21 |    2 |    3 |       0.913 |    0.875 | 0.894 |
| night      |         9 |    5 |    1 |    4 |       0.833 |    0.556 | 0.667 |

### weather
| weather   |   support |   tp |   fp |   fn |   precision |   recall |    f1 |
|:----------|----------:|-----:|-----:|-----:|------------:|---------:|------:|
| clear     |        24 |   22 |    3 |    2 |        0.88 |    0.917 | 0.898 |
| fog       |         3 |    1 |    0 |    2 |        1    |    0.333 | 0.5   |
| rain      |         6 |    3 |    0 |    3 |        1    |    0.5   | 0.667 |

### occlusion
| occlusion   |   support |   tp |   fp |   fn |   precision |   recall |    f1 |
|:------------|----------:|-----:|-----:|-----:|------------:|---------:|------:|
| high        |         6 |    3 |    0 |    3 |       1     |    0.5   | 0.667 |
| low         |        17 |   16 |    2 |    1 |       0.889 |    0.941 | 0.914 |
| medium      |        10 |    7 |    1 |    3 |       0.875 |    0.7   | 0.778 |

## Recommendations
- Class 'pedestrian' has F1=0.50 (support=10). Collect more training data and review annotation quality.
- Performance drops under weather='fog' (F1=0.50). Add more weather='fog' samples to the training set.
- 6 safety-critical objects were missed (pedestrians/cyclists). Prioritise these for relabeling and retraining.
- 3 confident false positives detected. Mine these as hard negatives for the next training round.