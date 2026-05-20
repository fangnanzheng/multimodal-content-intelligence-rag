# Model Evaluation Report

## Classification Report

```text
                       precision    recall  f1-score   support

          counterfeit      0.917     0.868     0.892        38
       financial_scam      0.962     1.000     0.981        51
health_misinformation      0.905     1.000     0.950        19
                 safe      0.928     0.935     0.931       138
                 spam      0.902     0.852     0.876        54

             accuracy                          0.927       300
            macro avg      0.923     0.931     0.926       300
         weighted avg      0.926     0.927     0.926       300

```

## Confusion Matrix

Labels: `['counterfeit', 'financial_scam', 'health_misinformation', 'safe', 'spam']`

```text
[[ 33   0   0   5   0]
 [  0  51   0   0   0]
 [  0   0  19   0   0]
 [  1   2   1 129   5]
 [  2   0   1   5  46]]
```
