# BabyMamba ESP32 Metrics:

| Variant | Dataset | Status | Flash (B) | Scratch (B) | Heap Free Before (B) | Heap Used After (B) | Avg Latency (ms) | Parity vs PyTorch (%) | Predicted | Expected |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| ciBabyMambaHar | daphnet | ok | 220672 | 47240 | 226588 | 57988 | 491.156036 | 99.489288 | No Freeze | No Freeze |
| ciBabyMambaHar | motionsense | ok | 224320 | 71944 | 201952 | 57988 | 2651.248535 | 99.267189 | Downstairs | Downstairs |
| ciBabyMambaHar | opportunity | ok | 260512 | 71944 | 201952 | 57988 | 8420.959961 | 99.450012 | Null | Null |
| ciBabyMambaHar | pamap2 | ok | 230576 | 71944 | 201952 | 57988 | 1950.154053 | 99.564018 | Standing | Lying |
| ciBabyMambaHar | skoda | ok | 231616 | 59976 | 213920 | 57988 | 2344.808594 | 99.573967 | Gesture_5 | Gesture_5 |
| ciBabyMambaHar | ucihar | ok | 225856 | 71944 | 201952 | 57988 | 4221.790527 | 98.977974 | Standing | Standing |
| ciBabyMambaHar | unimib | ok | 221984 | 71944 | 201952 | 57988 | 385.829315 | 98.813705 | Activity_0 | Activity_0 |
| ciBabyMambaHar | wisdm | ok | 222768 | 71944 | 201952 | 57988 | 1679.188477 | 99.74939 | Jogging | Jogging |
| crossoverBiDirBabyMambaHar | daphnet | ok | 209632 | 20516 | 254460 | 41252 | 79.573006 | 99.920235 | No Freeze | No Freeze |
| crossoverBiDirBabyMambaHar | motionsense | ok | 211264 | 33892 | 241084 | 41252 | 154.563904 | 99.304688 | Standing | Downstairs |
| crossoverBiDirBabyMambaHar | opportunity | ok | 285616 | 33892 | 241084 | 41252 | 271.934509 | 99.162827 | Null | Null |
| crossoverBiDirBabyMambaHar | pamap2 | ok | 224928 | 33892 | 241084 | 41252 | 154.131607 | 99.539238 | Standing | Lying |
| crossoverBiDirBabyMambaHar | skoda | ok | 231536 | 27412 | 247564 | 41252 | 125.418709 | 97.202003 | Gesture_2 | Gesture_5 |
| crossoverBiDirBabyMambaHar | ucihar | ok | 214112 | 33892 | 241084 | 41252 | 147.573013 | 99.701012 | Standing | Standing |
| crossoverBiDirBabyMambaHar | unimib | ok | 208320 | 33892 | 241084 | 41252 | 150.672302 | 99.186699 | Activity_0 | Activity_0 |
| crossoverBiDirBabyMambaHar | wisdm | ok | 207936 | 33892 | 241084 | 41252 | 151.67131 | 99.598175 | Jogging | Jogging |
