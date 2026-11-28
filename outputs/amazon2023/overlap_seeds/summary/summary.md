# Limited-overlap results (test split)

Mean ± standard deviation over seeds. Lower is better.

| Overlap | Model | Seeds | Real users | MAE | RMSE |
|---|---|---|---|---|---|
| 1% | movie_mean | 3 | 890 | 0.7620 ± 0.0000 | 1.0507 ± 0.0000 |
| 1% | dupgt | 3 | 890 | 0.8146 ± 0.0132 | 1.1128 ± 0.0171 |
| 1% | semantics | 3 | 890 | 0.7954 ± 0.0165 | 1.0926 ± 0.0168 |
| 1% | semantics_llm_w0.01 | 3 | 890 | 0.7978 ± 0.0129 | 1.0954 ± 0.0142 |
| 1% | semantics_text_w0.01 | 3 | 890 | 0.7979 ± 0.0128 | 1.0955 ± 0.0142 |
| 1% | semantics_shuffled_w0.01 | 3 | 890 | 0.7978 ± 0.0129 | 1.0953 ± 0.0143 |
| 1% | semantics_llm_w0.1 | 3 | 890 | 0.7967 ± 0.0147 | 1.0924 ± 0.0166 |
| 1% | semantics_text_w0.1 | 3 | 890 | 0.7970 ± 0.0139 | 1.0934 ± 0.0161 |
| 1% | semantics_shuffled_w0.1 | 3 | 890 | 0.7963 ± 0.0150 | 1.0920 ± 0.0172 |
| 10% | movie_mean | 3 | 8,903 | 0.7620 ± 0.0000 | 1.0507 ± 0.0000 |
| 10% | dupgt | 3 | 8,903 | 0.7438 ± 0.0061 | 1.0301 ± 0.0012 |
| 10% | semantics | 3 | 8,903 | 0.7408 ± 0.0047 | 1.0258 ± 0.0009 |
| 10% | semantics_llm_w0.01 | 3 | 8,903 | 0.7417 ± 0.0072 | 1.0259 ± 0.0010 |
| 10% | semantics_text_w0.01 | 3 | 8,903 | 0.7416 ± 0.0072 | 1.0259 ± 0.0010 |
| 10% | semantics_shuffled_w0.01 | 3 | 8,903 | 0.7418 ± 0.0075 | 1.0259 ± 0.0010 |
| 100% | movie_mean | 1 | 89,031 | 0.7620 ± – | 1.0507 ± – |
| 100% | dupgt | 1 | 89,031 | 0.7237 ± – | 1.0119 ± – |
| 100% | semantics | 1 | 89,031 | 0.7199 ± – | 1.0075 ± – |
| 100% | semantics_llm_w0.01 | 1 | 89,031 | 0.7210 ± – | 1.0074 ± – |
| 100% | semantics_text_w0.01 | 1 | 89,031 | 0.7208 ± – | 1.0075 ± – |
| 100% | semantics_shuffled_w0.01 | 1 | 89,031 | 0.7208 ± – | 1.0074 ± – |

## Within-seed differences

ΔRMSE = model − reference on the same users and seed; negative means the model is better.

| Overlap | Model | Reference | Seeds | Mean ΔRMSE | SD | Mean ΔMAE | Model better in |
|---|---|---|---|---|---|---|---|
| 1% | semantics | dupgt | 3 | -0.02020 | 0.02734 | -0.01912 | 2/3 |
| 1% | semantics_llm_w0.01 | semantics | 3 | 0.00277 | 0.00417 | 0.00240 | 0/3 |
| 1% | semantics_text_w0.01 | semantics | 3 | 0.00288 | 0.00424 | 0.00244 | 0/3 |
| 1% | semantics_shuffled_w0.01 | semantics | 3 | 0.00274 | 0.00417 | 0.00237 | 0/3 |
| 1% | semantics_llm_w0.1 | semantics | 3 | -0.00023 | 0.00244 | 0.00128 | 2/3 |
| 1% | semantics_text_w0.1 | semantics | 3 | 0.00077 | 0.00262 | 0.00151 | 1/3 |
| 1% | semantics_shuffled_w0.1 | semantics | 3 | -0.00057 | 0.00292 | 0.00086 | 1/3 |
| 1% | semantics_llm_w0.01 | semantics_text_w0.01 | 3 | -0.00011 | 0.00008 | -0.00004 | 3/3 |
| 1% | semantics_llm_w0.01 | semantics_shuffled_w0.01 | 3 | 0.00003 | 0.00006 | 0.00003 | 1/3 |
| 1% | semantics_llm_w0.1 | semantics_text_w0.1 | 3 | -0.00100 | 0.00049 | -0.00022 | 3/3 |
| 1% | semantics_llm_w0.1 | semantics_shuffled_w0.1 | 3 | 0.00034 | 0.00074 | 0.00042 | 1/3 |
| 10% | semantics | dupgt | 3 | -0.00435 | 0.00046 | -0.00298 | 3/3 |
| 10% | semantics_llm_w0.01 | semantics | 3 | 0.00011 | 0.00019 | 0.00089 | 1/3 |
| 10% | semantics_text_w0.01 | semantics | 3 | 0.00015 | 0.00011 | 0.00083 | 0/3 |
| 10% | semantics_shuffled_w0.01 | semantics | 3 | 0.00013 | 0.00022 | 0.00101 | 1/3 |
| 10% | semantics_llm_w0.01 | semantics_text_w0.01 | 3 | -0.00004 | 0.00008 | 0.00007 | 2/3 |
| 10% | semantics_llm_w0.01 | semantics_shuffled_w0.01 | 3 | -0.00001 | 0.00004 | -0.00012 | 2/3 |
| 100% | semantics | dupgt | 1 | -0.00433 | – | -0.00377 | 1/1 |
| 100% | semantics_llm_w0.01 | semantics | 1 | -0.00010 | – | 0.00103 | 1/1 |
| 100% | semantics_text_w0.01 | semantics | 1 | -0.00007 | – | 0.00090 | 1/1 |
| 100% | semantics_shuffled_w0.01 | semantics | 1 | -0.00008 | – | 0.00091 | 1/1 |
| 100% | semantics_llm_w0.01 | semantics_text_w0.01 | 1 | -0.00003 | – | 0.00013 | 1/1 |
| 100% | semantics_llm_w0.01 | semantics_shuffled_w0.01 | 1 | -0.00002 | – | 0.00012 | 1/1 |
