# Major\_Exam\_MLOps

Objective
Build a complete MLOps pipeline focused on Linear Regression only, integrating training, testing,
quantization, Dockerization, and CI/CD — all managed within a single main branch.

1. Initiated the main branch with .gitignore, requirements.txt and README.md
2. created the src folder and written train.py file. After training the linearRegression model, I got the below results. And then saved the model using joblib.

python src/train.py
Loading California Housing dataset.
Creating LinearRegression model.
Training model.
R² Score: 0.5758
Mean Squared Error (Loss): 0.5559
Model saved to models/linear_regression_model.joblib

| R² Score | Loss   |
|----------|--------|
| 0.5758   | 0.5559 |

3. After that I performed testing using test_train.py file under the tests folder.

python tests/test_train.py

rootdir: /home/ritesh/Major_Exam_MLOps
collected 5 items                                                                                                                                           

tests/test_train.py::TestTraining::test_dataset_loading PASSED                                                                                        [ 20%]
tests/test_train.py::TestTraining::test_model_creation PASSED                                                                                         [ 40%] 
tests/test_train.py::TestTraining::test_model_training PASSED                                                                                         [ 60%]
tests/test_train.py::TestTraining::test_model_performance PASSED                                                                                      [ 80%]
tests/test_train.py::TestTraining::test_model_save_load PASSED                                                                                        [100%]

===================================================================== 5 passed in 0.28s =====================================================================


4. Then in the next step I did the manual quantization. I created the quantize.py file under src folder. I have applied quantization quality check and manual predictions on the 10 samples

python src/quantize.py

Loading trained model.

Original coefficients shape is : (8,)

Original intercept is : -37.023277706064

Original coef values is : [ 4.48674910e-01  9.72425752e-03 -1.23323343e-01  7.83144907e-01

 -2.02962058e-06 -3.52631849e-03 -4.19792487e-01 -4.33708065e-01]

 Quantizing intercept.

Intercept value: -37.02327771

Intercept scale factor: 5.40

Quantized parameters saved to models/quant_params.joblib

Max coefficient error: 0.00000002

Intercept error: 0.00000042

 Inference Test (the first 10 samples are..):

Original predictions (sklearn): [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725 2.01175367

 2.64550005 2.16875532 2.74074644 3.91561473]

Manual original predictions:    [0.71912284 1.76401657 2.70965883 2.83892593 2.60465725 2.01175367

 2.64550005 2.16875532 2.74074644 3.91561473]

Manual dequant predictions:     [0.71912454 1.76401826 2.70966059 2.83892763 2.60465899 2.01175536

 2.64550173 2.168757   2.7407482  3.91561644]

 Differences:

Sklearn vs manual original: [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

Original vs dequant manual:  [1.69724930e-06 1.69119024e-06 1.75485833e-06 1.70069523e-06

 1.73872417e-06 1.68434987e-06 1.68467863e-06 1.67666992e-06

 1.75952921e-06 1.71352492e-06]

Absolute differences: [1.69724930e-06 1.69119024e-06 1.75485833e-06 1.70069523e-06

 1.73872417e-06 1.68434987e-06 1.68467863e-06 1.67666992e-06

 1.75952921e-06 1.71352492e-06]

Max difference: 1.7595292121086459e-06

Mean difference: 1.710146983668892e-06

Quantization quality is good (max diff: 0.000002)