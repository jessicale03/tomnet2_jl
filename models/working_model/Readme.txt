--------------
WORKING MODEL:
--------------
(1) This is the working model built by Yun-Shiuan.
(2) The model treats data as trajectories rather than separate steps.
(3) The model includes both charnel and predent.
(3) The performance awesome. The model are trained separately for both simulation and human data.
	(1) For simulation data (S002a) with the model composed of both charnet and predent, the validation accuracy = 96.98%, testing accuracy = 97.48%
	(2) For simulation data (S002a) with the model composed of only charnet, the validation accuracy = 99.7%, testing accuracy = 99.7%
	(3) For human data (S030) with the model composed of both charnet and predent, the validation accuracy = ?%, testing accuracy = ?%
	(4) For human data (S030) with the model composed of only charnet, the validation accuracy = 99.9%, testing accuracy = 99.69%


----------
STRUCTURE:
-----------
├── working_model    
    ├── __pychache__                        # current working model
	├── main_imports 			            # folder for all extracted main_model functions 
        ├── __init__.py
        ├── _create_graphs.py
        ├── batch_generator.py
        ├── charnet.py
        ├── class_model.py
        ├── data_handler.py
        ├── model_parameter.py
        ├── nn_layers.py
        ├── optimization.py
        ├── prednet.py
        ├── preference_predictor_query.py
        ├── preference_preditor.py
        ├── tain_test_validate.py
	├── training_data_set_results           # training results on human (S030a) and simulated (S002a) data
        ├── test_on_human_data
        ├── test_on_simulation_data
    ├── main_model.py
    ├── Readme.txt


-------
USUAGE:
-------
Valid commandline arguments:
- "train" --> just train the model
    - returns training top error, validation top error, and validation loss
- "test" --> just test of the model
    - evaluates model based on training, validating, and test data
    - returns CSV file of validation and test performance
- "all" --> train and test the model 
    - trains and test consecutively after one another 

