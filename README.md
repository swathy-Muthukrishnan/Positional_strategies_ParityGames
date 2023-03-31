# Learning positional strategies for Parity Games using Graph Neural Networks

Prior knowledge of the positional strategies for each node in a parity game helps the player to make more informed decisions about which nodes to control, thereby increasing their chances of winning. The winning strategy generated by a PG solver for a player is often a single strategy that is dependent on the algorithm the PGSolver is modelled after. Since there are different algorithms for solving parity games, the strategies given by each of these algorithms have a higher chance of differing from one another. Hence it is more beneficial to learn a set of positional strategies rather than a fixed set of winning strategies. This repository contains a GAT (GATConv + GAT) model that can learn and predict positional strategies for parity games.
The model takes in the graph representation of the parity games as input and is part of a two-stage training to generate positional strategies for Parity Games. In the first stage, the previously developed node classifier Model will be used to compute the winning regions of the Parity Games. In the second stage, the predicted winning areas from the 1st stage will be used to compute the positional strategies for each node in the Game.

# Requirements

## Packages: <br>

torch: 11.0+cu102', <br>
torch_geometric: 2.1.0, <br>
torch_scatter: 2.0.9, <br>
torch_sparse: 0.6.13, <br>
Wandb (For logging and model specification) <br>

# Dataset file structure

Recommended File structure to store game data and solution files <br>

GNNPG <br>
 |-games-small <br>
 | |- test <br>
 | | |-mod_games_test (Test data i.e. modified games for 2 class classification problems) <br>
 | | |-Three_class_mod_games_test (Test data i.e. modified games for 3 class classification problem) <br>
 | | |-sol_test (Solution files for test data) <br>
 | |- train <br>
 | | |-mod_games_train (Train data i.e. modified games for 2 class classification problems) <br>
 | | |-Three_class_mod_games_train (Train data i.e. modified games for 3 class classification problem) <br>
 | | |-sol_train (Solution files for train data) <br>
 |-pg_data_2Class <br>
 | |-mod_data.pt (Graph data set for 2 Class classification problem) <br>
 |-pg_data_3Class <br>
 | |-mod_data.pt (Graph data set for 3 Class classification problem) <br>
 |- &lt; Weight Files &gt; .pth <br>
 |– &lt; prediction results &gt;.csv <br>

Note: <br>
The games-small directory contains the set of unmodified orriginal game files and their solutions.


# Python Scripts

pg_parser.py - For parsing the unmodified game file and the solution file <br>
modified_game_dataset.py - For generating the graph data set <br>
pre_process_game_data.py - For creating the modified game files for 2 class and 3 class problems with the winning region information from the Node 
classifier <br>
GAT_edge_classification.py - Contains the Train, test and evaluation classes <br>
EdgeClassificationNetwork.py - Model architecture (GATConv + GAT) <br>
Wandb.py - For model hyperparameter configuration and Wandb logging specification <br>

## Two-class and Three class classification: <br>

(The Two class and three class specifications can be set through the Wandb.py script) <br>

The Two class classification model predicts the positional strategies for player 0 <br>
<br>
 Class Labels: <br>
  0 - The edge is not a winning positional strategy for player 0 <br>
  1 - The edge is a winning positional strategy for player 0 <br>

The three-class classification model predicts the positional strategies for both player 0 and player 1 <br>
 <br>
 Class Labels: <br>
  0 - The edge is not a winning positional strategy for player 0 and player 1 <br>
  1 - The edge is a winning positional strategy for player 0 <br>
  2 - The edge is a winning positional strategy for player 1 <br>

# Train Models

Train the model by executing the train function in the GAT_edge_classification.py script with the proper configurations and root folder locations.
<br>
Note: <br>
 (1) If the data set folder structure differs, pass the correct arguments. <br>
 (2) If there are not enough prediction results from the node classifier model (stage 1), use the pre_process_train_dataset function from the pre_process_train_dataset.py script to generate more training data.

# Predict Results

Predict the results by executing the predict function in the GAT_edge_classification.py script with the right configurations and root folder locations. 
<br>
Note: <br>
 (1) If the data set folder structure is different, then pass the correct arguments

# Evaluate

Evaluate the results by executing the predict function in the GAT_edge_classification.py script with the right configurations and root folder locations. 
<br>
Note: <br>
 (1) If the data set folder structure is different, then pass the correct arguments
