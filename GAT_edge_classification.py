import numpy as np
import pandas as pd
import os
from gnn_pg_solver import Training
from torch_geometric.loader import DataLoader
from pre_process_game_data import ModifyGameData
import torch.nn.functional as F 
from EdgeClassificationNetwork import ParityGameGATConv
import re
from modified_game_dataset import ModifiedGameDataset
import torch
import ast
import wandb 
from Wandb import configurations
import pg_parser
from sklearn.metrics import classification_report

class TrainPhase2(Training):
    
    
    def train(self, mod_games, solutions, config):

        data = ModifiedGameDataset(config['train_data_root'], mod_games, solutions, num_classes= config['classes'])
        train_loader = DataLoader(data, batch_size=5, shuffle=True)

        self._train_internal(train_loader, config)

        torch.save(self._network.state_dict(), self._output_stream)


    def _train_internal(self, loader, config):


        run = wandb.init(project= config['Experiment_Name'], entity='swathy-muthukrishnan-24', config=config)

        with run:

            wandb.watch(self._network)
            optimizer = torch.optim.Adam(self._network.parameters(), lr=config['learning_rate'])

            scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma= config['lr_decay'])
            scheduler2 = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['MultiStepLR_milestones'], gamma= config['MultiStepLR_decay'])
            criterion_nodes = torch.nn.CrossEntropyLoss()

            criterion_edges = torch.nn.CrossEntropyLoss(weight= torch.tensor(config['weights_for_loss']).cuda()) 


            for epoch in range(1, config['epochs']):

                self._network.train()

                running_loss = 0.
                i = 0

                for data in loader:  
                    i += 1
                    optimizer.zero_grad() 

                    out_nodes, out_edges = self._network(data.x.cuda(), data.edge_index.cuda(), data.edge_attr.cuda())    

                    
                    src, des = data.edge_index

                    scores = (data.x[src] * data.x[des]).sum(dim = -1)
                    loss = criterion_nodes(out_nodes, data.y_nodes) + criterion_edges(out_edges, data.y_edges.cuda())  
                    
                    
                    
                    loss.backward()  
                    optimizer.step()  

                    running_loss += loss.item()
                    if i % 50 == 0:
                        last_loss = running_loss / 50  
                        running_loss = 0.
                print('epoch: ', epoch, 'loss: ', running_loss)
                wandb.log({'epoch': epoch + 1, 'loss': running_loss})

                if config['static_lr'] == False: 

                    scheduler1.step()
                    if config['MultiStepLR'] == True: 

                        scheduler2.step()
                
            artifact = wandb.Artifact(config['artifact_model_name'], type='model')
            torch.save(self._network.state_dict(), os.path.join(wandb.run.dir, config['modelFileName']))
            artifact.add_file(os.path.join(wandb.run.dir, config['modelFileName']))
            run.log_artifact(artifact)
            run.finish()
            
def train(wandb_config, mod_game_root_dir = None, sol_root_dir = None, output_wt_file=None):


    if mod_game_root_dir == None: 

        mod_game_root_dir= os.path.join(os.getcwd(),'Dataset/train/Three_class_mod_games_train')
    if sol_root_dir== None:  

        sol_root_dir = os.path.join(os.getcwd(),'Dataset/train/sol_train')
    if output_wt_file == None: 
        output_wt_file = os.path.join(os.getcwd(),'GATConv_3_class_weights.pth')

    training = TrainPhase2()


    training.network = ParityGameGATConv(wandb_config['hidden_channels_nodes'], wandb_config['hidden_channels_edges'], wandb_config['message_passing_iterations'], config = wandb_config).cuda()

    training.output = output_wt_file
    
    mod_game_files = os.listdir(os.path.normpath(mod_game_root_dir))
    mod_game_files.sort()



    sol_files = os.listdir(os.path.normpath(sol_root_dir))
    sol_files.sort()


    mod_games = [pd.read_csv(os.path.join(os.path.normpath(mod_game_root_dir) , file))  for file in  mod_game_files]
    solutions = []
    
    for sol_file in sol_files: 
        with open(os.path.join(os.path.normpath(sol_root_dir) , sol_file)) as f: 
            solutions.append(f.readlines())


    training.train(mod_games, solutions, wandb_config)

class PredictPhase2():
    def __init__(self, wandb_config):
        self.wandb_config = wandb_config

    def predict(self, mod_game_files):
        self._network.load_state_dict(torch.load(self._weights), strict=True)
        self.network.eval()
        strategies = []
        strategies_1 = []
        file_names = []
        predictions = {}

        config = self.wandb_config

        run =  wandb.init(project= (config['Experiment_Name'] + '_' + 'Predict'), entity='swathy-muthukrishnan-24', config= config )

        with run:

            for mod_game_file in mod_game_files:
                if config['classes'] == 2: 

                    f, s = self._predict_single(input= pd.read_csv(mod_game_file), identifier= mod_game_file)
                    file_names.append(f)
                    strategies.append(s)
                elif config['classes'] == 3: 
                    f, s, s_1 = self._predict_single(input= pd.read_csv(mod_game_file), identifier= mod_game_file)
                    file_names.append(f)
                    strategies.append(s)
                    strategies_1.append(s_1)
            if config['classes'] == 2:
                predictions['file'] = file_names
                predictions['strategies'] = strategies
            elif config['classes'] == 3: 
                predictions['file'] = file_names
                predictions['strategies'] = strategies
                predictions['strategies_1'] = strategies_1
            

            artifact = wandb.Artifact(config['artifact_model_name'], type='model')
            pd.DataFrame(predictions).to_csv(os.path.join(wandb.run.dir, config['prediction_result']), header=None)
            artifact.add_file(os.path.join(wandb.run.dir, config['prediction_result']))
            run.log_artifact(artifact)
            run.finish()

        pd.DataFrame(predictions).to_csv('results_edge_classification.csv', header=None)

    def _predict_single(self, input=None, identifier=None, test_data= None):

        winning_edges_0 = []
        winning_edges_1 = []

        nodes, edges, node_attr = ModifyGameData.parse_mod_game_file(input, self.wandb_config['classes'])
        y_predictions = np.zeros(len(edges))
        edge_attr, global_attr = ModifyGameData.parse_mod_game_edge_attr(input, self.wandb_config['classes'])
        out_nodes, out_edges = self._network(torch.tensor(node_attr, dtype=torch.float),
                                        torch.tensor(edges, dtype=torch.long).t().contiguous(), torch.tensor(edge_attr, dtype=torch.float))

        if self.wandb_config['classes'] == 2: 
            y_predictions = [1 if out_edges[i][1].item()> out_edges[i][0].item() else 0 for i in range (0, len(out_edges))]

        
            winning_region_0 = [str(node) for node in range(len(out_nodes)) if out_nodes[node][0].item() > out_nodes[node][1].item()]

            winning_region_1 = [str(node) for node in range(len(out_nodes)) if out_nodes[node][0].item() < out_nodes[node][1].item()]
        

            winning_edges_0 = [(edges[i][0],edges[i][1]) for i in range (0, len(out_edges)) if out_edges[i][1].item()> out_edges[i][0].item()]
            
            winning_edges_1 = [(edges[i][0],edges[i][1]) for i in range (0, len(out_edges)) if out_edges[i][1].item() < out_edges[i][0].item()]
        
        elif self.wandb_config['classes'] == 3:
            
            y_predcitions = [torch.argmax(tensor) for tensor in out_edges]

            winning_edges_0 = [(edges[i][0], edges[i][1]) for i in range (0, len(out_edges)) if torch.argmax(out_edges[i]) == 1]
            winning_edges_1 = [(edges[i][0], edges[i][1]) for i in range (0, len(out_edges)) if torch.argmax(out_edges[i]) == 2]
            

        if identifier is not None: 
            file_name = os.path.basename(identifier)
        else: 
            file_name = "Unidentified"
    

        print("Predicting for....." + str(file_name))

        if self.wandb_config['classes'] == 2: 

            return (file_name, winning_edges_0)
        else: 
            return (file_name, winning_edges_0, winning_edges_1)

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value



def predict(wandb_config, weights =None, mod_game_file_root = None):

    if weights == None: 
        weights = os.path.join(os.getcwd(),'GATConv_3_class_weights.pth')
    if mod_game_file_root == None: 
        mod_game_file_root = os.path.join(os.getcwd(),'Dataset/test/Three_class_mod_games_test')

    mod_game_file_names = os.listdir(os.path.normpath(mod_game_file_root))
    mod_game_file_names.sort()

    mod_game_file_names = mod_game_file_names[0:wandb_config['test_data_size']]
    mod_game_files = [os.path.join(os.path.normpath(mod_game_file_root), file) for file in mod_game_file_names]
    
    predictor = PredictPhase2(wandb_config)

   
    predictor.network = ParityGameGATConv(wandb_config['hidden_channels_nodes'], wandb_config['hidden_channels_edges'], wandb_config['message_passing_iterations'], config = wandb_config)
    
    predictor.weights = weights

    
    predictor.predict(mod_game_files)

class Evaluator:

    def __init__(self, wandb_config):
        self._statistics = []
        self.histogram = False
        self.wandb_config = wandb_config

    def evaluate(self, original_game_file_root = None, prediction_results_file=None, ref_sol_root = None, output_file_location=None):

        if original_game_file_root == None: 
            original_game_file_root = os.path.join(os.getcwd(), 'Dataset/test/Three_class_mod_games_test')
        if prediction_results_file == None: 
            prediction_results_file = os.path.join(os.getcwd(),'results_edge_classification.csv')
        
        if output_file_location == None:
            output_file_location = os.path.join(os.getcwd())

        if ref_sol_root == None: 
            ref_sol_root = os.path.join(os.getcwd(), 'Dataset/test/sol_test')

        self._evaluate(original_game_file_root, prediction_results_file, ref_sol_root, output_file_location)
    


    def _evaluate(self, game_file_root, inputs, reference_root, output):

        prediction_results = pd.read_csv(os.path.normpath(inputs), header=None)
        game_file_names = prediction_results.iloc[:,1]
        pred_strategies_0 = prediction_results.iloc[:,2]
        pre_startegies_1 = []
        if self.wandb_config['classes'] == 3:
            pred_strategies_1 = prediction_results.iloc[:,-1]
        references = os.listdir(reference_root)
        references.sort()
        references[0:self.wandb_config['test_data_size']]
        config = self.wandb_config

        y_predictions = []
        y_ture_vales = []


        run = run = wandb.init(project= (config['Experiment_Name'] + '_' + 'Evaluate'), entity='swathy-muthukrishnan-24', config= config )

        scatter_table_0 = []

        histogram_0 = []

        scatter_table_1 = []

        histogram_1 = []


        with run:

            for i in range(0, self.wandb_config['test_data_size']):

                if self.wandb_config['classes'] == 2:
                    correct, wrong, new_strategies, y_prediction, y_true = self._evaluate_single(os.path.join(game_file_root,game_file_names[i]), os.path.join(reference_root, references[i]), pred_strategies_0[i], output)            
                    scatter_table_0.append([correct, wrong])
                    histogram_0.append([new_strategies])
                elif self.wandb_config['classes'] == 3:
                    pred_strategies = (pred_strategies_0[i], pred_strategies_1[i])
                    correct, wrong, new_strategies, y_prediction, y_true = self._evaluate_single(os.path.join(game_file_root, game_file_names[i]), os.path.join(reference_root, references[i]), pred_strategies, output)
                    scatter_table_0.append([correct[0], wrong[0]])
                    histogram_0.append([new_strategies[0]])
                    scatter_table_1.append([correct[1], wrong[1]])
                    histogram_1.append([new_strategies[1]])

                y_predictions = y_predictions + y_prediction
                y_ture_vales = y_ture_vales + y_true
                

            report = classification_report(y_ture_vales, y_predictions, zero_division="warn",output_dict=True)
            df = pd.DataFrame(report).transpose()

        
            if self.wandb_config['classes'] == 2:

                scatter_table_0 = wandb.Table(data= np.array(scatter_table_0), columns=["correct_strategies", "wrong_strategies"])
                table_0 = wandb.Table(data= np.array(histogram_0), columns=["new_strategies"])
                wandb.log({"correct_wrong_strategies_0" : wandb.plot.scatter(scatter_table_0, "correct_strategies", "wrong_strategies", title="Percentage of known correct and wrong PGsolver strategies for player 0")})
                wandb.log({'new_strategies_0': wandb.plot.histogram(table_0, "new_strategies", title="Percentage of new startegies calculated by GNN per game - Player 0")})
                wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None,y_true=y_ture_vales, preds=y_predictions,class_names= ["Not_a_win_startegy", "Win_strategy"])})
            elif self.wandb_config['classes'] == 3:
                scatter_table_0 = wandb.Table(data= np.array(scatter_table_0), columns=["correct_strategies", "wrong_strategies"])
                table_0 = wandb.Table(data= np.array(histogram_0), columns=["new_strategies"])
                wandb.log({"correct_wrong_strategies_0" : wandb.plot.scatter(scatter_table_0, "correct_strategies", "wrong_strategies", title="Percentage of known correct and wrong PGsolver strategies for player 0")})
                wandb.log({'new_strategies_0': wandb.plot.histogram(table_0, "new_strategies", title="Percentage of new startegies calculated by GNN per game - Player 0")})
                scatter_table_1 = wandb.Table(data= np.array(scatter_table_1), columns=["correct_strategies", "wrong_strategies"])
                table_1 = wandb.Table(data= np.array(histogram_1), columns=["new_strategies"])
                wandb.log({"correct_wrong_strategies_1" : wandb.plot.scatter(scatter_table_1, "correct_strategies", "wrong_strategies", title="Percentage of known correct and wrong PGsolver strategies for player 1")})
                wandb.log({'new_strategies_1': wandb.plot.histogram(table_1, "new_strategies", title="Percentage of new startegies calculated by GNN per game - Player 1")})
                wandb.log({"confusion_matrix" : wandb.plot.confusion_matrix(probs=None,y_true=y_ture_vales, preds=y_predictions,class_names= ["Not_a_win_startegy", "Player_0_win_strategy", "Player_1_win_strategy"])})
                
            
            artifact = wandb.Artifact(config['artifact_model_name'], type='model')
            df.to_csv(os.path.join(wandb.run.dir, (config['artifact_model_name'] + '_' + config['classification_report'])))

            artifact.add_file(os.path.join(wandb.run.dir, (config['artifact_model_name'] + '_' + config['classification_report'])))
            run.log_artifact(artifact)
            run.finish()
        

        print(report)

        if self.histogram:
            self._print_histogram(output)

    def _evaluate_single(self, game_file, reference, prediction, output):

        
        game = pd.read_csv(game_file)
        nodes, edges, _ = ModifyGameData.parse_mod_game_file(game, self.wandb_config['classes'])
        win_region_0, act_strategy_0, win_region_1, act_strategy_1 = pg_parser.get_solution(reference)
        

        
        if self.wandb_config['classes'] == 2: 
            if isinstance(prediction, str):
                pred_strategy_0= ast.literal_eval(prediction)
            
                _strategy_0 = np.array(pred_strategy_0, dtype=int)

            act_strategy_0 = [tuple(edge) for edge in act_strategy_0]


            y_true = torch.zeros(edges.shape[0], dtype=torch.long)
            wins_0 = self.count_incides(edges, act_strategy_0)
    
            if len(wins_0)>0:
                y_true[wins_0] = 1
            

            y_prediction = torch.zeros(edges.shape[0], dtype=torch.long)

            predict_wins_0 = self.count_incides(edges, _strategy_0)
            
            if len(predict_wins_0) > 0: 
                y_prediction [predict_wins_0] = 1
            

            correct_percentage_0, percentage_wrong_0, new_startegies_0 =  self.evaluate_correct_wrong(edges, win_region_0, pred_strategy_0, act_strategy_0)

            
            return correct_percentage_0, percentage_wrong_0, new_startegies_0, y_prediction.tolist(), y_true.tolist()
        
        elif self.wandb_config['classes'] == 3:

            prediction_0 , prediction_1 = prediction

            if isinstance(prediction_0, str):
                pred_strategy_0= ast.literal_eval(prediction_0)
            
                _strategy_0 = np.array(pred_strategy_0, dtype=int)

            if isinstance(prediction_1, str):
                pred_strategy_1 = ast.literal_eval(prediction_1)
            
                _strategy_1 = np.array(pred_strategy_1, dtype=int)

            act_strategy_0 = [tuple(edge) for edge in act_strategy_0]
            act_strategy_1 = [tuple(edge) for edge in act_strategy_1]
            
            y_true = torch.zeros(edges.shape[0], dtype=torch.long)
            wins_0 = self.count_incides(edges, act_strategy_0)
            if len(wins_0)>0: 
                y_true[wins_0] = 1
            
            wins_1 = self.count_incides(edges, act_strategy_1)
            if len(wins_1)>0: 
                y_true[wins_1] = 2

            y_prediction = torch.zeros(edges.shape[0], dtype=torch.long)

            predict_wins_0 = self.count_incides(edges, _strategy_0)

            if len(predict_wins_0) > 0: 
                y_prediction [predict_wins_0] = 1
            
            predict_wins_1 = self.count_incides(edges, _strategy_1)

            if len(predict_wins_1) > 0: 
                y_prediction [predict_wins_1] = 2

            correct_percentage_0, percentage_wrong_0, new_startegies_0 =  self.evaluate_correct_wrong(edges, win_region_0, pred_strategy_0, act_strategy_0)

            correct_percentage_1, percentage_wrong_1, new_startegies_1 =  self.evaluate_correct_wrong(edges, win_region_1, pred_strategy_1, act_strategy_1)

            correct_percentage = (correct_percentage_0, correct_percentage_1) 
            percentage_wrong = (percentage_wrong_0, percentage_wrong_1)
            new_startegies = (new_startegies_0, new_startegies_1)   

            return correct_percentage, percentage_wrong, new_startegies, y_prediction.tolist(), y_true.tolist()

    def evaluate_correct_wrong(self, edges, win_regions, pred_strategy, act_strategy): 

        correct_startegy, unknown_strategy = compare_regions(pred_strategy, act_strategy)
        count_all_strat = self.count_start_from_win_regions(edges,win_regions)


        num_edges = len(edges)
        num_known_startegies = len(act_strategy)


        num_wrong_startegies = num_known_startegies - correct_startegy
        if count_all_strat > 0: # There are possibilities(inclusive of strategies kown & Unknown by PG solver) for Player 0 or 1 to win
            new_startegies = unknown_strategy * 1.0/count_all_strat 
        else: 
            new_startegies = 0.0   # There are no possibilities for player 0 or 1 to win
        if len(act_strategy) > 0: # There is a definite PG solver strategy to win the game 
            correct_percentage = correct_startegy * 1.0 / len(act_strategy)
            if not correct_startegy == 0:
                percentage_wrong = num_wrong_startegies * 1.0 / len(act_strategy)
            else:
                percentage_wrong = 0.0
        else: # There is no PG solver strategy to win the game 
            if correct_startegy == 0:
                correct_percentage = 1
                percentage_wrong = 0.0
            else:
                correct_percentage = 0.0
                percentage_wrong = 1
        return correct_percentage, percentage_wrong, new_startegies

    def count_incides(self, edges, strategy):

        predict_wins = []

        for s in strategy:
            matches = np.where((edges == s).all(axis=1))[0]
            if len(matches) > 0:
                predict_wins.append(matches[0])
            else:
                predict_wins.append(-1)
                
        return predict_wins

    def count_start_from_win_regions(self, edges, win_regions):
        count = 0
        for edge in edges:
            if edge[0] in win_regions:
                count += 1
        return count


    def _print_histogram(self, output):
        max_unknown= max(map(lambda x: x[2], self._statistics))
        for num_unknown  in range(0, max_unknown  + 1):
            output.write(f"{num_unknown } {len(list(filter(lambda x: x[2] == num_unknown, self._statistics)))}\n")

    @property
    def output(self):
        return self._output

    @output.setter
    def output(self, value):
        self._output = value


    @property
    def histogram(self):
        return self._histogram

    @histogram.setter
    def histogram(self, value):
        self._histogram = value

def evaluate(wandb_config):
    evaluator = Evaluator(wandb_config)
    evaluator.evaluate()

def compare_regions(pred, act):
    correct, unknown = 0, 0

    
    for pred_edge in pred:
        if pred_edge in act:
            correct += 1
        else:
            unknown += 1

    return correct, unknown

def main(): 
    cuda_id = torch.cuda.current_device()
    device = torch.device(cuda_id)  
    config = configurations().return_config()


    #train(config)
    predict(config)
    evaluate(config)

if __name__ == '__main__': 
    main()
    


    

