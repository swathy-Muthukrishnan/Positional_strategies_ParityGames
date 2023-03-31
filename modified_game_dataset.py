""" Data set Generation  for Edge classification 
    This script is used to generate the pytroch geometric instances for each game file 
"""

import argparse as ag_parse
import numpy as np 
import os
import pandas as pd
import pg_parser
from pre_process_game_data import ModifyGameData

import torch
from torch_geometric.data import InMemoryDataset, Data

 

class ModifiedGameDataset(InMemoryDataset):

    def __init__(self, root, mod_games, solutions, num_classes, transform=None, pre_transform=None, pre_filter=None):
        self._mod_games = mod_games 
        self._solutions = solutions
        self._num_of_classes = num_classes
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def processed_file_names(self):
        return['mod_data.pt']

    """
    ---------------make_graph----------------------------------------------------------------------------------------
    * This method is used to generte the pytorch geometric instances of each graoh with the corresponding node 
    * and edge attributes 
    * Ege attributes used in the method are generated in a similar way to the parse_mod_game_edge_attr method in the 
    * pre_process_gama_data.py script (calculated seperately due to file data dependency at run time)
    * Paramaters: (1) mod_game_df: data frame of the modified game file 
    *             (2) solution: solution data for each of the corrresponding game file 
    *             (3) num_of_classes: Number of classes 
    * Returns:    (1) data_snp: pytroch geometric instance of all game files as a data loader instance 
    * Note*:(1) Truth labels for a 2 class edge classification problem: 
                0 - The edge is not a  positional winning strategy for player 0 
                1 - The edge is a  positional winning strategy for player 0
            (2) Truth labels for a 3 class edge classification problem: 
                0 - The edge is not a  positional winning strategy for both player 0 and player 1  
                1 - The edge is a  positional winning strategy for player 0
                2 - The edge is a  positional winning strategy for player 1
    -----------------------------------------------------------------------------------------------------------------
    """
    def make_graph(self, mod_game_df, solution, num_of_classes):


        nodes, edges, node_attr = ModifyGameData.parse_mod_game_file(mod_game_df, num_of_classes)

        regions_0, strategy_0, regions_1, strategy_1 = pg_parser.parse_solution(solution) 
        
        y_nodes = torch.zeros(node_attr.shape[0], dtype=torch.long)
        y_nodes[regions_1] = 1  
        
        y_edges = torch.zeros(edges.shape[0], dtype=torch.long)

   
        index_0 = [np.where((edges == s).all(axis=1))[0][0] for s in strategy_0]

        index_1 = [np.where((edges == s).all(axis=1))[0][0] for s in strategy_1]


        y_edges[index_0] = 1

        if num_of_classes == 3: 

            y_edges[index_1] = 2

        edge_attributes = [] 

        win_regions = regions_0.tolist()
        win_regions_1 = regions_1.tolist()

        normalized_pri = torch.tensor(mod_game_df.iloc[:,1].astype(float) / np.max(mod_game_df.iloc[:,1].astype(float)))
        
        
        
        # Calculating global attributes 
        mean = torch.mean(normalized_pri)
        diffs = normalized_pri - mean
        var = torch.mean(torch.pow(diffs, 2.0))
        std = torch.pow(var, 0.5)
        zscores = diffs / std
        skews = torch.mean(torch.pow(zscores, 3.0))
        kurtoses = torch.mean(torch.pow(zscores, 4.0)) - 3.0 
        min = torch.min(normalized_pri)
        max = torch.max(normalized_pri)

        global_attributes = torch.tensor(np.array([mean, std, var, min, skews, kurtoses])).cuda()

        # Calculating the edge attributres

        for i in range(len(edges)):

            edge_attr = []
            priority = mod_game_df.iloc[edges[i][1],1].astype(float) / np.max(mod_game_df.iloc[:,1].astype(float))
        
            zscore = zscores[edges[i][1]] 
            owner = [1, 0] if mod_game_df.iloc[edges[i][1], 2] == 0 else [0, 1]

            winning_edge = [1, 0] if edges[i][1] in win_regions else [0, 1]
            color_of_priority = [1, 0] if mod_game_df.iloc[edges[i][1],1]%2 == 0 else [0,1]

            if num_of_classes == 2: 
                edge_attr = np.array([priority, owner[0], owner[1] , winning_edge[0], winning_edge[1], color_of_priority[0], color_of_priority[1], zscore, mean, std, var, min, skews, kurtoses ])
            elif num_of_classes == 3:
                winning_edge_1 = [1, 0] if edges[i][1] in win_regions_1 else [0, 1]
                edge_attr = np.array([priority, owner[0], owner[1] , winning_edge[0], winning_edge[1],  winning_edge_1[0],  winning_edge_1[1], color_of_priority[0], color_of_priority[1], zscore, mean, std, var, min, skews, kurtoses ])
            
            edge_attributes.append(edge_attr)

        edge_attributes = torch.tensor(np.array(edge_attributes), dtype=torch.float).cuda()
        
        data_snp = Data(x=torch.tensor(node_attr, dtype=torch.float).cuda(), edge_attr= edge_attributes, edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous().cuda(), y_nodes=y_nodes.cuda(), y_edges=y_edges.cuda(), u = global_attributes)
        return data_snp

    def process(self):
        # Read data into huge `Data` list.
        data_list = [self.make_graph(mod_game, solution, self._num_of_classes) for (mod_game, solution) in zip(self._mod_games, self._solutions)]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


"""
*Note: Execute without changing default values only if the directories match your folder structure or else pass the correct folder
*      and file locations 

"""
def main(): 
    
    mod_game_root_dir = os.path.join(os.getcwd(), 'games-small/mod_games_train') 
    sol_root_dir = os.path.join(os.getcwd(), 'games-small/sol_train') 
    mod_game_files = os.listdir(mod_game_root_dir)
    mod_game_files.sort()
    sol_files = os.listdir(sol_root_dir)
    sol_files.sort()
    mod_games = [pd.read_csv(os.path.join(os.path.normpath(mod_game_root_dir) , file))  for file in  mod_game_files ]
    solutions = []
    
    for sol_file in sol_files: 
        with open(os.path.join(os.path.normpath(sol_root_dir) , sol_file)) as f: 
            solutions.append(f.readlines())


    root = 'pg_data_2Class'

    data = ModifiedGameDataset(root, mod_games, solutions, 2)



if __name__ == "__main__":
    main()
        
            





