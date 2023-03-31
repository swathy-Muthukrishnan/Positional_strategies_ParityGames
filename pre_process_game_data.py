""" Data set Modifier  for Edge classification 

    This script modifies the game_xxxx.txt 

"""

import argparse as ag_parse
import numpy as np 
import os
import pandas as pd
import pg_parser

class ModifyGameData:

    def read_dataset( input_file_path = None):

       
        if input_file_path == None:
                        
            input_file_path = os.path.join(os.getcwd(),'Phase2_train_data.csv')
        try: 
            return (pd.read_csv(input_file_path, header= None).to_numpy())
        
                
        except Exception as e:
            
            print("Error occured while reading prediction results: " +  e)
            return None

    """
    -----------------modify_game_file-----------------------------------------------------------------------------
    * This method creates a .csv game file from the original unmodified .txt game file
    * The methode will only create the changes needed for classifying Player 0 strategies (class 2 modified game files)
    * The columns in the new csv game file are: "Node_ID",  "priority", "owner", "successors", "name", "winning node"
    * The winning node column contains entries marked with either a '1' or '0'
    *       '1' means the corresponding node is in the winning region for player 0 
    *       '0' means the corresponding node is in the winning region for player 0 
    * Parameter: (1) results : List containing the full paths of  original unmodified game file 
    *                *Note input .txt the file should contain delimitiers
    *            (2) output_file_location : location of traget directory
    *                *Note: change default value if needed or pass new entry
    * Return:    NA
    ----------------------------------------------------------------------------------------------------------------
    """     
        
    def modify_game_file(results, output_file_location = None):


        if output_file_location == None:
        
            output_file_location = os.getcwd()

        for result in results: 
            
            try: 

                game_data = pd.read_csv(result[0].strip(), sep=" ", header=None, skiprows=1)
                game_data.columns = ["Node_ID",  "priority", "owner", "successors", "name"]
                game_data["winning node"] = 0
                new_file_name = ("mod_" + os.path.basename(os.path.normpath(result[0].strip()))).replace('.txt', '.csv')
                new_file_location = os.path.join(os.path.normpath(output_file_location), new_file_name)
                game_data.set_index('winning node')
                
                if not pd.isna(result[1]):

                    game_data.iloc[np.array(result[1].split(" ")).astype(int), game_data.columns.get_loc('winning node')] = 1

                game_data.to_csv(new_file_location, mode = 'w', index=False)

            except Exception as e: 
                print("Error occurced while processing the file: " + result[0] + e)

    """
    -----------------pre_process_predict_results---------------------------------------------------------------------
    * This method is a helper method for representing the prediction results from node classifer in a .csv file format
    * The method cretes a .csv containg the winning regions for player 0 
    * The columns in the new csv game file are: "Game files", "Winning regions"
    * Parameters: (1) predict_results_file: Prediction or results from Node classifier model
    * Returns:    (1) The file path of the new prediction results in a .csv format
    -----------------------------------------------------------------------------------------------------------------
    """

    def pre_process_predict_results( predict_results_file): 

        results_df = pd.read_csv(os.path.normpath(predict_results_file), header = None)
        winning_regions = []
        file_paths = []
        mod_results = {}
        for i  in range(0, len(results_df)):
            line = results_df.iloc[i, -1]
            file_path, winning_region = line.split(" ", 1)
            file_paths.append(file_path)
            winning_regions.append(winning_region)

        mod_results['Game files'] = file_paths
        mod_results['Winning regions'] = winning_regions

        mod_results_df = pd.DataFrame(mod_results)
        # print(mod_results_df)
        mod_results_df.to_csv(os.path.join(os.path.dirname((os.path.normpath(predict_results_file))), "mod_results.csv"), index= False, header= False)
        return os.path.join(os.path.dirname((os.path.normpath(predict_results_file))), "mod_results.csv")

    """
    -----------------modify_game_for_3_class---------------------------------------------------------------------
    * This method modifies the class 2 modified game files to create the modified game files for the 3 class 
    * classification Problem
    * The new colum introduced to the inout class 2 gmodified game file is "winning_regions_1"
    * The winning_regions_1 column contains entries marked with either a '1' or '0'
    *       '1' means the corresponding node is in the winning region for player 1 
    *       '0' means the corresponding node is in the winning region for player 1
    * Parameters (1) mod_games_root : root directory of the class 2 modified game data     
    *            (2) sol_root : root dirrectory of the solution files for the corresponding modified class 2 game 
    *                           files
    -------------------------------------------------------------------------------------------------------------
    """

    def modify_game_for_3_class(mod_games_root=None, sol_root=None):

        if mod_games_root == None:
            mod_games_root = os.path.join(os.getcwd(), "Dataset/train/mod_games_train")
        if sol_root == None: 
            sol_root = os.path.join(os.getcwd(), "Dataset/train/sol_train")

        game_file_names = os.listdir(os.path.normpath(mod_games_root))
        game_file_names.sort()
        
        sol_file_names = os.listdir(os.path.normpath(sol_root))
        sol_file_names.sort()
        game_files = []
        for game_file in game_file_names: 
            game_files.append(os.path.join(mod_games_root, game_file ) )

        winning_regions_1 = []
        for sol_file in sol_file_names: 
            with open(os.path.join(os.path.normpath(sol_root) , sol_file)) as f: 
                regions_0, strategy_0, regions_1, strategy_1 = pg_parser.parse_solution(f.readlines())
                winning_regions_1.append((regions_1.astype(int).tolist()))
        mod_game_3_class = []

        try: 

            new_dir = os.mkdir(os.path.join( (os.path.join(os.getcwd(), 'Dataset/train')), "Three_class_mod_games_train"))
        
        except: 
            new_dir = os.path.join( (os.path.join(os.getcwd(), 'Dataset/train')), "Three_class_mod_games_train")

        for i in range(0, len(game_files)):
            
            mod_game_df = pd.read_csv(game_files[i])
            old_no_col = len(mod_game_df.columns) 
            win_1 = np.zeros(len(mod_game_df))
            win_region = np.array(winning_regions_1[i])
            if len(win_region) > 0:

                win_1[win_region] = 1
             
            file_name = os.path.basename(game_files[i])
            
            mod_game_df['winning_regions_1'] = win_1.astype(int).tolist()
            if len(mod_game_df.columns) != (old_no_col + 1):

                mod_game_df_2 = mod_game_df.drop(df.iloc[:, 0],axis = 1)

            mod_game_df.to_csv( os.path.join(new_dir,  ('three_class' + file_name)), index= False)
            



    """
    --------------------pre_process_train_dataset-------------------------------------------------------------------
    * This method should only be used when there is not enough predictions results from the node classifier model 
    * The method reads the unmodified game files and their solutions to generate a prediction result .csv file 
    * similar to the 
    * output produced by the node classifier model 
    * The columns in the new csv game file are: "Game files", "Winning regions"
    * The winning regions column represents the winning nodes in player 0's winning region
    ----------------------------------------------------------------------------------------------------------------
    """

    def pre_process_train_dataset( game_files_dir, sol_files_dir, mod_data_location):
        

        
        game_files = []
        winning_regions = []
        train = {}

        game_file_names = os.listdir(os.path.normpath(game_files_dir))
        game_file_names.sort()
        
        sol_file_names = os.listdir(os.path.normpath(sol_files_dir))
        sol_file_names.sort()

        for game_file in game_file_names: 
            game_files.append(os.path.join(os.path.normpath(game_files_dir), game_file ) )


        for sol_file in sol_file_names: 
            with open(os.path.join(os.path.normpath(sol_files_dir) , sol_file)) as f: 
                regions_0, strategy_0, regions_1, strategy_1 = pg_parser.parse_solution(f.readlines())
                winning_regions.append(" ".join(regions_0.astype(str).tolist()))
        
        train['Game Files'] = game_files
        train['winnign regions'] = winning_regions


        pd.DataFrame(train).to_csv(os.path.join(os.path.normpath(mod_data_location), "Phase2_train_data.csv"), index= False, header= False)
        return os.path.join(os.path.normpath(mod_data_location), "Phase2_train_data.csv")
    
    """
    ------------------parse_mod_game_edge_attr---------------------------------------------------------------------------------------------------------
    * This method generates the edge attributes for each game file 
    * Edge attributes
    * For 2 class classification problem a total of 10 edge features are generated and they are represented as a 1D tensor of 14 elements
    *       The first 7 elements of these fetaures are non statistical
    *       Non statistical features: priority, owner[0], owner[1] , winning_edge_0[0], winning_edge_0[1], color_of_priority[0], color_of_priority[1]
    *    
    * For 3 class classification problem a total of 11 edge features are generated and they are represented in a 1D tensor of 16 elements
    *       The first 7 elements of these fetaures are non statistical
    *       Non statistical features: priority, owner[0], owner[1] , winning_edge_0[0], winning_edge_0[1], color_of_priority[0], color_of_priority[1]    
    * Statistical features for both classes: zscore, mean, std, var, min, kurtoses, skews
    * Parameters : (1) mod_game_df: Pandas data frame of the modified game files
    *              (2) num_of_classes: Number of classes 
    * Note: The global attributes are nothing but the statistical features (Use only for future experiments with Meta Layer)
    ----------------------------------------------------------------------------------------------------------------------------------------------------
    """


    def parse_mod_game_edge_attr(mod_game_df, num_of_classes):

        nodes = mod_game_df.iloc[:,0].to_numpy()
        edges = np.concatenate([(np.array(list(np.broadcast([mod_game_df.iloc[i, 0]], mod_game_df.iloc[i, 3].split(',')))).astype(int)) for i in range(len(mod_game_df))])


        win_regions = [mod_game_df.iloc[i,0]  for i  in range(len(mod_game_df)) if mod_game_df.iloc[i, 5] == 1 ] # One hot encoding for winning regions of 0
        win_regions_1 = []
        if num_of_classes == 3: 
            win_regions_1 = [mod_game_df.iloc[i,0]  for i  in range(len(mod_game_df)) if mod_game_df.iloc[i, 6] == 1 ]

        edge_attributes = [] 

        normalized_pri = np.array(mod_game_df.iloc[:,1].astype(float) / np.max(mod_game_df.iloc[:,1].astype(float)))
        
        mean = np.mean(normalized_pri) # mean 
        diffs = normalized_pri - mean  
        var = np.mean(np.power(diffs, 2.0)) # Variance
        std = np.power(var, 0.5) # standard deviation
        zscores = diffs / std # Zscores
        skews = np.mean(np.power(zscores, 3.0)) # skews
        kurtoses = np.mean(np.power(zscores, 4.0)) - 3.0  # Kurtoses 
        min = normalized_pri.min() # Minimum of the normalized priority values 
        max = normalized_pri.max() # Maximum o the normalized priority vales

        global_attributes = np.array([mean, std, var, min, kurtoses, skews])

        for i in range(len(edges)):
            edge_attr = []
            priority = mod_game_df.iloc[edges[i][1],1].astype(float) / np.max(mod_game_df.iloc[:,1].astype(float))
            owner = [1, 0] if mod_game_df.iloc[edges[i][1], 2] == 0 else [0, 1]
            winning_edge_0 = [1, 0] if edges[i][1] in win_regions else [0, 1]
            winning_edge_1 = []
            zscore = zscores[edges[i][1]]
            color_of_priority = [1, 0] if mod_game_df.iloc[edges[i][1],1]%2 == 0 else [0,1]
            if num_of_classes == 2: 
                edge_attr = np.array([priority, owner[0], owner[1] , winning_edge_0[0], winning_edge_0[1], color_of_priority[0], color_of_priority[1], zscore, mean, std, var, min, kurtoses, skews])
            elif num_of_classes == 3: 

                winning_edge_1 = [1, 0] if edges[i][1] in win_regions_1 else [0, 1]
                edge_attr = np.array([priority, owner[0], owner[1] , winning_edge_0[0], winning_edge_0[1], winning_edge_1[0], winning_edge_1[1], color_of_priority[0], color_of_priority[1], zscore, mean, std, var, min, kurtoses, skews])

            edge_attributes.append(edge_attr)


        return edge_attributes, global_attributes
    """
    -------------------------parse_mod_game_file--------------------------------------------------------------------------------------------------------------
    * This method generates the node attributes for each game file 
    * Node attributes
    * For 2 class classification problem a total of 10 edge features are generated and they are represented as a 1D tensor of 14 elements
    *       The first 7 elements of these fetaures are non statistical
    *       Non statistical features: priority, owner[0], owner[1] , in_winning_region_0[0],in_winning_region_0[1], color_of_priority[0], color_of_priority[1]
    *       
    * For 3 class classification problem a total of 11 edge features are generated and they are represented in a 1D tensor of 16 elements
    *       The first 7 elements of these fetaures are non statistical
    *       Non statistical features: priority, owner[0], owner[1] ,in_winning_region_0[0], in_winning_region_0[1], in_winning_region_1[0], 
    *                                 in_winning_region_1[1], color_of_priority[0], color_of_priority[1]    
    * Statistical features for both classes: zscore, mean, std, var, min, kurtoses, skews
    * Parameters : (1) mod_game_df: Pandas data frame of the modified game files
    *              (2) num_of_classes: Number of classes
    * Returens : (1) tuple of Nodes, edges and node attributes of each game file
    * Note: Use this function in the prediction phase
    * Note: The global attributes are nothing but the statistical features (Use only for future experiments with Meta Layer)
    * Note: ower of the node, winning region flags, and color (i.e parity of the priority which coule be even or odd) are categorically encoded
    * 
    -----------------------------------------------------------------------------------------------------------------------------------------------------------
    """


    def parse_mod_game_file(mod_game_df, num_classes):
                                
        nodes = mod_game_df.iloc[:,0].to_numpy()
        edges = np.concatenate([(np.array(list(np.broadcast([mod_game_df.iloc[i, 0]], mod_game_df.iloc[i, 3].split(',')))).astype(int)) for i in range(len(mod_game_df))])
        
        normalized_pri = np.array(mod_game_df.iloc[:,1].astype(float) / np.max(mod_game_df.iloc[:,1].astype(float))) # Normalized prioirty
        Ones_tensor = np.ones_like(normalized_pri) 
        mean = np.mean(normalized_pri) # Mean 
        diffs = normalized_pri - mean 
        var = np.mean(np.power(diffs, 2.0)) # Variance
        std = np.power(var, 0.5) # Standard deviation
        zscores = diffs / std # Zscore 
        skews = np.mean(np.power(zscores, 3.0)) # Skews
        kurtoses = np.mean(np.power(zscores, 4.0)) - 3.0  # Kurtoses
        min = normalized_pri.min() # Minimum normalized priority
        max = normalized_pri.max() # Maximum Normalized prioirty 

        mean_tensor = mean*Ones_tensor
        var_tensor = var*Ones_tensor
        std_tensor = std*Ones_tensor
        min_tensor = min*Ones_tensor
        max_tensor = max*Ones_tensor
        kur_tensor = kurtoses*Ones_tensor
        skews_tensor = kurtoses*Ones_tensor
        node_attr_train_1 = np.append(

            np.expand_dims(mod_game_df.iloc[:,1].astype(float) / np.max(mod_game_df.iloc[:,1].astype(float)), axis=1), # Normalizing the range of colours / priorities 
            #[[1, 0] if mod_game_df.iloc[i, 2] == '0' else [0, 1] for i  in range(len(mod_game_df))], 
            [[1, 0] if mod_game_df.iloc[i, 2] == 0 else [0, 1] for i  in range(len(mod_game_df))],
             axis=1 # Categorical encoding
        )

        node_attr_train_1 = np.concatenate((node_attr_train_1, [[1, 0] if mod_game_df.iloc[i, 5] == 0 else [0, 1] for i  in range(len(mod_game_df))]), axis = 1)

        if num_classes == 3: 
            node_attr_train_1 = np.concatenate((node_attr_train_1, [[1, 0] if mod_game_df.iloc[i, 6] == 0 else [0, 1] for i  in range(len(mod_game_df))]), axis = 1)

        node_attr_train_1 = np.concatenate((node_attr_train_1, [[1, 0] if mod_game_df.iloc[i, 1]%2 == 0 else [0, 1] for i  in range(len(mod_game_df))]), axis = 1)

        #node_attr_train_1 = np.concatenate((node_attr_train_1, np.expand_dims(zscores, axis= 1)), axis = 1)
        node_attr_train_1 = np.concatenate((node_attr_train_1, np.expand_dims(zscores, axis= 1), np.expand_dims(mean_tensor, axis =1) , np.expand_dims(std_tensor,axis = 1),np.expand_dims( var_tensor, axis = 1), np.expand_dims( min_tensor, axis = 1), np.expand_dims( kur_tensor, axis = 1), np.expand_dims( skews_tensor, axis = 1)), axis = 1)

        return(nodes, edges, node_attr_train_1)

"""
*Note: Execute without arguments only if the deafult root directories match your folder structure or else pass the correct folder
*      and file locations 

"""
def main(): 

    #input_file_path = ModifyGameData.pre_process_predict_results(os.path.join(os.getcwd(),'enter file location'))
    #ModifyGameData.modify_game_file(results= ModifyGameData.read_dataset(os.path.join(os.getcwd(),'enter file location')), output_file_location='enter location' )
    #ModifyGameData.pre_process_train_dataset('enter file location', 'enter file location', 'enter file location')
    ModifyGameData.modify_game_for_3_class()
if __name__ == "__main__":
    main()
        
            