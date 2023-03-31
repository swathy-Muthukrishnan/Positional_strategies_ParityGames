class configurations():
    
    def __init__(self):

        self.config = dict(

            
            
            Experiment_Name = 'Test_GAT_256_3class', 
            classes=3,
            batch_size= 1000,
            epochs = 35,
            learning_rate=0.001,
            static_lr = True,
            Dynamic_lr_type = 'Exponential',
            lr_decay = 0.5,
            MultiStepLR = False,
            MultiStepLR_milestones = [0.2, 1.0],
            MultiStepLR_decay = 0.1, 
            dataset="Parity_Games",
            modelFileName = "GAT_3C_256_static_lr.pth",
            prediction_result = 'results_edge_classification.csv',
            hidden_channels_edges = 256,
            hidden_channels_nodes = 256,
            node_feat = 16, 
            edge_feat = 16,
            message_passing_iterations = 10,
            architecture="GATConV + GAT", 
            weights_for_loss = [0.2, 1.0, 1.0], 
            artifact_model_name = 'GAT_3C_256',
            train_data_root = 'pg_data_3Class',
            test_data_root = 'pg_test',
            test_data_size = 999,
            classification_report = 'classification_report.csv',
            Use_test_data_loader = True
            )
            
        
    def return_config(self): 
        return self.config