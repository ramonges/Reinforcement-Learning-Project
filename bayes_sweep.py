import yaml
import wandb





def run_sweeping():
    a = wandb.init()



    



def do_sweep():
    with open("sweep.yaml") as file:
        sweep_config = yaml.load(file , Loader=yaml.FullLoader)
    
    sweep_id = wandb.sweep(sweep=sweep_config, project="RL Trading")
    wandb.agent(sweep_id, function=run_sweeping , count=100)




if __name__ == '__main__':
    do_sweep()