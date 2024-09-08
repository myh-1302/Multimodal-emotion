from MMSA import MMSA_run, get_config_regression

def mutil():
    # get default config of mult on sims
    config = get_config_regression('mtfn', 'sims')

    # alter the default config
    
    config['train_mode'] = 'classification'
    config['batch_size'] = 16
    config['num_classes'] = 3
   
    print(config)
    # check config

    # run with altered config
    MMSA_run('mtfn', 'sims', config=config, model_save_dir='model', res_save_dir='output\multi', log_dir='log\mutli', num_workers=4)
if __name__ == '__main__':
    mutil() 
