import os
from experiments import run_all

if __name__ == '__main__':
    if not os.path.exists('./experiments_results'):
        os.mkdir('./experiments_results')

    if not os.path.exists('./generation_results'):
        os.mkdir('./generation_results')

    run_all('cpu')
