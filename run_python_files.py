import os
import subprocess

print(os.getcwd())
os.chdir("./PoinTr")
print(os.getcwd())
for fold in range(2,11):
    try:
        print(f"Running AdaPoinTr fold {fold}...")

        subprocess.run([
            'python',
            'main.py', '--test',
            '--config', './cfgs/PCN_models/AdaPoinTr_fold{}.yaml'.format(fold),
            '--ckpt', './experiments/AdaPoinTr_fold{}/PCN_models/default/ckpt-best.pth'.format(fold)
        ], check=True)
    except subprocess.CalledProcessError as e:
            print(f"Error occurred while running AdaPoinTr: {e}\n")

