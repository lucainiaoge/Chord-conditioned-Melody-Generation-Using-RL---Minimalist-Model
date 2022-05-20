# Chord-conditioned Melody Generation Using RL - Minimalist Model

## How to run the code
- Install dependencies

  ```bash
  pip install -r requirements.txt
  ```

- Train the model

  ```bash
  python train_policy.py PPO
  ```

  Optional arguments:

  ```bash
  python train_policy.py PPO --train_steps 500000
  ```
  ```bash
  python train_policy.py PPO --aug True
  ```
  ```bash
  python train_policy.py PPO --overfit False
  ```
  ```bash
  python train_policy.py PPO --n_env 4
  ```
  
  Results will be saved in ```"result/"``` dir.

- Run the model

  After training the model, load the checkpoint and do validation to save the validation results.

  ```bash
  python run_policy.py [policy_name] [root_path_of_output_dir] [checkpoint_path(optional)]
  ```

  For ```[policy_name]```: use ```scale_basic``` to run benchmark policy; use ```random``` to run randompolicy; use ```PPO``` to run trained policy, and in this case the checkpoint path should be given.

- Evaluate the results
  
  After running the model, find the directory which contains the validation results (in json formats, and also in midi for listening). Run

  ```bash
  python evaluate.py [json_dir]
  ```
  The evaluation results are printed after successfully running. Users are free to modify the main function in evaluate.py to check different reward configurations. Harmony reward with scale note configuration takes more time to run, due to music21 chord symbol finding function.