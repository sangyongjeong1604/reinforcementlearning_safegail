python scripts/imitate_mj.py --mode ga --data trajectories/expert_trajectories-Hopper --limit_trajs 25 --data_subsamp_freq 20 --env_name Hopper-v1 --log training_logs/Hopper/ga_501-iter_Hopper_CVaR_lambda_0.5.h5 --max_iter 501 --useCVaR --CVaR_Lambda_not_trainable --CVaR_Lambda_val_if_not_trainable 0.5