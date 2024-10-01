from util import *
from rl_mdp.model_free_prediction.monte_carlo_evaluator import MCEvaluator
from rl_mdp.model_free_prediction.td_evaluator import TDEvaluator
from rl_mdp.model_free_prediction.td_lambda_evaluator import TDLambdaEvaluator


def main() -> None:
    """
    Starting point of the program, you can instantiate any classes, run methods/functions here as needed.
    """
    mdp = create_mdp()
    policy_1 = create_policy_1()
    policy_2 = create_policy_2()
    
    print("MC value functions: ")

    mc_eval = MCEvaluator(mdp)
    mc_val_func_1 = mc_eval.evaluate(policy_1, 1000)
    print("MC policy 1 value function: ", mc_val_func_1)
    mc_val_func_2 = mc_eval.evaluate(policy_2, 1000)
    print("MC policy 2 value function: ", mc_val_func_2)

    print("TD value functions: ")
    td_eval = TDEvaluator(mdp, 0.1)
    td_val_func_1 = td_eval.evaluate(policy_1, 1000)
    print("TD policy 1 value function: ", td_val_func_1)
    td_val_func_2 = td_eval.evaluate(policy_2, 1000)
    print("TD policy 2 value function: ", td_val_func_2)

    print("TD(lambda) value functions: ")
    td_lambda_eval = TDLambdaEvaluator(mdp, alpha=0.1, lambd=0.5)
    td_lambda_val_func_1 = td_lambda_eval.evaluate(policy_1, 1000)
    print("TD(lambda) policy 1 value function: ", td_lambda_val_func_1)
    td_lambda_val_func_2 = td_lambda_eval.evaluate(policy_2, 1000)
    print("TD(lambda) policy 2 value function: ", td_lambda_val_func_2)


if __name__ == "__main__":
    main()
