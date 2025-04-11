import wandb

if __name__ == "__main__":
    api = wandb.Api()

    # Your W&B entity and project
    entity = "shreyadhondi-indian-institute-of-technology-madras"
    project = "da24m019_shreya_da6401_assignment1"

    # Get all runs in the project
    runs = api.runs(f"{entity}/{project}")

    # Filter out only finished runs that have validation accuracy
    valid_runs = [run for run in runs if run.state == "finished" and "validation_accuracy" in run.summary]

    # Find the best run by validation accuracy
    best_run = max(valid_runs, key=lambda run: run.summary["validation_accuracy"])

    print(f"\nBest Run across all experiments:")
    print(f"Run Name       : {best_run.name}")
    print(f"Validation Acc : {best_run.summary['validation_accuracy']}")
    print(f"Run URL        : {best_run.url}")
    print("\nHyperparameters:")
    for k, v in best_run.config.items():
        print(f"  {k}: {v}")
