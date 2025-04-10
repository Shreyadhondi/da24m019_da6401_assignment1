if __name__=="__main__":
    with open("sweep_config.yaml") as f:
        sweep_config = yaml.safe_load(f)
    print(sweep_config)