import wandb
import numpy as np
import matplotlib.pyplot as plt
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
run = api.run("garrett4wade/hide_and_seek/3jz079jr")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
print(metrics_dataframe['policy_0/episode_return_hider'].astype(np.float32))
# plt.figure()
# plt.plot(metrics_dataframe['policy_0/episode_return_seeker'].astype(np.float32))
# plt.show()
# metrics_dataframe.to_csv("metrics.csv")
