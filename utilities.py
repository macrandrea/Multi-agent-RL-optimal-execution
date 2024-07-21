import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math as m

def heat_v(states):
    arr = states
    df = pd.DataFrame(np.nan, index=np.arange(0, 21), columns=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'])
    #df = df.fillna(-1)
    z = df.to_numpy()
    ds = pd.DataFrame(arr, columns=['Q', 'T', 'x'])
    h = ds.groupby(['Q', 'T']).mean().reset_index()
    a = h.to_numpy()

    for i in range(31):  # Maximum inventory is 20
        for ii in range(0, 11):  # Maximum time is 5
            for iii in range(len(a)):
                if i == a[iii, 0] and ii == a[iii, 1]:
                    z[int(i), int(ii)] = a[iii, 2]
    sns.heatmap(z, cmap="YlGnBu")
    plt.xlabel('Time')
    plt.ylabel('Inventory') 
    plt.title('Heatmap of the average action per inventory and time step')
    plt.show()

def column_min_max_normalize(matrix, min, max):#, data
    """
    Normalizes a matrix of real numbers between 1 and -1 domain using min-max normalization.
    """
    # Find the minimum and maximum values for each column
    #scaler = pre.MinMaxScaler(feature_range=(-1, 1)).fit(matrix)
    min_vals = min#np.min(matrix, axis=0)#np.min(data)#
    max_vals = max#np.max(matrix, axis=0)#np.max(data)#
    range_vals = max_vals - min_vals
    
    # Perform column-wise min-max normalization
    normalized_matrix = 2 * (matrix - min_vals) / range_vals - 1
    
    if normalized_matrix.shape != ():
        for i in range(normalized_matrix.shape[0]):
            for ii in range(normalized_matrix.shape[1]):
                if normalized_matrix[i,ii] > 1: normalized_matrix[i,ii] = 1
                elif normalized_matrix[i,ii] <-1: normalized_matrix[i,ii] = -1

    return normalized_matrix 

def heatAct(data, min, max):

    n = 101
    #data = mu0qtp
    FILLER = -1
    def heatdn(states, thr1, thr2, h_min, h_max):
        arr = states
        df = pd.DataFrame(np.nan, index=np.arange(-1,n), columns=['1', '2', '3', '4', '5','6','7','8','9','10'])
        df = df.fillna(np.nan)
        z = df.to_numpy()#np.zeros((21,5))
        ds = pd.DataFrame(arr,columns=['Q', 'T', 'p', 'x'])
        ds['p'] = ds['p'].apply(column_min_max_normalize, args=(h_min, h_max))
        h = ds.where(ds['p'] >= thr1).where(ds['p'] <= thr2).groupby(['Q','T','p']).mean().reset_index()
        a = h.to_numpy()
        for i in range(n): #mx 20
            for ii in range(10): #mx 5
                for iii in range(len(a[:])):
                    if i == a[:,0][iii] and ii == a[:,1][iii]:
                        z[int(i),int(ii)] = a[iii,3]
        return z    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 20))

    # Loop through each subplot and generate a heatmap for it
    cmap = sns.color_palette("YlGnBu", as_cmap=True)
    cmap.set_bad((1,1,0.7))
    #ns.heatmap(z, cmap=cmap)
    for i in range(2):
        for j in range(2):
            if i == 0 and j == 0:

                ax = sns.heatmap(heatdn(data,-0.75  ,-0.5, min,  max), ax=axs[i,j], cmap=cmap)
                ax.set_title('$-1 \leq \\bar{S} \leq -0.5$')#                

            elif i == 0 and j == 1:
                ax =sns.heatmap(heatdn(data,-0.5  ,0, min,  max), ax=axs[i,j],  cmap=cmap)
                ax.set_title('$-0.5 \leq \\bar{S} \leq 0$')#
            elif i == 1 and j == 0:
                ax =sns.heatmap(heatdn(data,0  ,0.5, min,  max), ax=axs[i,j],   cmap=cmap)
                ax.set_title('$0 \leq \\bar{S} \leq 0.5$')#
            else:
                ax =sns.heatmap(heatdn(data,0.5  ,0.75, min,  max), ax=axs[i,j],   cmap=cmap)
                ax.set_title('$0.5 \leq \\bar{S} \leq 1$')#
            #for h in range(len(q)):
            #    rect = plt.Rectangle((h, int(q[h])), 1, 1, fill=None, edgecolor='red', linewidth=1)
            #    ax.add_patch(rect)   
    # Add a main title to the figure
    fig.suptitle('Average $v$ conditioned to Q,T,$\\bar{S}$')

    # Show the figure
    plt.show()
    
def remove_outliers(data):
    # Calculate the IQR (Interquartile Range)
    Q1 = np.percentile(data, 10)
    Q3 = np.percentile(data, 90)
    IQR = Q3 - Q1

    # Define lower and upper bounds to identify outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Remove outliers
    filtered_data = data[(data >= lower_bound) & (data <= upper_bound)]
    return filtered_data

def rewards_per_episode(rew_0, rew_1, rew_sch_0, rew_sch_1):
    # Load rewards data
    rewards_0 = rew_0
    rewards_1 = rew_1

    # Load rewards_sch data
    rewards_sch_0 = rew_sch_0
    rewards_sch_1 = rew_sch_1

    # Time steps to plot
    time_steps_to_plot = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500]

    # Create figure and subplots
    fig, axs = plt.subplots(2, len(time_steps_to_plot)//2, figsize=(20, 8))

    # Store all data points in lists
    all_rewards_0 = []
    all_rewards_1 = []

    # Plot scatter plots for specified time steps
    for i, time_step in enumerate(time_steps_to_plot):
        row = i // (len(time_steps_to_plot)//2)
        col = i % (len(time_steps_to_plot)//2)
        ax = axs[row, col]

        # Generate a list of colors for each point
        colors = plt.cm.jet_r(np.linspace(0, 1, len(rewards_0)))  # Different colormap for rewards_0
        colors_sch = plt.cm.Set2(np.linspace(0, 1, len(rewards_sch_0)))  # Different colormap for rewards_sch

        # Plot rewards_0 and rewards_1
        for j in range(len(rewards_0)):
            ax.scatter(rewards_0[j, time_step], rewards_1[j, time_step], color=colors[j], s=300, alpha=1, label=f'time step {j+1}')
            ax.text(rewards_0[j, time_step], rewards_1[j, time_step], str(j+1), ha='center', va='center', fontsize=8, color='white')

        # Plot rewards_sch_0 and rewards_sch_1
        for k in range(len(rewards_sch_0)):
            ax.scatter(rewards_sch_0[k, time_step], rewards_sch_1[k, time_step], color=colors_sch[k], s=300, alpha=0.25, label=f'time step Sch. {k+1}')
            ax.text(rewards_sch_0[k, time_step], rewards_sch_1[k, time_step], str(k+1), ha='center', va='center', fontsize=8, color='white')

        ax.set_xlabel('Reward Ag. 1')
        ax.set_ylabel('Reward Ag. 2')
        ax.hlines(0, -300, 300, colors='k', linestyles='dashed', alpha=0.5)
        ax.vlines(0, -300, 300, colors='k', linestyles='dashed', alpha=0.5)
        ax.set_title(f'Scatter Plot episode {time_step}')

        # Store all data points
        all_rewards_0.extend(rewards_0[:, time_step])
        all_rewards_1.extend(rewards_1[:, time_step])

    # Set the same x-axis and y-axis limits for all subplots
    #for ax in axs.flat:
    #    ax.set_xlim(-70, 70)
    #    ax.set_ylim(-70, 70)
    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1), title='Legend')
    
    # Adjust spacing between subplots
    plt.tight_layout()

    # Show the plots
    plt.show()

def rewards_per_episode_1_solo(rew_0, rew_1, rew_sch_0, rew_sch_1, time_step):
    # Load rewards data
    rewards_0 = rew_0
    rewards_1 = rew_1

    # Load rewards_sch data
    rewards_sch_0 = rew_sch_0
    rewards_sch_1 = rew_sch_1

    # Create figure and subplot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Generate a list of colors for each point
    colors = plt.cm.jet_r(np.linspace(0, 1, len(rewards_0)))  # Different colormap for rewards_0
    colors_sch = plt.cm.Set2(np.linspace(0, 1, len(rewards_sch_0)))  # Different colormap for rewards_sch

    # Plot rewards_0 and rewards_1
    for j in range(len(rewards_0)):
        ax.scatter(rewards_0[j, time_step], rewards_1[j, time_step], color=colors[j], s=300, alpha=1, label=f'time step {j+1}')
        ax.text(rewards_0[j, time_step], rewards_1[j, time_step], str(j+1), ha='center', va='center', fontsize=8, color='white')

    # Plot rewards_sch_0 and rewards_sch_1
    for k in range(len(rewards_sch_0)):
        ax.scatter(rewards_sch_0[k, time_step], rewards_sch_1[k, time_step], color=colors_sch[k], s=300, alpha=0.25, label=f'time step Sch. {k+1}')
        ax.text(rewards_sch_0[k, time_step], rewards_sch_1[k, time_step], str(k+1), ha='center', va='center', fontsize=8, color='white')

    ax.set_xlabel('Reward Ag. 1')
    ax.set_ylabel('Reward Ag. 2')
    ax.hlines(0, -300, 300, colors='k', linestyles='dashed', alpha=0.5)
    ax.vlines(0, -300, 300, colors='k', linestyles='dashed', alpha=0.5)
    ax.set_title(f'Scatter Plot episode {time_step}')

    # Add legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', title='Legend')

    # Show the plot
    plt.show()

def rewards_per_simulation(re_tot, rewards_sch):

    #rewards_sch_0 = rewards_sch[:, 0]
    #rewards_sch_1 = rewards_sch[:, 1]



    # Assuming re_tot, rewards_0, rewards_1, and rewards_sch_0 have been defined earlier

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns

    colors = plt.cm.jet_r(np.linspace(0, 1, re_tot.shape[2]))  # Different colormap for rewards_0
    colors_sch = plt.cm.Set2(np.linspace(0, 1, len(rewards_sch[:, 0])))

    for simu in range(min(re_tot.shape[0], 10)):  # Iterate over simulation steps, limited to 10
        row = simu // 5  # Determine the row index
        col = simu % 5  # Determine the column index
        ax = axs[row, col]
        for j in range(re_tot.shape[2]):  # Iterate over the simulation steps
            # Accessing individual elements of re_tot[simu, 0, j] and re_tot[simu, 1, j]
            ax.scatter(re_tot[simu, 0, j, -1].item(), re_tot[simu, 1, j, -1].item(),
                       color=colors[j], s=300, alpha=1, label=f'time step RL {j+1}')
            ax.text(re_tot[simu, 0, j, -1].item(), re_tot[simu, 1, j, -1].item(),
                    str(j), ha='center', va='center', fontsize=8, color='white')


            ax.set_xlabel('Reward Ag. 1')
            ax.set_ylabel('Reward Ag. 2')
            ax.hlines(0, -400, 400, colors='k', linestyles='dashed', alpha=0.5)
            ax.vlines(0, -450, 400, colors='k', linestyles='dashed', alpha=0.5)
            ax.set_title(f'Scatter Plot simulation {simu}')


            ax.scatter(rewards_sch[simu, 0, j, -1].item(), rewards_sch[simu, 1, j, -1].item(),
                         color=colors_sch[j], s=300, alpha=0.25, label=f'time step Nash. {j}')
            ax.text(rewards_sch[simu, 0, j, -1].item(), rewards_sch[simu, 1, j, -1].item(),
                    str(j), ha='center', va='center', fontsize=8, color='white')

        # Calculate correlation coefficient
        corr_coef = np.corrcoef(re_tot[simu, 0, :, -1], re_tot[simu, 1, :, -1])[0, 1]
        # Print correlation coefficient below the plot
        #ax.text(0.5, -0.2, f'Correlation Coefficient: {corr_coef:.2f}', ha='center', va='center', transform=ax.transAxes)

    for ax in axs.flat:
        ax.set_xlim(-400,300)
        ax.set_ylim(-450,300)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.9), title='Legend:')
    fig.suptitle('Scatter Plots of Average Rewards per Time Step for Unconstrained Agents over 10 Simulations')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

def plot_rewards(rewards_file_path, rewards_1_file_path, rewards_2_file_path, rewards_3_file_path, rewards_sch_file_path):
    
    # Load rewards data
    rewards = np.load(rewards_file_path, allow_pickle=True)
    rewards_1 = np.load(rewards_1_file_path, allow_pickle=True)
    rewards_2 = np.load(rewards_2_file_path, allow_pickle=True)
    rewards_3 = np.load(rewards_3_file_path, allow_pickle=True)
    rewards_sch = (np.load(rewards_sch_file_path, allow_pickle=True))#*0+ 1.122
    
    # Create a grid of subplots
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.5))
    i = 1
    # Plot the histograms in each subplot
    axs[0].set_title('$\sigma = 0$')
    axs[0].set_ylim(0, 500)     
    axs[0].axvline(1.122, color='r', linestyle='dashed', linewidth=1, label='Nash')  
    axs[0].axvline((rewards[:,0].mean(axis=1).mean(axis = 0).mean() + rewards[:,1].mean(axis=1).mean(axis = 0).mean())/2, color='green', linestyle= 'dotted', label = 'average Agents')
    axs[0].axvline(rewards[:,0].mean(axis=0).mean(axis = 1).mean(), color='blue', linestyle= 'dotted', label = 'average Agent 1')  
    axs[0].axvline(rewards[:,1].mean(axis=0).mean(axis = 1).mean(), color='orange', linestyle= 'dotted', label = 'average Agent 2')  
    sns.histplot(remove_outliers(rewards[:,0].mean(axis=0).mean(axis = 1)), bins=100, kde=True, ax=axs[0], label = 'Agent 1')
    sns.histplot(remove_outliers(rewards[:,1].mean(axis=0).mean(axis = 1)), bins=100, kde=True, ax=axs[0], label = 'Agent 2')
    #sns.histplot(remove_outliers(rewards_sch[:,0].mean(axis=1).mean(axis = 0)), bins=100, kde=True, ax=axs[0], label = 'Nash')
    #axs[1].axvline(rewards_1[:,0].mean(axis=1).mean(axis = 0).mean(), color='blue', linestyle= 'dotted', label = 'average Agent 1') 
    #axs[1].axvline(rewards_1[:,1].mean(axis=1).mean(axis = 0).mean(), color='orange', linestyle= 'dotted', label = 'average Agent 2') 
    #axs[1].axvline(1.122, color='r', linestyle='dashed', linewidth=1, label='Nash')     
    #axs[1].set_title('$\sigma = 0.0001$')
    #axs[1].set_ylim(0, 500)            
    #sns.histplot(remove_outliers(rewards_1[:,0].mea.n(axis=1).mean(axis = 0)), bins=100, kde=True, ax=axs[1], label = 'Agent 1')
    #sns.histplot(remove_outliers(rewards_1[:,1].mean(axis=1).mean(axis = 0)), bins=100, kde=True, ax=axs[1], label = 'Agent 2')
    #sns.histplot(remove_outliers(rewards_sch[:,0].mean(axis=1).mean(axis = 0)), bins=100, kde=True, ax=axs[1], label = 'Nash')
    axs[1].axvline(rewards_1[:,0].mean(axis=1).mean(axis = 0).mean(), color='blue', linestyle= 'dotted', label = 'average Agent 1')
    axs[1].axvline(rewards_1[:,1].mean(axis=1).mean(axis = 0).mean(), color='orange', linestyle= 'dotted', label = 'average Agent 2')
    axs[1].axvline(1.122, color='r', linestyle='dashed', linewidth=1, label='Nash')
    axs[1].axvline((rewards_1[:,0].mean(axis=1).mean(axis = 0).mean() + rewards_1[:,1].mean(axis=1).mean(axis = 0).mean())/2, color='green', linestyle= 'dotted', label = 'average Agents')
    axs[1].set_title('$\sigma = 0.0001$')
    axs[1].set_ylim(0, 500)            
    sns.histplot(remove_outliers(rewards_1[:,0].mean(axis=1).mean(axis = 0)), bins=100, kde=True, ax=axs[1], label = 'Agent 1')
    sns.histplot(remove_outliers(rewards_1[:,1].mean(axis=1).mean(axis = 0)), bins=100, kde=True, ax=axs[1], label = 'Agent 2')
    #sns.histplot(remove_outliers(rewards_sch[:,0].mean(axis=1).mean(axis = 0)), bins=100, kde=True, ax=axs[1], label = 'Nash')
    axs[2].axvline(rewards_2[:,0].mean(axis=1).mean(axis = 0).mean(), color='blue', linestyle= 'dotted', label = 'average Agent 1')
    axs[2].axvline(rewards_2[:,1].mean(axis=1).mean(axis = 0).mean(), color='orange', linestyle= 'dotted', label = 'average Agent 2')
    axs[2].axvline(1.122, color='r', linestyle='dashed', linewidth=1, label='Nash')
    axs[2].axvline((rewards_2[:,0].mean(axis=1).mean(axis = 0).mean() + rewards_2[:,1].mean(axis=1).mean(axis = 0).mean())/2, color='green', linestyle= 'dotted', label = 'average Agents')
    axs[2].set_title('$\sigma = 0.001$')
    axs[2].set_ylim(0, 500)            
    sns.histplot(remove_outliers(rewards_2[:,0].mean(axis=1).mean(axis = 0)), bins=100, kde=True,   ax=axs[2], label = 'Agent 1')
    sns.histplot(remove_outliers(rewards_2[:,1].mean(axis=1).mean(axis = 0)), bins=100, kde=True,   ax=axs[2], label = 'Agent 2')
    #sns.histplot(remove_outliers(rewards_sch[:,0].mean(axis=1).mean(axis = 0)), bins=100, kde=True, ax=axs[2], label = 'Nash')
    axs[3].axvline(rewards_3[:,0].mean(axis=1).mean(axis = 0).mean(), color='blue', linestyle= 'dotted', label = 'average Agent 1')
    axs[3].axvline(rewards_3[:,1].mean(axis=1).mean(axis = 0).mean(), color='orange', linestyle= 'dotted', label = 'average Agent 2')
    axs[3].axvline(1.122, color='r', linestyle='dashed', linewidth=1, label='Nash')
    axs[3].axvline((rewards_3[:,0].mean(axis=1).mean(axis = 0).mean() + rewards_3[:,1].mean(axis=1).mean(axis = 0).mean())/2, color='green', linestyle= 'dotted', label = 'average Agents')
    axs[3].set_title('$\sigma = 0.01$')
    axs[3].set_ylim(0, 500)            
    sns.histplot(remove_outliers(rewards_3[:,0].mean(axis=1).mean(axis = 0)), bins=100, kde=True,   ax=axs[3], label = 'Agent 1')
    sns.histplot(remove_outliers(rewards_3[:,1].mean(axis=1).mean(axis = 0)), bins=100, kde=True,   ax=axs[3], label = 'Agent 2')
 # Adjust the spacing between subplots
    plt.tight_layout()

    plt.legend()

    # Show the figure
    plt.show()

    # Print statistics
    print(f'IS mean via rewards for RL agents : {remove_outliers((rewards[:,0]  .mean(axis=1).mean() + rewards[:,1]  .mean(axis=1).mean())/2).item():.2f}, std: {remove_outliers((rewards[:,0]  .mean(axis=1) + rewards[:,1]  .mean(axis=1))).std():.2f}')
    print(f'IS mean via rewards for RL agents : {remove_outliers((rewards_1[:,0].mean(axis=1).mean() + rewards_1[:,1].mean(axis=1).mean())/2).item():.2f}, std: {remove_outliers((rewards_1[:,0].mean(axis=1) + rewards_1[:,1].mean(axis=1))).std():.2f}')
    print(f'IS mean via rewards for RL agents : {remove_outliers((rewards_2[:,0].mean(axis=1).mean() + rewards_2[:,1].mean(axis=1).mean())/2).item():.2f}, std: {remove_outliers((rewards_2[:,0].mean(axis=1) + rewards_2[:,1].mean(axis=1))).std():.2f}')
    print(f'IS mean via rewards for RL agents : {remove_outliers((rewards_3[:,0].mean(axis=1).mean() + rewards_3[:,1].mean(axis=1).mean())/2).item():.2f}, std: {remove_outliers((rewards_3[:,0].mean(axis=1) + rewards_3[:,1].mean(axis=1))).std():.2f}')

def process_data(file_path):
    INV = 100
    azioni = (np.load(file_path, allow_pickle=True))
    
    azionimu0 = azioni[: ,0].mean(axis=0).mean(axis=1)
    q0 = np.zeros(11)
    q0[0] = INV
    for i in range(1, 10):
        q0[i] = q0[i - 1] - azionimu0[i - 1]

    azionimu0qtp = azioni[:, 1].mean(axis=0).mean(axis=1)
    q = np.zeros(11)
    q[0] = INV
    for i in range(1, 10):
        q[i] = q[i - 1] - azionimu0qtp[i - 1]

    azioni_tw = np.ones((5_000, 10)) * 2
    twat = azioni_tw.reshape(-1, 10).mean(0)
    twap = np.zeros(11)
    twap[0] = INV
    for i in range(1, 10):
        twap[i] = twap[i - 1] - twat[i - 1]

    T = 10  # Assuming T is defined somewhere in your code
    q_0 = 200  # Assuming q_0 is defined somewhere in your code
    azioni1 = azioni[:, 0].mean(axis=1).mean(axis=0)
    azioni2 = azioni[:, 1].mean(axis=1).mean(axis=0)
    azioni_combined = (azioni1 + azioni2)  #/2
    qt = np.zeros(T + 1)
    qt[0] = q_0
    for i in range(1, T):
        qt[i] = qt[i - 1] - azioni_combined[i - 1]
    
    return q0, q, qt
    

def rewards_per_simulation_hist(re_tot, rewards_sch):

    #rewards_sch_0 = rewards_sch[:, 0]
    #rewards_sch_1 = rewards_sch[:, 1]



    # Assuming re_tot, rewards_0, rewards_1, and rewards_sch_0 have been defined earlier

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns

    colors = plt.cm.jet_r(np.linspace(0, 1, re_tot.shape[2]))  # Different colormap for rewards_0
    colors_sch = plt.cm.Set2(np.linspace(0, 1, len(rewards_sch[:, 0])))

    for simu in range(min(re_tot.shape[0], 10)):  # Iterate over simulation steps, limited to 10
        row = simu // 5  # Determine the row index
        col = simu % 5  # Determine the column index
        ax = axs[row, col]
        for j in range(re_tot.shape[2]):  # Iterate over the simulation steps
            # Accessing individual elements of re_tot[simu, 0, j] and re_tot[simu, 1, j]
            ax.hist(remove_outliers(re_tot[simu, 0].mean(0).flatten()), bins=100, alpha = 0.1, color = 'green', label= j)#f'agente 1' if j == 0 else None
            ax.hist(remove_outliers(re_tot[simu, 1].mean(0).flatten()), bins=100, alpha = 0.1, color = 'blue' , label= j)#f'agente 2' if j == 0 else None
            #ax.hist(remove_outliers(rewards_sch[simu, 0].mean(0).flatten()), bins=100, alpha = 0.2, color = 'red', label = 'agente Nash'if j == 0 else None)
            ax.vlines(1.12, 0, 150, color = 'red', label = 'Nash Agent'if j == 0 else None)
            ax.set_title(f"Simulation {simu}")
            ax.set_xlabel('Rewards')
            ax.set_ylabel('Frequency')
            ax.vlines((re_tot[simu, 0].mean(0).mean(0)+re_tot[simu, 1].mean(0).mean(0))/2, 0, 150, color='orange', label='Agent Mean' if j == 0 else None)


        # Calculate correlation coefficient
        #corr_coef = np.corrcoef(re_tot[simu, 0, :, -1], re_tot[simu, 1, :, -1])[0, 1]
        # Print correlation coefficient below the plot
        #ax.text(0.5, -0.2, f'Correlation Coefficient: {corr_coef:.2f}', ha='center', va='center', transform=ax.transAxes)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.9), title='Legend:')
    fig.suptitle('Scatter Plots of Average Rewards per Time Step for Unconstrained Agents over 10 Simulations')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()


def make_selling_schedule(a, azioni, T, i=0, inv=100):
    azioni_tot_1 = azioni[i, 0]
    azioni_tot_2 = azioni[i, 1]
    agent_1 = azioni_tot_1.mean(axis=1)
    agent_2 = azioni_tot_2.mean(axis=1)
    
    q0_1 = np.zeros(T + 1)
    q0_1[0] = inv
    for j in range(1, T):
        q0_1[j] = q0_1[j - 1] - agent_1[j - 1]
    q0_2 = np.zeros(T + 1)
    q0_2[0] = inv
    for j in range(1, T):
        q0_2[j] = q0_2[j - 1] - agent_2[j - 1]
    q_tot = inv# * 2
    azioni_t = (agent_1 + agent_2) / 2
    qt = np.zeros(T + 1)
    qt[0] = q_tot
    for j in range(1, T):
        qt[j] = qt[j - 1] - azioni_t[j - 1]

    azioni_tw = np.ones((5_000,10)) * 10
    twat = azioni_tw.reshape(-1,10).mean(0)
    twap = np.zeros(11) 
    twap[0] = 100
    for i in range(1,10):
        twap[i] =  twap[i - 1] - twat[i - 1]

    b = np.asarray(a) / 2

    ax = plt.gca()  # Add this line to define "ax"
    ax.plot(q0_1, label='Agent 1' if i == 0 else None)
    ax.plot(q0_2, label='Agent 2' if i == 0 else None)
    ax.plot(qt, alpha = 0.5, label='Mean Agents' if i == 0 else None, linestyle='--')
    ax.plot(b, label='Nash Agent'if i == 0 else None, linestyle=':')
    ax.plot(twap, label = 'twap' if i == 0 else None, linestyle='-.')

def do_is(T, i, dati, azioni, alpha=0.002):
    dati =      dati[i,:,0]
    azioni1 =  azioni[i, 0]
    azioni2 =  azioni[i, 0]
    azioni = (azioni1 + azioni2)
    iss = []

    for i in range(dati.reshape(-1,T).shape[0]):
        iss.append((dati.reshape(-1,T)[i])* azioni[:,i] - alpha * azioni[:,i]**2)

    agents = np.sum((np.asarray(iss)),axis=1)
    agents_std = np.sum((np.asarray(iss)),axis=1).std()

    return 2000-remove_outliers(agents).mean(),  agents_std

def rewards_per_simulation_hist(re_tot, rewards_sch):

    #rewards_sch_0 = rewards_sch[:, 0]
    #rewards_sch_1 = rewards_sch[:, 1]



    # Assuming re_tot, rewards_0, rewards_1, and rewards_sch_0 have been defined earlier

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))  # 2 rows, 5 columns

    colors = plt.cm.jet_r(np.linspace(0, 1, re_tot.shape[2]))  # Different colormap for rewards_0
    colors_sch = plt.cm.Set2(np.linspace(0, 1, len(rewards_sch[:, 0])))

    for simu in range(min(re_tot.shape[0], 10)):  # Iterate over simulation steps, limited to 10
        row = simu // 5  # Determine the row index
        col = simu % 5  # Determine the column index
        ax = axs[row, col]
        for j in range(re_tot.shape[2]):  # Iterate over the simulation steps
            # Accessing individual elements of re_tot[simu, 0, j] and re_tot[simu, 1, j]
            ax.hist(remove_outliers(re_tot[simu, 0].mean(0).flatten()), bins=100, alpha = 0.1, color = 'green', label= j)#f'agente 1' if j == 0 else None
            ax.hist(remove_outliers(re_tot[simu, 1].mean(0).flatten()), bins=100, alpha = 0.1, color = 'blue' , label= j)#f'agente 2' if j == 0 else None
            #ax.hist(remove_outliers(rewards_sch[simu, 0].mean(0).flatten()), bins=100, alpha = 0.2, color = 'red', label = 'agente Nash'if j == 0 else None)
            ax.vlines(1.12, 0, 150, color = 'red', label = 'Nash Agent'if j == 0 else None)
            ax.set_title(f"Simulation {simu}")
            ax.set_xlabel('Rewards')
            ax.set_ylabel('Frequency')
            ax.vlines((re_tot[simu, 0].mean(0).mean(0)+re_tot[simu, 1].mean(0).mean(0))/2, 0, 150, color='orange', label='Agent Mean' if j == 0 else None)


        # Calculate correlation coefficient
        #corr_coef = np.corrcoef(re_tot[simu, 0, :, -1], re_tot[simu, 1, :, -1])[0, 1]
        # Print correlation coefficient below the plot
        #ax.text(0.5, -0.2, f'Correlation Coefficient: {corr_coef:.2f}', ha='center', va='center', transform=ax.transAxes)


    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 0.9), title='Legend:')
    fig.suptitle('Scatter Plots of Average Rewards per Time Step for Unconstrained Agents over 10 Simulations')

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

def process_data(file_path):
    INV = 100
    azioni = (np.load(file_path, allow_pickle=True))
    
    azionimu0 = azioni[:,0].mean(axis=0).mean(axis=1)
    q0 = np.zeros(11)
    q0[0] = INV
    for i in range(1, 10):
        q0[i] = q0[i - 1] - azionimu0[i - 1]

    azionimu0qtp = azioni[:,1].mean(axis=0).mean(axis=1)
    q = np.zeros(11)
    q[0] = INV
    for i in range(1, 10):
        q[i] = q[i - 1] - azionimu0qtp[i - 1]

    azioni_tw = np.ones((5_000, 10)) * 2
    twat = azioni_tw.reshape(-1, 10).mean(0)
    twap = np.zeros(11)
    twap[0] = INV
    for i in range(1, 10):
        twap[i] = twap[i - 1] - twat[i - 1]

    T = 10  # Assuming T is defined somewhere in your code
    q_0 = 100  # Assuming q_0 is defined somewhere in your code
    azioni1 = azioni[:, 0].mean(axis=0).mean(axis=1)
    azioni2 = azioni[:, 1].mean(axis=0).mean(axis=1)
    azioni_combined = (azioni1 + azioni2)  #/2
    qt = np.zeros(T + 1)
    qt[0] = q_0
    for i in range(1, T):
        qt[i] = qt[i - 1] - azioni_combined[i - 1]
    
    return q0, q, qt, twap

def process_data_no_path(azioni):
    INV = 100
    
    azionimu0 = azioni[:,0].mean(axis=0).mean(axis=1)
    q0 = np.zeros(11)
    q0[0] = INV
    for i in range(1, 10):
        q0[i] = q0[i - 1] - azionimu0[i - 1]

    azionimu0qtp = azioni[:,1].mean(axis=0).mean(axis=1)
    q = np.zeros(11)
    q[0] = INV
    for i in range(1, 10):
        q[i] = q[i - 1] - azionimu0qtp[i - 1]

    azioni_tw = np.ones((5_000, 10)) * 10
    twat = azioni_tw.reshape(-1, 10).mean(0)
    twap = np.zeros(11)
    twap[0] = INV
    for i in range(1, 10):
        twap[i] = twap[i - 1] - twat[i - 1]

    T = 10  # Assuming T is defined somewhere in your code
    q_0 = 100  # Assuming q_0 is defined somewhere in your code
    azioni1 = azioni[:, 0].mean(axis=0).mean(axis=1)
    azioni2 = azioni[:, 1].mean(axis=0).mean(axis=1)
    azioni_combined = (azioni1 + azioni2)  #/2
    qt = np.zeros(T + 1)
    qt[0] = q_0
    for i in range(1, T):
        qt[i] = qt[i - 1] - azioni_combined[i - 1]
    
    return q0, q, qt, twap