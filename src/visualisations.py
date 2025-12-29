import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def ratio_of_responders(df_erk):
    df = df_erk.copy()

    df['above'] = df['ERKKTR_ratio_norm']<1
    return df.groupby('frame')['above'].mean().apply(lambda x: 2*x if 2*x<1 else 1).reset_index()

def plot_erkktr_with_confidence_intervals(df, t_add, ax=None, show_plot=True):
    """
    Plot ERKKTR ratio over time by well with 95% confidence intervals.
    
    Parameters:
    df: DataFrame with columns ['well', 'frame', 'ERKKTR_ratio']
    t_add: time point for vertical line
    ax: matplotlib axis object (optional)
    show_plot: whether to show the plot (default True)
    
    Returns:
    fig, ax: matplotlib figure and axis objects
    """
    if ax is None:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    colors = plt.cm.Set1(np.linspace(0, 1, df['well'].nunique()))
    
    for i, (well_id, well) in enumerate(df.groupby('well')):
        # Group by frame and calculate statistics
        frame_stats = well.groupby('frame')['ERKKTR_ratio'].agg([
            'mean', 'std', 'sem', 'count'
        ]).reset_index()
        
        # Calculate 95% confidence intervals using SEM and t-distribution
        confidence_level = 0.95
        alpha = 1 - confidence_level
        # Use t-distribution for small sample sizes, normal for large
        ci_multiplier = np.where(
            frame_stats['count'] < 30,
            stats.t.ppf(1 - alpha/2, frame_stats['count'] - 1),
            stats.norm.ppf(1 - alpha/2)
        )
        ci = ci_multiplier * frame_stats['sem']
        
        # Plot line with semi-transparent confidence interval
        color = colors[i]
        # Main line
        ax.plot(frame_stats['frame'], frame_stats['mean'],
                color=color, linewidth=2, label=f'Well {well_id}',
                marker='o', markersize=4, alpha=0.8)
        # Semi-transparent confidence interval
        ax.fill_between(frame_stats['frame'],
                       frame_stats['mean'] - ci,
                       frame_stats['mean'] + ci,
                       color=color, alpha=0.2,
                       label=f'Well {well_id} 95% CI')
    
    ax.axvline(x=t_add, color='grey', linestyle='--')
    # Styling
    ax.set_xlabel('Frame', fontsize=12, fontweight='bold')
    ax.set_ylabel('ERKKTR Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Mean ERKKTR Ratio',
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend with only main lines (not CI)
    handles, labels = ax.get_legend_handles_labels()
    # Keep only every other handle/label (the main lines, not CI)
    main_handles = handles[::2]
    main_labels = labels[::2]
    ax.legend(main_handles, main_labels, loc='best', frameon=True,
             fancybox=True, shadow=True)
    
    # Grid and layout
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    return fig, ax

def plot_median_erkktr(df, t_add, ax=None, show_plot=True):
    """
    Plot median ERKKTR ratio with quartiles.
    
    Parameters:
    df: DataFrame with columns ['well', 'frame', 'ERKKTR_ratio', 'CellType1']
    t_add: time point for vertical line
    ax: matplotlib axis object (optional)
    show_plot: whether to show the plot (default True)
    
    Returns:
    fig, ax: matplotlib figure and axis objects
    """
    if ax is None:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    colors = plt.cm.Set1(np.linspace(0, 1, df['well'].nunique()))
    
    for i, (well_id, well) in enumerate(df.groupby('well')):
        # Group by frame and calculate statistics
        frame_stats = well.groupby('frame')['ERKKTR_ratio'].agg([
            'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
        ]).reset_index().rename({'<lambda_0>': '1st_quantile', '<lambda_1>': '3rd_quantile'}, axis=1)
        
        # Plot line with semi-transparent confidence interval
        color = colors[i]
        treatment = well['CellType1'].iloc[0]
        
        # Main line
        ax.plot(frame_stats['frame'], frame_stats['median'],
                color=color, linewidth=2, label=treatment,
                alpha=0.8)
        
        # Semi-transparent confidence interval
        ax.fill_between(frame_stats['frame'],
                       frame_stats['1st_quantile'],
                       frame_stats['3rd_quantile'],
                       color=color, alpha=0.2,
                       label=f'Well {well_id} 95% CI')
    
    ax.axvline(x=t_add, color='grey', linestyle='--')
    
    # Styling
    ax.set_xlabel('Frame', fontsize=12, fontweight='bold')
    ax.set_ylabel('ERKKTR Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Median ERKKTR Ratio',
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend with only main lines (not CI)
    handles, labels = ax.get_legend_handles_labels()
    # Keep only every other handle/label (the main lines, not CI)
    main_handles = handles[::2]
    main_labels = labels[::2]
    ax.legend(main_handles, main_labels, loc='best', frameon=True,
             fancybox=True, shadow=True)
    
    # Grid and layout
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    return fig, ax

def plot_proportion_non_responders(df, t_add, ax=None, show_plot=True):
    """
    Plot estimated proportion of non-responders.
    
    Parameters:
    df: DataFrame with columns ['well', 'CellType1']
    t_add: time point for vertical line
    ax: matplotlib axis object (optional)
    show_plot: whether to show the plot (default True)
    
    Returns:
    fig, ax: matplotlib figure and axis objects
    """
    if ax is None:
        plt.style.use('seaborn-v0_8-darkgrid')
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure
    
    colors = plt.cm.Set1(np.linspace(0, 1, df['well'].nunique()))
    
    for i, (well_id, well) in enumerate(df.groupby('well')):
        # Group by frame and calculate statistics
        frame_stats = ratio_of_responders(well)
        
        # Plot line
        color = colors[i]
        treatment = well['CellType1'].iloc[0]
        print(treatment)
        
        # Main line
        ax.plot(frame_stats['frame'], frame_stats['above'],
                color=color, linewidth=2, label=treatment,
                alpha=0.8)
    
    ax.axvline(x=t_add, color='grey', linestyle='--')
    
    # Styling
    ax.set_xlabel('Frame', fontsize=12, fontweight='bold')
    ax.set_ylabel('Proportion of non-responders', fontsize=12, fontweight='bold')
    ax.set_title('Estimated proportion of non-responders',
                fontsize=14, fontweight='bold', pad=20)
    
    # Legend with only main lines (not CI)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', frameon=True,
             fancybox=True, shadow=True)
    
    # Grid and layout
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if show_plot:
        plt.tight_layout()
        plt.show()
    
    return fig, ax

def create_combined_erkktr_plots(df, t_add, save_path='combined_erkktr_plots.png', figsize=(10, 18)):
    """
    Create all three ERKKTR plots stacked vertically and save as PNG.
    
    Parameters:
    df: DataFrame for all plots
    t_add: time point for vertical line
    save_path: path to save the combined plot
    figsize: figure size tuple (width, height)
    
    Returns:
    fig: matplotlib figure object
    """

    df['ERKKTR_ratio'] = df['per_ERKKTR_intensity_mean']/df['nuc_ERKKTR_intensity_mean']
    m = df[df['frame']==t_add+2].groupby('well')['ERKKTR_ratio'].mean()

    df['ERKKTR_ratio_norm'] = df.apply(lambda x: x['ERKKTR_ratio']/m[x['well']], axis=1)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    
    # Plot 1: ERKKTR with confidence intervals
    plot_erkktr_with_confidence_intervals(df, t_add, ax=axes[0], show_plot=False)
    
    # Plot 2: Median ERKKTR
    plot_median_erkktr(df, t_add, ax=axes[1], show_plot=False)
    
    # Plot 3: Proportion of non-responders
    plot_proportion_non_responders(df, t_add, ax=axes[2], show_plot=False)
    
    # Remove individual legends from each subplot
    axes[0].legend().remove()
    axes[1].legend().remove()
    axes[2].legend().remove()
    
    # Remove x-axis labels from top two plots (since we're sharing x-axis)
    axes[0].set_xlabel('')
    axes[1].set_xlabel('')
    
    # Create a single legend at the bottom
    # Get handles and labels from the middle plot (median plot) since it has treatment labels
    handles, labels = axes[1].get_legend_handles_labels()
    main_handles = handles[::2]  # Keep only main lines, not CI
    main_labels = labels[::2]
    
    # Add the legend below the last subplot
    fig.legend(main_handles, main_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0), ncol=len(main_labels), 
               frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout and save
    #plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for the legend
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined plots saved to: {save_path}")
    
    return fig