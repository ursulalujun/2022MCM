import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

def scaler(X):
    column_headers=X.columns 
    min_max_scaler = preprocessing.MinMaxScaler()    
    scaled_X=pd.DataFrame(min_max_scaler.fit_transform(X))
    scaled_X.columns=column_headers
    return scaled_X

def barplot():
    df=pd.read_excel('现金变化.xlsx')
    df=df.drop_duplicates('cash', keep='first')
    print(df.shape[0])
    df =df [~df.isin([0])].dropna(axis=0)
    df=df.iloc[0:150]
    df2=df.iloc[:,1:]
    df2=scaler(df2)
    cash=df2['cash']*100
    # set figure size
    plt.figure(figsize=(20,10))

    # plot polar axis
    ax = plt.subplot(111, polar=True)

    # remove grid
    plt.axis('off')

    # Compute max and min in the dataset
    max = cash.max()

    # Set the coordinates limits
    upperLimit = 100
    lowerLimit = 0

    # Let's compute heights: they are a conversion of each item value in those new coordinates
    # In our example, 0 in the dataset will be converted to the lowerLimit (10)
    # The maximum will be converted to the upperLimit (100)
    slope = (max - lowerLimit) / max
    heights = slope * cash + lowerLimit

    # Compute the width of each bar. In total we have 2*Pi = 360°
    width = 2*np.pi / len(df.index)

    # Compute the angle each bar is centered on:
    indexes = list(range(1, len(df.index)+1))
    angles = [element * width for element in indexes]

    # Draw bars
    bars = ax.bar(
        x=angles, 
        height=heights, 
        width=width, 
        bottom=lowerLimit,
        linewidth=2, 
        color="lightskyblue",
        edgecolor="white")

    # little space between the bar and the label
    labelPadding = 4

    # Add labels
    for bar, angle, height, label in zip(bars,angles, heights, df["Date"]):

        # Labels are rotated. Rotation must be specified in degrees :(
        rotation = np.rad2deg(angle)

        # Flip some labels upside down
        alignment = ""
        if angle >= np.pi/2 and angle < 3*np.pi/2:
            alignment = "right"
            rotation = rotation + 180
        else: 
            alignment = "left"

        # Finally add the labels
        ax.text(
            x=angle, 
            y=lowerLimit + bar.get_height() + labelPadding, 
            s=label, 
            ha=alignment, 
            va='center', 
            rotation=rotation, 
            rotation_mode="anchor")
    plt.show()

def frequency():
    df=pd.read_excel('同向变化对比.xlsx')
    x=np.arange(0,df.shape[0])
    y=np.array(df).transpose()
    COLORS = ["#D0D1E6", "#A6BDDB", "#74A9CF", "#2B8CBE"]
    fig, ax = plt.subplots(figsize=(10, 7))
    grid = np.linspace(0, 59, num=500)
    y_smoothed = [gaussian_smooth(x, y_, grid, 1) for y_ in y]
    ax.stackplot(grid, y_smoothed, colors=COLORS,baseline ='zero')
    plt.show()

def gaussian_smooth(x, y, grid, sd):
    weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
    weights = weights / weights.sum(0)
    return (weights * y).sum(1)


frequency()