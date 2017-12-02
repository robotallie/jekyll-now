---
published: false
---


```python
def plot_two_hist(df1,df2):
    """Function takes two data series and creates side by side histogram plots"""
    import matplotlib.pyplot as plt
    fig= plt.figure(figsize=(15,4))
    # Plot 1 
    plt.subplot(1,2,1)
    cm = plt.cm.get_cmap('ocean')
    n, bins, patches = plt.hist(df1, rwidth=0.95, color='green')
    # Normalize  values
    col = (n-n.min())/(n.max()-n.min())
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))
    # Plot 2 
    plt.subplot(1,2,2)
    cm = plt.cm.get_cmap('ocean')
    n, bins, patches = plt.hist(df2, rwidth=0.95, color='green')
    # Normalize values
    col = (n-n.min())/(n.max()-n.min())
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))        
    return
```
