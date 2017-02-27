#
# hw3pr1.py 
#
#  lab problem - matplotlib tutorial (and a bit of numpy besides...)
#
# this asks you to work through the first part of the tutorial at   
#     www.labri.fr/perso/nrougier/teaching/matplotlib/
#   + then try the scatter plot, bar plot, and one other kind of "Other plot" 
#     from that tutorial -- and create a distinctive variation of each
#
# include screenshots or saved graphics of your variations of those plots with the names
#   + plot_scatter.png, plot_bar.png, and plot_choice.png
# 
# Remember to run  %matplotlib  at your ipython prompt!
#

#
# in-class examples...
#

def inclass1():
    """
    Simple demo of a scatter plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt


    N = 50
    x = np.random.rand(N)
    y = np.random.rand(N)
    colors = np.random.rand(N)
    area = np.pi * (15 * np.random.rand(N))**2  # 0 to 15 point radiuses
    #alpha is the transparancy 
    #it is random and it changes every time 
    plt.scatter(x, y, s=area, c=colors, alpha=0.5)
    plt.show()



#
# First example from the tutorial/walkthrough
#


#
# Feel free to replace this code as you go -- or to comment/uncomment portions of it...
# 

def example1():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm


    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C,S = np.cos(X), np.sin(X)

    plt.plot(X,C)
    plt.plot(X,S)

    plt.show()






#
# Here is a larger example with many parameters made explicit
#

def example2():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm



    # Create a new figure of size 8x6 points, using 100 dots per inch
    plt.figure(figsize=(8,6), dpi=80)

    # Create a new subplot from a grid of 1x1
    plt.subplot(111)

    X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
    C,S = np.cos(X), np.sin(X)

    # Plot cosine using blue color with a continuous line of width 1 (pixels)
    t = 2*np.pi/3
    plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="cosine")
    plt.plot([t,t],[0,np.cos(t)], color ='blue', linewidth=1.5, linestyle="--")
    plt.scatter([t,],[np.cos(t),], 50, color ='blue')

    plt.annotate(r'$\sin(\frac{2\pi}{3})=\frac{\sqrt{3}}{2}$',
             xy=(t, np.sin(t)), xycoords='data',
             xytext=(+10, +30), textcoords='offset points', fontsize=16,
             arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    # Plot sine using green color with a continuous line of width 1 (pixels)
    plt.plot(X, S, color="red", linewidth=2.5, linestyle="-", label="sine")

    plt.plot([t,t],[0,np.sin(t)], color ='red', linewidth=1.5, linestyle="--")
    plt.scatter([t,],[np.sin(t),], 50, color ='red')

    plt.annotate(r'$\cos(\frac{2\pi}{3})=-\frac{1}{2}$',
                 xy=(t, np.cos(t)), xycoords='data',
                 xytext=(-90, -50), textcoords='offset points', fontsize=16,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    #legend 
    plt.legend(loc = 'upper left', frameon=False)

    # Set x limits
    plt.xlim(X.min()*1.1 , X.max()*1.1) 


    # Set x ticks
    plt.xticks([np.pi, -np.pi/2,0,np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$'])

    # Set y limits
    plt.ylim(C.min()*1.1 , C.max()*1.1) 

    # Set y ticks
    plt.yticks([-1, 0, +1], [r'$-1$', r'$0$', r'$+1$'])

    # Save figure using 72 dots per inch
    # savefig("../figures/exercice_2.png",dpi=72)


    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data',0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data',0))

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(16)
        label.set_bbox(dict(facecolor='white', edgecolor='None', alpha=0.65 ))


    # Show result on screen
    plt.show()



#
# using style sheets:
#   # be sure to               import matplotlib
#   # list of all of them:     matplotlib.style.available
#   # example of using one:    matplotlib.style.use( 'seaborn-paper' )
#


def scatter():
""" This function will create a scatter plot with different colors""" 

    import numpy as np
    import matplotlib.pyplot as plt


    n = 1024
    X = np.random.normal(0,1,n)
    Y = np.random.normal(0,1,n)
    T = np.arctan2(Y,X)

    plt.axes([0.025,0.025,0.95,0.95])
    plt.scatter(X,Y, s=75, c=T, alpha=.5)

    plt.xlim(-3.0 ,1.5), plt.xticks([])
    plt.ylim(-1.0 ,1.5), plt.yticks([])
    plt.show()


def bar ():
"""this bar graph has different colors and labels""" 
    import numpy as np
    import matplotlib.pyplot as plt

    n = 12
    X = np.arange(n)
    Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
    Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)


    plt.axes([0.025,0.025,0.95,0.95])
    plt.bar(X, +Y1, facecolor='#F5A9A9', edgecolor='#BDBDBD')
    plt.bar(X, -Y2, facecolor='#F7819F', edgecolor='#BDBDBD')

    for x,y in zip(X,Y1):
        plt.text(x+0.4, y+0.05, '%.2f' % y, ha='center', va= 'bottom')

    for x,y in zip(X,Y2):
        plt.text(x+0.4, -y-0.05, '%.2f' % y, ha='center', va= 'top')

    plt.xlim(-.5,n), plt.xticks([])
    plt.ylim(-1.25,+1.25), plt.yticks([])

    # savefig('../figures/bar_ex.png', dpi=48)
    plt.show()

def polar(): 
    import numpy as np
    import matplotlib.pyplot as plt

    ax = plt.axes([0.025,0.025,0.95,0.95], polar=True)

    N = 20
    theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
    radii = 10*np.random.rand(N)
    width = np.pi/4*np.random.rand(N)
    bars = plt.bar(theta, radii, width=width, bottom=0.0)

    for r,bar in zip(radii, bars):
        bar.set_facecolor( plt.cm.jet(r/20.))
        bar.set_alpha(0.5)

    ax.set_xticklabels(["hi"])
    ax.set_yticklabels([])
    # savefig('../figures/polar_ex.png',dpi=48)
    plt.show()

