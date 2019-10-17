import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
plt.style.use('seaborn-dark')

def animate(i):
    pullData = open("best_value2.txt","r").read()
    dataArray = pullData.split('\n')
    xar = []
    yar = []
    for eachLine in dataArray:
        if len(eachLine)>1:
            x,y, best_index, individuals1, individuals2 = eachLine.split(',')
            teste = individuals1.split('\t')[-1:-1]
            xar.append(float(x))
            yar.append(float(y))
    ax1.clear()
    ax1.plot(xar,yar)


ani = animation.FuncAnimation(fig, animate, interval=1000)
plt.show()