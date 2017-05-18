#!/usr/bin/python
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import random
class plot_class:
    def plotting(self,record):
        plt.ion()
        #vectors to plot: 4D for this example
        y1=[2,3,5,6,7,19,0]
        #y2=[15,12,4,3,8,10,2]

        x=[1,2,3,4,5,6,7] # spines

        fig,(ax,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(1, 7, sharey=False)

        # now zoom in each of the subplots
        ax.set_xlim([ x[0],x[1]])
        ax2.set_xlim([ x[1],x[2]])
        ax3.set_xlim([ x[2],x[3]])
        ax4.set_xlim([ x[3],x[4]])
        ax5.set_xlim([ x[4],x[5]])
        ax6.set_xlim([ x[5],x[6]])
        #ax7.set_xlim([ x[6],x[7]])

        # set the x axis ticks
        for axx,xx in zip([ax,ax2,ax3,ax4,ax5,ax6,ax7],x[:-1]):
         axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ax7.xaxis.set_major_locator(ticker.FixedLocator([x[-6],x[-5]]))  # the last one

        # EDIT: add the labels to the rightmost spine
        for tick in ax7.yaxis.get_major_ticks():
         tick.label2On=True

        # stack the subplots together
        plt.subplots_adjust(wspace=0)
        '''
        a = random.randint(0,20)
        b = random.randint(0,20)
        c = random.randint(0,20)
        d = random.randint(0,20)
        e = random.randint(0,20)
        f = random.randint(0,20)
        g = random.randint(0,2)
        y1=[a,b,c,d,e,f,g]
        '''
        for elements in record:
            y1 = elements[3] + [elements[1]]
            # plot the same on all the subplots
            ax.plot(x,y1,'r-')#, x,y2,'b-')
            ax2.plot(x,y1,'r-')#, x,y2,'b-')
            ax3.plot(x,y1,'r-')#, x,y2,'b-')
            ax4.plot(x,y1,'r-')#, x,y2,'b-')
            ax5.plot(x,y1,'r-')#, x,y2,'b-')
            ax6.plot(x,y1,'r-')#, x,y2,'b-')
            ax7.plot(x,y1,'r-')#, x,y2,'b-')


            plt.pause(0.05)