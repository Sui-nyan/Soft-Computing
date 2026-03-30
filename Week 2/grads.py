import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent2(gradient, start, learn_rate, n_iter=50): 
    tolerance=1e-03
    vector = start
    yield vector
    for _ in range(n_iter):
        step = -learn_rate * gradient(vector)
        if np.all(np.abs(step) <= tolerance): #check that the difference is small enough
            break
        vector += step
        yield vector
    #return results

def C(x):
    return x*x

def gradC(v):
    return 2.0*v

def animate(i,gen):
	vec=next(gen)
	print(vec)
	plt.plot(vec[0],C(vec[0]),'ro')

def main():	
	x=np.arange(-10,11)
	y=C(x)
	plt.plot(x,y)
	start=np.array([10.])
	gen=gradient_descent2(gradC,start,0.2)
	ani = animation.FuncAnimation(plt.gcf(), animate,fargs=(gen,), interval=1000) 

	plt.show()


if __name__ == "__main__":
    main()