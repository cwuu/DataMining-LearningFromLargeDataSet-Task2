import numpy as np

#31000 0.832 1.16, 27000 0.814 1.03, 25000 0.819 0.59,21000 0.808 0.50, 23000 0.8145 0.55, 22000 0.8145 0.54



#step size
R= 0.0005
#[0,1), exponential decay rates for the moment estimates
BETA1 = 0.99
BETA2 = 0.9993
#samples amount(as much as possible, and also consider the run time)
PAIRS = 27000
EPSILON = 1e-10
#gamma=1/(2*omega^2)
GAMMA = 20
#gusee just the seed, try change it to another number
RANDOM_STATE = 10
LEARN=50


#RBF (Radial basis function kernel)
#k(x,x')=exp(-||x-x'||2/(2*omega^2))=exp(-gamma*||x-x'||2)
def transform(image):
    random_state = np.random.RandomState(RANDOM_STATE)
    #Draw random samples from a Gaussian distribution.
    #weights 
    w = (np.sqrt(2 * GAMMA) *
               random_state.normal(size=(image.shape[1], PAIRS)))
               
    #Draw samples from a uniform distribution.
    #offset , uniform [0, 2*pi]
    b = random_state.uniform(0, 2 * np.pi, size=PAIRS)

    # projection
    #z(x)=sqrt(2)*cos(weight dot x + b)
    projection = np.dot(image, w) + b
    np.cos(projection, projection)
    projection = projection*(np.sqrt(2.) / np.sqrt(PAIRS))

    return projection

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    #set of data (label,image)=(y,X)
    # 2D NumPy array containing the original feature vectors
    image = np.zeros([len(value), len(value[0].split()) - 1])
    # 1D NumPy array containing the classifications of the training data
    label = np.zeros(len(value))

    # populate features and classifications
    for i in range(len(value)):
        tokens = value[i].split()
        label[i] = tokens[0]
        image[i] = tokens[1:]

    # project features into higher dimensional space
    image = transform(image)

    # Adam (works for both sprase and sense function)
    #inilial parameter vector
    w = np.zeros(image.shape[1])
    #initialize first moment vector
    m = np.ones(image.shape[1])
    #initialize second momemt vector
    v = np.ones(image.shape[1])
    #while the parameter vector havent coverged
    for t in range(image.shape[0]):
        if label[t] * np.dot(w, image[t, :]) < 1:
            #get gradients w.r.t stochastic objective ar timestep t
            g = -label[t] * image[t, :]
            #update biased first moment estimate
            m = BETA1 * m + (1. - BETA1) * g
            #update biased second raw moment estimate
            v = BETA2 * v + (1. - BETA2) * g ** 2
            #compute bias-corrected first moment estimate
            ms = m / (1. - BETA1 ** (t + 1.))
            #compute bias-corrected first moment estimate
            vs = v / (1. - BETA2 ** (t + 1.))
            #update parameters
            #Stochastic Gradient Descent classifier
            w = w - (LEARN / np.sqrt((t + 1.))) * ms / (np.sqrt(vs) + EPSILON)
            w = w * min(1, 1 / (np.sqrt(R) * np.linalg.norm(w, 2)))
    yield "key", w


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    w = np.array(values).mean(axis=0)
    w_final = w.T
    yield w_final
