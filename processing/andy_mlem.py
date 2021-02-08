import numpy as np
import time
from scipy import ndimage
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import linregress, moment


def computeMLEM(sysMat, counts, nIter=10, sens_j=None, ):
    ''' this function computes iterations of MLEM
        it returns the image after nIter iterations

        sysMat is the system matrix, it should have shape:
            (n_measurements, n_pixels)
            it can be either a 2D numpy array, numpy matrix, or scipy sparse
            matrix
        counts is an array of shape (n_measurements) that contains the number
            of observed counts per detector bin

        sens_j is the sensitivity for each image pixel
            is this is None, uniform sensitivity is assumed
    '''

    nPix = sysMat.shape[1]

    if sens_j is None:
        sens_j = np.ones(nPix)

    lamb = np.ones(nPix)
    lamb_previous = np.zeros(nPix)
    diff = 10 ** 6 * np.ones(nPix)
    outSum = np.zeros(nPix)
    iIter = 0

    if counts is None:
        counts = np.ones(sysMat.shape[0])

    backProj = (sysMat.T.dot(counts))

    print
    'Computing Iterations'
    t1 = time.time()
    # for iIter in range(nIter):
    while diff.sum() > 0.001 * counts.sum() + 100:
        sumKlamb = sysMat.dot(lamb)
        outSum = (sysMat * counts[:, np.newaxis]).T.dot(1 / sumKlamb)
        lamb = lamb * outSum / sens_j
        lamb = lamb * correction

        if iIter > 5:
            # lamb = lamb.reshape(51,51)
            lamb = lamb.reshape(61, 61)

            lamb = ndimage.gaussian_filter(lamb, 1)
            # lamb = ndimage.median_filter(lamb,3)
            # lamb = lamb.reshape(51*51)
            lamb = lamb.reshape(61 * 61)

        print
        'Iteration %d, time: %f sec' % (iIter, time.time() - t1)
        diff = abs(lamb - lamb_previous)
        print
        diff.sum()
        lamb_previous = lamb
        iIter += 1
    return lamb


###############################################


def plot_2D(result):
    plt.close()
    # result = result.reshape(51,51)
    result = result.reshape(61, 61)
    result = np.fliplr(result)
    # im = plt.plot(result)
    # im = plt.imshow(result, cmap=plt.cm.jet, interpolation='nearest', extent = [-25,25,-25,25])
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter

    im = plt.imshow(result, cmap=plt.cm.jet, interpolation='nearest', extent=[-30, 30, -30, 30])

    im2 = plt.plot((-edge_location, -edge_location), (-30, 30), 'w-')
    im2 = plt.plot((-edge_location + 19, -edge_location + 19), (-30, 30), 'w-')

    moment2D = moment(result, moment=2)
    # np.savetxt('moment.csv',moment2D,delimiter=',')
    # print 'moment = ', moment2D

    max_index = np.argmax(moment2D)
    X = range(-30, 31)[max_index:max_index + 10]
    Y = moment2D[max_index:max_index + 10]

    (aCoeff, bCoeff, rVal, pVal, stdError) = linregress(X, Y)  # y = ax + b
    print
    'aCoeff = ', aCoeff
    print
    'bCoeff = ', bCoeff
    print
    'rVal = ', rVal
    print
    ' '

    # percent60 = (0.60*plot1D.max() - bCoeff)/aCoeff
    # print '60percent = ',percent60

    Prange = 18.7 - edge_location
    PG = (aCoeff * Prange + bCoeff) / (moment2D.max())

    # print
    # 'PG% = ', PG

    plt.xlabel('[mm]')
    plt.ylabel('[mm]')
    plt.xlim(-26, 26)
    plt.ylim(-5, 20)
    plt.minorticks_on()
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    # plt.xticks(range(-30,30))
    # plt.grid(True,which='both',linestyle='-',alpha=.5)
    # plt.grid(True,which='major',color='g',linestyle='-')
    cbar = plt.colorbar()


# plt.show()
# plt.savefig('lined'+str(edge_location)+'.png',bbox_inches='tight')
# plt.savefig(filename[:6]+'2D'+'.png',bbox_inches='tight')

def plot_1D(result):
    plt.close()
    result = result.reshape(61, 61)
    result = np.fliplr(result)
    plot1D = np.zeros(len(result[0]))
    max, max_row = 0, 0
    for i in range(len(result)):
        if result[i].sum() > max:
            max = result[i].sum()
            max_row = i
    x = range(-30, 31)
    for i in range(-1, 2):
        plot1D += result[max_row + i]

    fig, ax1 = plt.subplots()
    # ax1.plot(x,result[max_row],'r',linewidth=2)
    ax1.plot(x, plot1D, 'r', linewidth=2)
    # im = plt.plot(x,result[max_row])
    # im = plt.plot(x,plot1D)
    plt.xlim((-25, 25))
    # plt.ylim((0,125000))
    plt.xlabel('[mm]')
    plt.ylabel('Counts')

    # np.save(str(edge_location)+'plot',plot1D) #saves plot data as .npy
    # np.savetxt(str(edge_location)+'plot.csv',plot1D,delimiter=',')

    ## Here I do some linear regression on the PG falloff to determine the 60% point
    X = np.zeros(13)
    Y = np.zeros(13)
    max_index = np.argmax(plot1D)
    X = range(-30, 31)[max_index:max_index + 13]
    Y = plot1D[max_index:max_index + 13]

    (aCoeff, bCoeff, rVal, pVal, stdError) = linregress(X, Y)  # y = ax + b
    print
    'aCoeff = ', aCoeff
    print
    'bCoeff = ', bCoeff
    print
    'rVal = ', rVal
    print
    ' '

    # percent60 = (0.60*plot1D.max() - bCoeff)/aCoeff
    # print '60percent = ',percent60

    Prange = 18.7 - edge_location
    PG = (aCoeff * Prange + bCoeff) / (plot1D.max())

    print
    'PG% = ', PG

    ax2 = ax1.twinx()
    braggX, braggY = np.load('BraggPeakInPMMA.npy')
    BraggMax = braggY.max()
    braggY = braggY / BraggMax
    x2 = np.arange(100, -100, -.03)
    y2 = np.zeros(len(x2))
    for i in range(0, len(braggY)):
        y2[3351 + i] = y2[3351 + i] + braggY[i]
    x2 = x2 + 30  # I'm not sure how I got shifted by 30...
    ax2.plot(x2 - edge_location, y2, 'k--', linewidth=3)
    plt.xlim((-30, 30))
    plt.ylabel('Dose')


# im2 = plt.axvline(edge_location,color='r')
# im2 = plt.axvline(edge_location-19,color='r')
# plt.show()
# plt.savefig(filename[:6]+'1D'+'.png',bbox_inches='tight')

###############################################

matrix = np.load('padded_adjusted_system_matrix_2D_21Nov15.npy')
# matrix = np.load('padded_1D_matrix_28Jan16.npy')

# filename = '2015-10-14-06-41-25_1e8proton_5.csv'
filename = '2015-10-14-04-26-42_counts.csv'
counts = np.loadtxt(filename, delimiter=',', unpack=True)[0:2304]
edge_location = 22
minutes = 10
current = 1.12 * 10 ** 10  # 0.03nA
counts_adjust = 10 ** 9 / (minutes * current)
'''
0.03nA, 50 MeV beam (= 1.12*10^10 protons/min)
2015-10-14-04-00-40: -20mm, 10 minutes 	1 edge_location = 23
2015-10-14-04-26-42: -19mm, 5 min		2				22
2015-10-14-04-38-25: -18mm, 5 min		3				21
2015-10-14-04-50-06: -17mm, 5 min		4				20
2015-10-14-05-07-25: -16mm, 5 min		5				19
2015-10-14-05-20-56: -15mm, 5 min		6				18
2015-10-14-05-35-38: -14mm, 5 min		7				17
2015-10-14-05-48-42: -13mm, 5 min		8				16
2015-10-14-06-04-12: -12mm, 5 min		9				15
2015-10-14-06-28-22: -11mm, 5 min		10				14
2015-10-14-06-41-25: -10mm, 5 min		11				13
2015-10-14-08-07-03: -9mm, 15 min, 0.08nA 12			12
2015-10-14-08-28-08: -8mm, 15 min, 0.08nA 13			11
2015-10-14-09-01-02: -7mm, 15 min, 0.08nA 14			10
2015-10-14-09-24-04: -6mm, 15 min, 0.08nA 15			09
2015-10-14-10-24-44: -5mm, 15 min, 0.08nA 16			08
'''

# counts = np.loadtxt('2015-10-13-05-08-20_counts.csv',delimiter=',',unpack=True)[0:2304] #point
# counts = np.loadtxt('2015-10-13-08-21-47_counts.csv',delimiter=',',unpack=True)[0:2304]
# correction = np.load('sensitivity_correction_21Nov15.npy')
correction = np.load('sensitivity_correction_17Dec15_4.npy')
# correction = np.ones(61*61)
# correction = np.load('correction_1D_19Nov15.npy')

matrix = matrix.T
matrix = gaussian_filter(matrix, sigma=1)

result = computeMLEM(matrix, counts, nIter=45)
plot_2D(result)