import os
import cv2
import csv
import argparse
import warnings
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
from utils.biometrics import biometrics
from utils.image_folder import make_dataset
from utils.utils import read_indices, write_txt
from siamiris_embedding import siamiris_embedding

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Force to use CPU
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = ''

def iris_recognition(args):
    # Get model ID:
    if args.backbone == 'resnet50':
        SiamIris_ID = 'SiamIris R50-iris'
    elif args.backbone == 'mobilenetv2':
        SiamIris_ID = 'SiamIris MN2-iris'
    # Load SiamIris model
    SiamIris = siamiris_embedding(args.backbone)
    print('\n\n{} loaded for Iris Recognition'.format(SiamIris_ID))

    # Make opt_dir
    opt_dir = os.path.join(args.opt_dir, args.split + '_results/')
    if not os.path.exists(opt_dir):
        os.makedirs(opt_dir)

    # Load dataset
    dataset = make_dataset(args.dataset)
    N_images = len(dataset)
    print('Dataset size: {:>6,d}'.format(len(dataset)))

    # Read mated and non-mated indices
    list_mated = os.path.join(args.list_dir, args.split + '_mated.txt')
    list_non_m = os.path.join(args.list_dir, args.split + '_non_mated.txt')
    ind_m, Nm = read_indices(list_mated)
    ind_n, Nn = read_indices(list_non_m)
    ind_test = np.unique(np.array(ind_m).flatten())
    if args.reduced_test:
        np.random.shuffle(ind_n)
        ind_n = ind_n[:Nm]
        Nn = len(ind_n)
        #np.random.shuffle(ind_m)
        #ind_m = ind_m[:1000]
        #Nm = len(ind_m)
    print('Test Images : {:>6,d}'.format(len(ind_test)))
    print('Mated Comparisons: {:>7,d}'.format(Nm))
    print('Non-M Comparisons: {:>7,d}'.format(Nn))
    print(' ')

    # Get embeddins for the test set
    emb = np.zeros((len(dataset), SiamIris.embedding_size), dtype='float')
    for ind in tqdm(ind_test, desc='Getting embeddings'):
        # Read iris image:
        name = dataset[ind]
        iris = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
        # Image on correct format:
        iris = SiamIris.process_image(iris)
        # Get embedding:
        emb[ind] = SiamIris.get_embedding(iris).flatten()
    print(' ')

    # Perform Mated Comparissons
    dist_mated = []
    txt_mated = os.path.join(opt_dir, 'dist_mated.txt')
    for ind_i, ind_j in tqdm(ind_m, desc='Mated Comparisons'):
        dist = SiamIris.compare(emb[ind_i], emb[ind_j])
        dist_mated.append(dist)
    write_txt(dist_mated, txt_mated)
    dist_mated = np.array(dist_mated)
    print(' ')

    # Perform Non-Mated Comparissons
    dist_non_m = []
    txt_non_m = os.path.join(opt_dir, 'dist_non_mated.txt')
    for ind_i, ind_j in tqdm(ind_n, desc='Non-M Comparisons'):
        dist = SiamIris.compare(emb[ind_i], emb[ind_j])
        dist_non_m.append(dist)
    write_txt(dist_non_m, txt_non_m)
    dist_non_m = np.array(dist_non_m)
    print(' ')

    # Perform Biometric tests
    biom = biometrics(dist_mated, dist_non_m)
    d_prime = biom.d_prime()
    EER, EER_th = biom.get_EER()
    FAR_a, FRR_a, th_a = biom.get_FRR_at(10.0)
    FAR_b, FRR_b, th_b = biom.get_FRR_at(5.0)
    FAR_c, FRR_c, th_c = biom.get_FRR_at(1.0)

    # Print Biometric Scores
    print('Biometric Scores')
    print("d' = {:0.4f}".format(d_prime))
    print('at EERt={:6.4f} : FMR={:6.3f}%, FNMR={:6.3f}%'.format(EER_th, EER, EER))
    print('at  th={:6.4f}  : FMR={:6.3f}%, FNMR={:6.3f}%'.format(th_a, FAR_a, FRR_a))
    print('at  th={:6.4f}  : FMR={:6.3f}%, FNMR={:6.3f}%'.format(th_b, FAR_b, FRR_b))
    print('at  th={:6.4f}  : FMR={:6.3f}%, FNMR={:6.3f}%'.format(th_c, FAR_c, FRR_c))
    print(' ')

    # Save Biometric indices
    results_csv = os.path.join(opt_dir, 'results.csv')
    with open(results_csv, 'w') as f:
        write = csv.writer(f)
        write.writerow(["d'", d_prime])
        write.writerow(["Description","threshold","FMR [%]","FNMR [%]"])
        write.writerow(["EER", EER_th, EER, EER])
        write.writerow(["FMR@FNMR=10%", th_a, FAR_a, FRR_a])
        write.writerow(["FMR@FNMR=5.0%", th_b, FAR_b, FRR_b])
        write.writerow(["FMR@FNMR=1.0%" , th_c, FAR_c, FRR_c])

    # Show histograms
    opt_plots = os.path.join(opt_dir, 'Hist_and_DET.svg')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9,4))
    kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40, ec="k")
    ax1.hist(dist_mated, **kwargs)
    ax1.hist(dist_non_m, **kwargs);
    ax1.legend(['mated','non-mated'])
    ax1.set_title("{} (d'={:0.3f})".format(SiamIris_ID, d_prime))
    ax1.set_xlabel('{} distance'.format(SiamIris.distance))
    ax1.set_ylabel('Normalized Frequency')
    #ax1.axis((-0.05,1.05,0,45))

    # Plot DET curve
    FAR, FRR, all_th = biom.for_plots()
    xylims = np.array([1e-4, 5e-1, 1e-4, 5e-1])
    ticks = np.array([1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 4e-1])
    tick_labels = np.array(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "40"])
    ax2.plot(FAR/100, FRR/100, label='DET')
    ax2.scatter(EER/100, EER/100, c='g', label='EER')
    ax2.plot(xylims[:2],xylims[:2], c='k', label='EE', linestyle='--', linewidth=0.5)
    ax2.set_title("{} DET curve (EER={:0.3f}%)".format(SiamIris_ID, EER))
    ax2.set_xlabel('False Match Rate (FMR) [%]')
    ax2.set_ylabel('False Non-Match Rate (FNMR) [%]')
    ax2.set_xscale("function", functions=(norm.ppf, norm.cdf))
    ax2.set_yscale("function", functions=(norm.ppf, norm.cdf))
    ax2.axis(xylims)
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    ax2.set_xticklabels(tick_labels)
    ax2.set_yticklabels(tick_labels)
    ax2.grid(linestyle = '--')
    plt.savefig(opt_plots)
    if not args.no_plot:
        plt.show()

    # Compile opt metrics dictionary
    IR_metrics = dict()
    IR_metrics['d_prime'] = d_prime
    IR_metrics['EER'] = EER
    IR_metrics['EER_th'] = EER_th
    IR_metrics['FMR@FNMR=0.1%'] = FAR_a
    IR_metrics['th@FNMR=0.1%']  = th_a
    IR_metrics['FMR@FNMR=1.0%'] = FAR_b
    IR_metrics['th@FNMR=1.0%']  = th_b
    IR_metrics['FMR@FNMR=10%']  = FAR_c
    IR_metrics['th@FNMR=10%']   = th_c

    return IR_metrics

if __name__ == '__main__':
    # Get test argumets
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--backbone', default='resnet50')
    parser.add_argument('-o', '--opt_dir', default='./results/')
    parser.add_argument('-d', '--dataset', default='/home/ubuntu/Datasets/SiamIris-LG4000-LR/iris_png/')
    parser.add_argument('-l', '--list_dir', default='./ND-LG4000-LR/lists_comparisons/')
    parser.add_argument('-s', '--split', default='test', help='Evaluate on train, test or val split.')
    parser.add_argument('-r', '--reduced_test', action='store_true', help='Use less non-mated comparisons.')
    parser.add_argument('-n', '--no_plot', action='store_true')
    args = parser.parse_args()

    # Run Iris Recognition test
    _ = iris_recognition(args)
