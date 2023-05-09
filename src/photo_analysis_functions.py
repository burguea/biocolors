# Contains functions for the photo analysis notebook
import imageio
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from tqdm import tqdm


def make_displayable(image, scaling=4):
    """Normalizes the 12 bit image to 0-1
    Inputs:
        Image - np array of 12 bit image
        scaling - Multiplier for all values in case the image is too dark
    Outputs:
        displayable_image - np array of 0-1 values image"""
    image[image < 0] = 0
    displayable_image = image / 2**14 * scaling
    displayable_image[displayable_image > 1] = 1

    return displayable_image

def white_balance(img, Rscale=1, Gscale=1, Bscale=1):
    """Scales channels to achieve white baance
    Inputs:
        Image - np array of 12 bit image
        Rscale, Gscale, Bscale - Multipliers for each channel
    Outputs:
        Image - np array image"""
    image = img.copy()
    image[:, :, 0] = image[:, :, 0] * Rscale
    image[:, :, 1] = image[:, :, 1] * Gscale
    image[:, :, 2] = image[:, :, 2] * Bscale

    return image
    
    

def reduce_colors(image, ncolors):
    """Reduces the image to its main n colors
    Inputs:
        Image - np array of 12 bit image
        ncolors - (int) number of colors to reduce the image to

    Outputs:
        main_colors - np array vector of the main RGB values of the image
        reduced_image - np array with the 12 bit reduced colors image
        labels - Label array indicating to which cluster each pixel belongs
    """

    # Flatten image to a vector of RGB triplets
    image_RGB_vector = image.reshape((-1, 3))
    # Allocate reduced image space
    reduced_image = image_RGB_vector.copy()
    # Find cluster centers
    clt = KMeans(n_clusters=ncolors, verbose=False).fit(image_RGB_vector)

    main_colors = clt.cluster_centers_

    # Make reduced colors image
    reduced_image = main_colors[clt.labels_].reshape(image.shape)

    return main_colors, reduced_image, clt.labels_


def find_optimal_clusters(imagefile, minn=4, maxn=10, plot=False):
    """Finds the optimal number of clusters for color quantization of an image

    Inputs:
        Imagefile -  path to image file
        maxn - Maximum number of clusters to try
        plot - Boolean to plot clusters vs error

    Outputs:
        optN  - Optimal number of clusters as per the knee criteria"""

    # Read image and convert to a vector of RGB values
    raw_photo = imageio.v2.imread(imagefile)
    # image_RGB_vector = raw_photo.reshape((-1, 3))
    errs = np.empty(maxn - minn)
    for q in tqdm(range(minn, maxn)):
        _, reduced_photo, _ = reduce_colors(raw_photo, q)
        # clt = KMeans(n_clusters=q, verbose=False).fit(image_RGB_vector)
        # errs[q-1] = clt.inertia_
        errs[q - minn] = np.mean(np.abs((raw_photo - reduced_photo)))

    kneedle = KneeLocator(
        np.arange(minn, maxn) + 1,
        errs,
        S=1,
        curve="convex",
        direction="decreasing",
        online=True,
    )
    if plot:
        kneedle.plot_knee()
    return kneedle


def cluster_variation(rawphoto, main_colors, pixel_labels):
    """Computes the average variation for the RGB values of the pixels within each of the clusters of a photo"""
    # iterate over clusters
    image_RGB_vector = rawphoto.reshape((-1, 3)).astype("float64")
    maxerrold=0
    for q, centroid in enumerate(main_colors):
        pixel_vector = image_RGB_vector[pixel_labels == q]
        # print(
        #     "\nCluster {}, {} pixels, central RGB:{}".format(
        #         q, pixel_vector.size, centroid
        #     )
        # )

        error_vector = np.abs(pixel_vector - centroid)
        avg_variation_rgb = np.mean(error_vector, axis=0)
        # std_rgb_vec = np.std(pixel_vector,axis=0)
        # print(
        #     "Avg variation from centroid:\n14 bits {}\n8 bits {}".format(
        #         avg_variation_rgb, avg_variation_rgb / 2**6
        #     )
        # )
        # print(
        #     "Avg variation across channels from centroid:\n14 bits {}\n8 bits {}. or {:.3}%\n".format(
        #         avg_variation_rgb.mean(), np.mean(avg_variation_rgb / 2**6), 100*avg_variation_rgb.mean()/2**14)
        #     )
        maxerr = 100*avg_variation_rgb.mean()/2**14
        if maxerr>maxerrold:
            maxerrold=maxerr
    print('\nGlobal max percent error: {:.3}'.format(maxerrold))
        # print('STD = {}'.format(std_rgb_vec))
        # errtot = np.array([0,0,0],dtype='float64')
        # for pixel in tqdm(pixel_vector):
        #     err_vec = np.abs(pixel-pixel_vector)
        #     errtot += np.mean(err_vec,axis=0)
        # print('Errtot={}'.format(errtot/pixel_vector.size))
    
    
    
# ===================START========================= #
def cluster_map_recoveryb(
    LU_LUT,
    dvec,
    nvec,
    image,
    color_labels,
    cluster_idx,
    num_pixels,
    th=0.05,
    plot=False,
    plot_subsample=10,
    saveplots=False,
):

    """Find the thickness and refractive index maps of the pixels within a cluster for a photo

    Input:
        LU_LUT - An array containing the RGB values for each thickness and RI.
        dvec  - A 1D array containing the thickness values of the LUT
        nvec - A 1D array containing the RI values of the LUT
        image - np array of raw RGB values
        color_labels - A 1 column array containg the cluster labels for each image pixel
        cluster_idx  - An integer indicating the number of cluster to analyze
        num_pixels  - Integer indicating the number of pixels in the cluster to pool
        th - Threshold for plotting only lowest values of cost function
        plot_subsample - Sampling interval for plotting (increases speed for dense cost functions)

    Output:
        nfound - Found n value for the cluster
        dfound - Found d value for the cluster
        RGB_foung - Found RGB value for the cluster
        cluster_cost - Pooled cost function for all pixels in the cluster
    """

    # Flatten image to a vector of RGB triplets
    image_RGB_vector = image.reshape((-1, 3))

    # Extract pixels belonging to cluster
    cluster_pixels = image_RGB_vector[color_labels == cluster_idx]

    # Allocate cost function map space
    cost_gray = np.empty((LU_LUT.shape[0], LU_LUT.shape[1]))

    # Choose a set of random pixels in the cluster
    # If cluster size is small, use all pixels in cluster

    if cluster_pixels.shape[0] < num_pixels:
        # print('Cluster {} is small, using all pixels'.format(cluster_idx))
        random_pixels_idxs = np.arange(cluster_pixels.shape[0])
    else:
        random_pixels_idxs = np.random.choice(
            cluster_pixels.shape[0], num_pixels, replace=False
        )

    # Compute cost function map for set of cluster pixels, add all of them up
    Cols = 4
    Rows = num_pixels // Cols
    Rows += num_pixels % Cols
    Position = range(1, num_pixels + 1)
    if plot:
        fig = plt.figure(figsize=(Cols * 5, Rows * 3))

    for k, pixel in tqdm(enumerate(cluster_pixels[random_pixels_idxs])):
        if plot:
            # add every single subplot to the figure with a for loop
            ax = fig.add_subplot(Rows, Cols, Position[k])

        # cost = np.abs(LU_LUT - pixel)  # Substract value of pixel from LUT
        cost = (LU_LUT - pixel) ** 2
        cost = cost.sum(axis=2)  # make grayscale
        cost_gray += cost  # Sum R+G+B to create 2D array, add previous

        cost /= cost.max()  # Normalize pixel cost
        cost[cost > th] = 1  # Threshold cutoff to display only lowest points
        pixidx = np.where(cost == cost.min())

        if plot:
            ax.pcolormesh(
                dvec[::plot_subsample],
                nvec[::plot_subsample],
                np.log10(cost[::plot_subsample, ::plot_subsample]),
                cmap="hot_r",
                norm=mpl.colors.LogNorm(),
                rasterized=True,
            )
            ax.axes.set_aspect(1000)
            ax.set_title(
                "pixel:{}\nd:{:.1f}, n:{:.3f}".format(
                    pixel, dvec[pixidx[1]][0], nvec[pixidx[0]][0]
                ),
                color=make_displayable(pixel, scaling=4),
            )

    # Locate best matching thickness and refractive index from LUT
    nfound = nvec[np.unravel_index(np.argmin(cost_gray), cost_gray.shape)[0]]
    dfound = dvec[np.unravel_index(np.argmin(cost_gray), cost_gray.shape)[1]]
    RGB_found = LU_LUT[np.unravel_index(np.argmin(cost_gray), cost_gray.shape)]
    print("Found RGB:", RGB_found)

    if plot:
        plt.tight_layout()
        if saveplots:
            plt.savefig("./other_plots/f4ac.pdf", bbox_inches="tight", transparent=True)

        plt.show()

    # Normalize and clip cost map
    cost_gray /= cost_gray.max()
    cost_gray[cost_gray > th] = 1

    # Plot confidence
    if plot:
        conf_n, conf_d = confidence(dvec, nvec, cost_gray)

        fig, ax = plt.subplots(figsize=(8, 8))
        divider = make_axes_locatable(ax)
        ax_x = divider.append_axes("top", 1.2, pad=1, sharex=ax)
        ax_y = divider.append_axes("right", 1.2, pad=1, sharey=ax)
        # ax.set_aspect(1.)

        e = ax.pcolormesh(
            dvec,
            nvec,
            cost_gray,
            cmap="hot_r",
            norm=mpl.colors.LogNorm(),
            rasterized=True,
        )
        # fig.colorbar(e, ax=ax)
        ax.set_title("Pooled cost.\nFound d:{:.1f}, n:{:.3f}".format(dfound, nfound))
        # ax.axvline(dfound,color='red')
        # ax.axhline(nfound,color='red')
        ax.scatter([dfound], [nfound], marker="o", s=400, c="red")

        ax_x.fill_between(dvec, conf_d, -0.01, color="C6", alpha=0.7)
        ax_x.set_title("confidence in thickness")
        ax_x.vlines(dfound, 0, conf_d.max(), color="red")
        print("Confidence in thickness minima: ", conf_d.max())

        ax_y.fill_betweenx(nvec, conf_n, -0.02, color="C9", alpha=0.7)
        ax_y.set_title("confidence in RI")
        print("Confidence in RI minima: ", conf_n.max())
        ax_y.hlines(nfound, 0, conf_n.max(), color="red")

        plt.tight_layout()
        if saveplots:
            plt.savefig("./other_plots/colorbar.pdf", bbox_inches="tight")

        plt.show()

    # return found values
    return nfound, dfound, RGB_found, cost_gray


# ===================END========================= #


def confidence(dvec, nvec, costmap):
    """Computes the confidence graphs on RI and thickness given a cost map.
    Works by obtaining the vector of minumum values along each axis,
    finding all the minima in the graphs and assigning a score to each one depending on their
    relative height values

    Inputs:
        costmap - The normalized costmap

    Outputs:
        conf_n - 1D vector with RI confidence
        conf_d - 1D vector with thickness confidence
    """
    conf_d = -np.log10(costmap.min(axis=0))
    conf_d /= conf_d.max()
    idpeaks, dpprop = find_peaks(
        conf_d, distance=200, height=0.1, width=10, rel_height=0.5
    )
    dheights = dpprop["peak_heights"]
    conf_d /= dheights.sum()

    conf_n = -np.log10(costmap.min(axis=1))
    conf_n /= conf_n.max()
    inpeaks, npprop = find_peaks(
        conf_n, distance=200, height=0.1, width=10, rel_height=0.5
    )
    nheights = npprop["peak_heights"]
    conf_n /= nheights.sum()

    return conf_n, conf_d


def plot_result_maps(
    dfound_vec,
    nfound_vec,
    found_RGB_vec,
    color_labels,
    raw_photo,
    reduced_photo,
    main_colors,
    colors_to_analyze,
    photo_file_name="noname",
    save=False,
):

    """Plot the reconstructed thickness and RI maps of the image"""

    print("Creating reconstructed RI and thickness maps images...")
    # Construct final results images
    dimage = dfound_vec[color_labels].reshape(raw_photo.shape[0], raw_photo.shape[1])
    nimage = nfound_vec[color_labels].reshape(raw_photo.shape[0], raw_photo.shape[1])
    recon_image = found_RGB_vec[color_labels].reshape(raw_photo.shape)
    
    np.save('./lastrun/dmap.npy',dimage)
    np.save('./lastrun/nmap.npy',nimage)
    np.save('./lastrun/recon_image.npy',recon_image)



    # Plot all results
    fig, ax = plt.subplots(2, 3, figsize=(1.618 * 14, 14))

    ax[0, 0].imshow(make_displayable(raw_photo))
    ax[0, 0].set_title("Original image")

    ax[0, 1].imshow(make_displayable(reduced_photo))
    ax[0, 1].set_title("Reduced image ({} colors)".format(main_colors.shape[0]))

    ax[0, 2].imshow(make_displayable(recon_image))
    ax[0, 2].set_title("Reconstructed photo")

    im1 = ax[1, 0].imshow(dimage, origin="lower")
    ax[1, 0].invert_yaxis()
    ax[1, 0].set_title("Recovered thickness map")
    bar1 = fig.colorbar(im1, ax=ax[1, 0])
    bar1.set_label("Thickness (nm)", rotation=270)

    im2 = ax[1, 1].imshow(nimage, origin="lower")
    ax[1, 1].invert_yaxis()
    ax[1, 1].set_title("Recovered refractive index map")
    bar2 = fig.colorbar(im2, ax=ax[1, 1])
    bar2.set_label("refractive index", rotation=270)

    for a in ax.ravel():
        a.axis("off")

    composition = np.empty(colors_to_analyze)
    for idx in range(colors_to_analyze):
        composition[idx] = color_labels[color_labels == idx].size / color_labels.size

    ax[1, 2].axis("on")
    ax[1, 2].set_title("Clusters")
    ax[1, 2].set_ylabel("Relative composition of image")

    clusters_rgb = 4 * found_RGB_vec / 2**14
    clusters_rgb[clusters_rgb > 1] = 1
    clusters_rgb[clusters_rgb < 0] = 0

    ax[1, 2].bar(range(colors_to_analyze), composition, color=clusters_rgb)
    for x, y, d, n in zip(
        range(colors_to_analyze), composition, dfound_vec, nfound_vec
    ):
        ax[1, 2].text(
            x - 0.25, y + 0.01, "d={:.0f}nm, n={:.3f}".format(d, n), rotation=90
        )
        # ax[1, 2].text(x - 0.5, y + 0.02, "n={:.3f}".format(n),rotation=90)

    plt.tight_layout()
    if save:
        plt.savefig(
            "./lastrun/{}_{}clusters.pdf".format(
                photo_file_name.split("/")[-1][:-4], colors_to_analyze
            ),
            dpi=300,
        )
        plt.show()


def my_cycle(seq):
    while seq:
        for element in seq:
            yield element


# =========== CLUSTERS ====== %
def plot_cluster_results(
    reduced_photo,
    main_colors,
    colors_to_analyze,
    dfound_vec,
    nfound_vec,
    dvec,
    nvec,
    cost_grayvec,
    plot_subsample,
    photo_file_name="noname",
    save=False,
):
    """Display the result for each cluster"""

    # Compute figure size, rows and columns

    Cols = 4
    Rows = colors_to_analyze // Cols
    Rows += colors_to_analyze % Cols
    Position = range(1, colors_to_analyze + 1)
    fig = plt.figure(figsize=(Cols * 5, Rows * 3))

    # Custom colors in 14 bit format
    colors_list = [
        [2**14, 0, 0],
        [0, 2**14, 0],
        [0, 0, 2**14],
        [2**14, 2**14, 0],
        [0, 2**14, 2**14],
        [2**14, 0, 2**14],
    ]
    color_cycle = my_cycle(colors_list)

    # Plot clusters over the image as solid color pixels
    for k in range(colors_to_analyze):
        cluster_photo = reduced_photo.copy()

        cluster_photo[
            reduced_photo != main_colors[k]
        ] = 0  # Make non-cluster pixels black
        cluster_photo[(cluster_photo != 0).all(-1)] = next(
            color_cycle
        )  # Color of cluster

        # add every single subplot to the figure with a for loop
        ax = fig.add_subplot(Rows, Cols, Position[k])
        # ax.imshow(make_displayable(raw_photo))

        ax.imshow(make_displayable(cluster_photo))
        # ax.imshow(make_displayable(cluster_photo),cmap=cm.tab10,interpolation='none') # Or whatever you want in the subplot

        ax.set_title(
            "Clstr {}, d={:.0f}nm, n={:.3f}".format(k, dfound_vec[k], nfound_vec[k])
        )

    plt.tight_layout()
    if save:
        plt.savefig(
            "./results/{}_{}clusters_clusters.pdf".format(
                photo_file_name.split("/")[-1][:-4], colors_to_analyze
            ),
            dpi=300,
        )
    plt.show()

    # Plot cost maps for each cluster

    fig = plt.figure(figsize=(Cols * 5, Rows * 3))

    for k in range(colors_to_analyze):
        # add every single subplot to the figure with a for loop
        costmap = cost_grayvec[k]
        conf_n, conf_d = confidence(dvec, nvec, costmap)

        ax = fig.add_subplot(Rows, Cols, Position[k])
        ax.pcolormesh(
            dvec[::plot_subsample],
            nvec[::plot_subsample],
            costmap[::plot_subsample, ::plot_subsample],
            cmap="hot_r",
            norm=mpl.colors.LogNorm(),
            rasterized=True,
        )
        ax.set_xlabel("thickness")
        ax.set_ylabel("RI")
        ax.set_title(
            "C{}, found d:{:.1f}, n:{:.3f}\nConfidence n:{:.1f}, d:{:.1f}".format(
                k, dfound_vec[k], nfound_vec[k], conf_n.max(),
            conf_d.max(),
            ))

    plt.tight_layout()
    if save:
        plt.savefig(
            "./lastrun/{}_{}clusters_cost_maps.pdf".format(
                photo_file_name.split("/")[-1][:-4], colors_to_analyze
            ),
            dpi=300,
        )
    plt.show()


#  # Plot confidence
#     if plot:
#         conf_n, conf_d = confidence(dvec, nvec, cost_gray)

#         fig, ax = plt.subplots(figsize=(8, 8))
#         divider = make_axes_locatable(ax)
#         ax_x = divider.append_axes("top", 1.2, pad=1, sharex=ax)
#         ax_y = divider.append_axes("right", 1.2, pad=1, sharey=ax)
#         # ax.set_aspect(1.)

#         e=ax.pcolormesh(
#             dvec,
#             nvec,
#             cost_gray,
#             cmap="hot_r",
#             norm=mpl.colors.LogNorm(),
#             rasterized=True,
#         )
#         fig.colorbar(e, ax=ax)
#         ax.set_title("Pooled cost.\nFound d:{:.1f}, n:{:.3f}".format(dfound, nfound))
#         # ax.axvline(dfound,color='red')
#         # ax.axhline(nfound,color='red')
#         ax.scatter([dfound], [nfound], marker="o", s=400, c="red")

#         ax_x.fill_between(dvec, conf_d, -0.01, color="C6", alpha=0.7)
#         ax_x.set_title("confidence in thickness")
#         ax_x.vlines(dfound, 0, conf_d.max(), color="red")
#         print("Confidence in thickness minima: ", conf_d.max())

#         ax_y.fill_betweenx(nvec, conf_n, -0.02, color="C9", alpha=0.7)
#         ax_y.set_title("confidence in RI")
#         print("Confidence in RI minima: ", conf_n.max())
#         ax_y.hlines(nfound, 0, conf_n.max(), color="red")

#         plt.tight_layout()
#         if saveplots:
#             plt.savefig("./other_plots/colorbar.pdf", bbox_inches="tight")

#         plt.show()

# ------------------- #


def full_photo_analysis(
    LU_LUT, dvec, nvec, num_pixels, imagefile, ncolors, th=0.01, plot_subsample=3
):
    """Complete analysis of an image"""

    import imageio
    import numpy as np

    # Reduce image colors and obtain clusters
    raw_photo = imageio.v2.imread(imagefile)
    print("Reducing image to {:} clusters".format(ncolors))
    main_colors, reduced_photo, color_labels = reduce_colors(raw_photo, ncolors)

    # Allocate recovered values vectors
    nfound_vec = np.empty(ncolors)
    dfound_vec = np.empty(ncolors)
    found_RGB_vec = np.empty(main_colors.shape)

    # --------- Parallel cluster processing -------- #
    from joblib import Parallel, delayed

    print(
        "Computing thickness and refractive index values for {} color clusters in parallel using {} pixels for each".format(
            ncolors, num_pixels
        )
    )
    # ---- Analyze the pixels of each cluster --- #
    result = Parallel(n_jobs=-1)(
        delayed(cluster_map_recoveryb)(
            LU_LUT,
            dvec,
            nvec,
            raw_photo,
            color_labels,
            cluster_idx,
            num_pixels,
            th,
            plot_subsample,
        )
        for cluster_idx in range(ncolors)
    )

    cost_grayvec = []
    # Extract results into final vectors
    for q, value in enumerate(result):
        nfound_vec[q] = value[0]
        dfound_vec[q] = value[1]
        found_RGB_vec[q] = value[2]
        cost_grayvec.append(value[3])
    # print(main_colors)
    # print(found_RGB_vec)

    # ----- Plot results ------#
    plot_result_maps(
        dfound_vec,
        nfound_vec,
        found_RGB_vec,
        color_labels,
        raw_photo,
        reduced_photo,
        main_colors,
        ncolors,
        photo_file_name=imagefile,
        save=True,
    )

    plot_cluster_results(
        reduced_photo,
        main_colors,
        ncolors,
        dfound_vec,
        nfound_vec,
        dvec,
        nvec,
        cost_grayvec,
        plot_subsample,
        photo_file_name=imagefile,
        save=True,
    )