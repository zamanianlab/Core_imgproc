import argparse
import string
import itertools
import cv2
import numpy as np
import csv
import glob
from pathlib import Path
from PIL import Image, ImageDraw
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from skimage.morphology import disk, binary_closing
from skimage import filters
from matplotlib import cm
from scipy import ndimage
from datetime import datetime


def create_plate(plate):
    '''
    Create a list with all the well names for a given plate format.
    '''
    # create a list of the possible well names (add for all plate formats)
    wells = []
    if plate == 96:
        rows = string.ascii_uppercase[:8]
        columns = list(range(1, 13))
        columns = [str(column).zfill(2) for column in columns]
        wells = list(itertools.product(rows, columns))
        wells = [''.join(well) for well in wells]
    # print(wells)

    return wells


def organize_arrays(input, output, work, plate, frames, rows, columns, reorganize):
    '''
    Create a list of lists where each internal list is all the paths to all the
    images for a given well, and the entire list is the plate.
    '''

    for well in plate:
        well_counter = 0
        print("Getting the paths for well {}".format(well))

        # initialize a list that will contain paths to each frame
        well_paths = []

        # append the path pointing to each frame from each well
        for frame in range(1, frames + 1):
            data_dir = Path.home().joinpath(input)
            plate_name = data_dir.parts[-1]
            plate_name = plate_name.split('_')[0]
            frame_path = data_dir.joinpath("TimePoint_" + str(frame),
                                           plate_name + "_" + well + ".TIF")
            well_paths.append(frame_path)

        # read the TIFFs at the end of each path into a np array; perform
        # various operations on the array

        # write the first frame for dx; the TIF files are 16-bit images
        first_frame = cv2.imread(str(well_paths[0]), cv2.IMREAD_ANYDEPTH)
        if first_frame is None:
            print("Well {} not found. Moving to next well.".format(well))
            well_counter += 1
        else:
            work_dir = Path.home().joinpath(work)
            plate_name = work_dir.parts[-1]
            plate_name = plate_name.split('_')[0]
            work_dir.joinpath(well, 'img').mkdir(parents=True, exist_ok=True)
            outpath = work_dir.joinpath(
                well, 'img', plate_name + "_" + well + '_orig' + ".png")
            cv2.imwrite(str(outpath), first_frame)

            # initialize an array with the correct shape of the final array
            height, width = first_frame.shape
            well_array = np.zeros((frames, height, width))

            # read images from well_paths
            print("Reading images for well {}".format(well))
            timepoint_counter = 0
            for frame in well_paths:
                print("Reading Timepoint_{}".format(timepoint_counter + 1))
                image = cv2.imread(str(frame), cv2.IMREAD_ANYDEPTH)
                well_array[timepoint_counter] = image

                if reorganize:
                    timepoint = str(timepoint_counter + 1).zfill(2)
                    work_dir.joinpath(well, 'vid').mkdir(
                        parents=True, exist_ok=True)
                    outpath = work_dir.joinpath(
                        well, 'vid',
                        plate_name + "_" + well + "_" + timepoint + ".tiff")
                    cv2.imwrite(str(outpath), image)
                timepoint_counter += 1

            # run flow on the well
            total_sum, sum_img = dense_flow(
                well,
                well_array,
                input,
                output,
                work)

            # segment the worms
            normalization_factor, sobel, blur, bin = segment_worms(
                well,
                well_array,
                input,
                output,
                work)

            # wrap_up
            wrap_up(
                well,
                total_sum,
                normalization_factor,
                input,
                output)

            well_counter += 1

            # add to the dict with the well as the key and the array as the value
            # vid_dict[well] = well_array

            # saving as 16 bit AVI not currently working
            # fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            # outvid = dir.joinpath(well, plate_name + "_" + well + ".avi")
            # out = cv2.VideoWriter(str(outvid), fourcc, 4, (height,  width), False)
            # for frame in well_array:
            #     frame = frame.astype('uint8')
            #     out.write(frame)

    # return vid_dict


def dense_flow(well, well_array, input, output, work):
    '''
    Uses Farneback's algorithm to calculate optical flow for each well. To get
    a single motility values, the magnitude of the flow is summed across each
    frame, and then again for the entire array.
    '''

    start_time = datetime.now()
    print("Starting optical flow analysis on {}.".format(well))

    length, width, height = well_array.shape

    # initialize emtpy array of video length minus one (the length of the dense flow output)
    all_mag = np.zeros((length - 1, height, width))
    count = 0
    frame1 = well_array[count]

    while(1):
        if count < length - 1:
            frame1 = well_array[count].astype('uint16')
            frame2 = well_array[count + 1].astype('uint16')

            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3,
                                                30, 3, 5, 1.1, 0)
            mag = np.sqrt(np.square(flow[..., 0]) + np.square(flow[..., 1]))

            frame1 = frame2

            # replace proper frame with the magnitude of the flow between prvs and next frames
            all_mag[count] = mag
            count += 1

        else:
            break

    # calculate total flow across the entire array
    sum_img = np.sum(all_mag, axis=0)
    total_sum = np.sum(sum_img)

    # write out the dx flow image
    work_dir = Path.home().joinpath(work)
    plate_name = work_dir.parts[-1]
    work_dir.joinpath(well, 'img').mkdir(parents=True, exist_ok=True)
    outpath = work_dir.joinpath(well, 'img',
                                plate_name + "_" + well + '_flow' + ".png")

    # write to png
    # if there is not a single saturated pixel (low flow), set one to 255 in order to prevent rescaling
    pixel_max = np.amax(sum_img)

    if pixel_max < 255:
        print("Max flow is {}. Rescaling".format(pixel_max))
        sum_img = sum_img * 0.8
        sum_img[0, 0] = 255
    # if there are saturated pixels (high flow), adjust everything > 255 to prevent rescaling
    elif pixel_max > 255:
        print("Max flow is {}. Rescaling".format(pixel_max))
        sum_img = sum_img * 0.8
        sum_img[sum_img > 255] = 255
    else:
        print("Something went wrong.")

    sum_blur = ndimage.filters.gaussian_filter(sum_img, 1.5)
    cv2.imwrite(str(outpath), sum_blur.astype('uint8'))

    print("Optical flow = {0}. Analysis took {1}".format(
        total_sum, datetime.now() - start_time))

    return total_sum, sum_img


def segment_worms(well, well_array, input, output, work):
    '''
    Segments worms to use for downstream normalization.
    '''

    start_time = datetime.now()
    print("Starting normalization calculation for {}.".format(well))

    array = well_array

    print("Segmenting 5th frame...")

    # sobel edge detection
    sobel = filters.sobel(array[4])

    # gaussian blur
    blur = ndimage.filters.gaussian_filter(sobel, 1.5)

    # set threshold, make binary
    threshold = threshold_otsu(blur)
    binary = blur > threshold

    # create a disk mask for 2X images
    def create_circular_mask(h, w, center=None, radius=None):
        if center is None:  # make the center the center of the image
            center = (int(w/2), int(h/2))
        if radius is None:  # make the radius the size of the image
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = dist_from_center <= radius
        return mask

    mask = create_circular_mask(2048, 2048, radius=975)

    # mask the binary image
    binary = binary * mask

    # dilate, fill holes, and size filter
    selem = disk(30)
    dilated = binary_closing(binary, selem)
    # filled = ndimage.binary_fill_holes(dilated).astype('uint8')
    nb_components, labelled_image, stats, centroids = cv2.connectedComponentsWithStats(dilated.astype('uint8'), connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    # empirically derived minimum size
    min_size = 25000

    filtered = np.zeros((labelled_image.shape))
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            filtered[labelled_image == i + 1] = 255

    work_dir = Path.home().joinpath(work)
    plate_name = work_dir.parts[-1]
    outpath = work_dir.joinpath(well, 'img')

    sobel_png = work_dir.joinpath(outpath,
                                  plate_name + "_" + well + '_edge' + ".png")
    cv2.imwrite(str(sobel_png), sobel.astype('uint8'))

    blur_png = work_dir.joinpath(outpath,
                                 plate_name + "_" + well + '_blur' + ".png")
    cv2.imwrite(str(blur_png), blur.astype('uint8'))

    bin_png = work_dir.joinpath(outpath,
                                plate_name + "_" + well + '_binary' + ".png")
    cv2.imwrite(str(bin_png), binary * 255)

    # fill_png = work_dir.joinpath(outpath,
    #                              plate_name + "_" + well + '_filled' + ".png")
    # cv2.imwrite(str(fill_png), filled * 255)

    filtered_png = work_dir.joinpath(outpath,
                                     plate_name + "_" + well + '_filtered' + ".png")
    cv2.imwrite(str(filtered_png), filtered * 255)

    print("Calculating normalization factor.")

    # the area is the sum of all the white pixels (1.0)
    area = np.sum(binary)
    print("Normalization factor calculation completed. Calculation took {}".
          format(datetime.now() - start_time))

    return area, sobel, blur, binary


def wrap_up(well, motility, normalization_factor, input, output):
    '''
    Takes dictionaries of values and writes them to a CSV.
    '''

    out_dir = Path.home().joinpath(output)
    plate_name = out_dir.parts[-1]
    outpath = out_dir.joinpath(plate_name + '_data' + ".csv")

    with open(str(outpath), 'a') as of:
        writer = csv.writer(of, delimiter=',')
        writer.writerow([well, motility, normalization_factor])


def thumbnails(rows, cols, input, output, work):
    '''
    Takes a dict that contains a video, rescales into thumbnails, and pastes
    into the structure of the plate.
    '''

    print("Generating diagnostic thumbnails.")
    # get the paths to all the dx images
    work_dir = Path.home().joinpath(work)

    orig_search = str(work_dir.joinpath('**/img/*_orig.png'))
    orig_paths = glob.glob(orig_search)

    flow_search = str(work_dir.joinpath('**/img/*_flow.png'))
    flow_paths = glob.glob(flow_search)

    binary_search = str(work_dir.joinpath('**/img/*_filtered.png'))
    binary_paths = glob.glob(binary_search)

    # rescale images and store them in a dict with well/image
    def create_thumbs(paths, type):
        thumb_dict = {}
        for path in paths:
            well = path.split('/')[-3]
            image = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
            # rescale the image with anti-aliasing
            rescaled = rescale(image, 0.125, anti_aliasing=True, clip=False)
            # normalize to 0-255
            if type == 'flow':
                rescaled[0, 0] = 1
            rescaled_norm = cv2.normalize(src=rescaled, dst=None, alpha=0,
                                          beta=255, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_8U)
            thumb_dict[well] = rescaled_norm
        return thumb_dict

    # generate thumbnails for all dx images
    # returns a dict with well[thumb]
    print("Creating OG thumbnails.")
    orig_thumbs = create_thumbs(orig_paths, 'orig')
    print("Creating colorized flow thumbnails.")
    flow_thumbs = create_thumbs(flow_paths, 'flow')
    print("Creating segmented thumbnails.")
    binary_thumbs = create_thumbs(binary_paths, 'filtered')

    # 0.125 of the 4X ImageXpress image is 256 x 256 pixels
    height = rows * 256
    width = cols * 256

    # stitch the thumbnails together with the proper plate dimensions and save
    def create_plate_thumbnail(thumbs, type):
        # new blank image with gridlines
        new_im = Image.new('L', (width, height))

        for well, thumb in thumbs.items():
            # row letters can be converted to integers with ord()
            # and then rescaled by subtracting a constant
            row = int(ord(well[:1]) - 64)
            col = int(well[1:].strip())
            new_im.paste(Image.fromarray(thumb),
                         ((col - 1) * 256, (row - 1) * 256))

        if type == 'flow':
            # apply a colormap if it's a flow image
            new_im = np.asarray(new_im) / 255
            new_im = Image.fromarray(np.uint8(cm.inferno(new_im) * 255))
            draw = ImageDraw.Draw(new_im)
            for col_line in range(0, width + 256, 256):
                draw.line((col_line, 0, col_line, height), fill=255, width=10)
            for row_line in range(0, height + 256, 256):
                draw.line((0, row_line, width, row_line), fill=255, width=10)
        else:
            draw = ImageDraw.Draw(new_im)
            for col_line in range(0, width + 256, 256):
                draw.line((col_line, 0, col_line, height), fill=64, width=10)
            for row_line in range(0, height + 256, 256):
                draw.line((0, row_line, width, row_line), fill=64, width=10)

        out_dir = Path.home().joinpath(output)
        out_dir.joinpath('thumbs').mkdir(parents=True, exist_ok=True)
        plate_name = out_dir.parts[-1]
        outfile = out_dir.joinpath('thumbs', plate_name + '_' + type + ".png")

        new_im.save(outfile)

    create_plate_thumbnail(orig_thumbs, 'orig')
    create_plate_thumbnail(flow_thumbs, 'flow')
    create_plate_thumbnail(binary_thumbs, 'binary')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='To be determined...')

    # required positional arguments
    parser.add_argument('input_directory',
                        help='A path to a directory containing subdirectories \
                        filled with TIFF files (i.e. 20201118-p01-MZ_172/     \
                        TimePoint_1, etc.)')

    parser.add_argument('output_directory',
                        help='A path to the output directory.')

    parser.add_argument('work_directory',
                        help='A path to the work directory.')

    parser.add_argument('rows', type=int,
                        help='The number of rows in the imaging plate.')

    parser.add_argument('columns', type=int,
                        help='The number of columns in the imaging plate.')

    parser.add_argument('time_points', type=int,
                        help='The number of frames recorded.')

    # optional flags
    parser.add_argument('--reorganize', dest='reorganize', action='store_true',
                        default=False,
                        help='Invoke if you want to save the TIFF files       \
                        organized by well instead of time point (default is   \
                        to not reorganize).')

    args = parser.parse_args()

    # create the plate shape
    plate_format = args.rows * args.columns

    # create a list of all the possible wells in the plate
    plate = create_plate(plate_format)

    vid_dict = organize_arrays(
        args.input_directory,
        args.output_directory,
        args.work_directory,
        plate,
        args.time_points,
        args.rows,
        args.columns,
        args.reorganize)

    thumbnails(
        args.rows,
        args.columns,
        args.input_directory,
        args.output_directory,
        args.work_directory)
